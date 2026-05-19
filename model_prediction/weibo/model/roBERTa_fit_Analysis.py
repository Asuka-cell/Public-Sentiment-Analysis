from __future__ import annotations

import glob
import json
import os
import sys
from typing import Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ASPECT_PRIORS = {
    "食品安全与卫生": 0.30,
    "预制菜与食材": 0.40,
    "价格与性价比": 0.40,
    "服务与体验": 0.45,
    "口味与品质": 0.45,
    "营销与公关": 0.45,
    "企业责任与诚信": 0.55,
    "用工与合规": 0.45,
    "其他": 0.50,
}


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def choose_mode_interactively() -> Optional[str]:
    prompt = "\n请选择运行模式:\n  1) 样本\n  2) 全量\n输入选项(1/2): "
    s = input(prompt).strip()
    if s == "1":
        return "sample"
    if s == "2":
        return "full"
    return None


def load_latest_doc_topics(model_dir: str) -> Optional[pd.DataFrame]:
    candidates = []
    output_dir_a = os.path.join(os.path.dirname(model_dir), "prediction", "bertopic_output")
    output_dir_b = os.path.join(model_dir, "bertopic_output")
    for output_dir in [output_dir_a, output_dir_b]:
        candidates.extend(glob.glob(os.path.join(output_dir, "doc_topics.csv")))
    if not candidates:
        return None
    latest_path = max(candidates, key=lambda p: os.path.getmtime(p))
    try:
        df = pd.read_csv(latest_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(latest_path)
    if "weibo_id" not in df.columns:
        return None
    keep_cols = ["weibo_id"]
    if "topic" in df.columns:
        keep_cols.append("topic")
    if "probability" in df.columns:
        keep_cols.append("probability")
    if "aspect" in df.columns:
        keep_cols.append("aspect")
    df = df[keep_cols].copy()
    df["weibo_id"] = df["weibo_id"].astype(str)
    return df


def combine_comment_with_aspect(comment_score, aspect, aspect_prob, text):
    try:
        s = float(comment_score)
    except Exception:
        s = 0.5
    if s != s:
        s = 0.5
    s = max(0.0, min(1.0, s))

    asp = str(aspect).strip() if aspect is not None else ""
    base_aspect_score = ASPECT_PRIORS.get(asp, ASPECT_PRIORS["其他"])
    try:
        tp = float(aspect_prob)
    except Exception:
        tp = 0.0
    if tp != tp:
        tp = 0.0
    tp = max(0.0, min(1.0, tp))
    aspect_score = 0.5 + tp * (base_aspect_score - 0.5)

    conf = abs(s - 0.5) * 2.0
    comment_weight = 0.4 + 0.4 * max(0.0, min(1.0, conf))
    topic_weight = 1.0 - comment_weight

    t = "" if text is None else str(text)
    t = t.lower()
    strong_neg = ("很烂" in t) or ("太烂" in t) or ("垃圾" in t) or ("恶心" in t) or ("骗子" in t) or ("骗" in t)
    strong_pos = ("yyds" in t) or ("太棒" in t) or ("真棒" in t) or ("真香" in t) or ("好吃" in t)
    if strong_neg or strong_pos:
        topic_weight = min(topic_weight, 0.15)
    else:
        topic_weight = min(topic_weight, 0.35)
    comment_weight = 1.0 - topic_weight

    final_score = comment_weight * s + topic_weight * aspect_score
    final_score = max(0.0, min(1.0, final_score))
    label = "积极" if final_score >= 0.5 else "消极"
    return label, final_score


def get_preferred_device():
    override = os.environ.get("SENTIMENT_DEVICE")
    if override:
        v = override.strip().lower()
        if v in {"cpu", "-1"}:
            return torch.device("cpu")
        if v in {"mps"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if v in {"cuda", "gpu", "0"} and torch.cuda.is_available():
            return torch.device("cuda:0")
        try:
            idx = int(v)
            if torch.cuda.is_available():
                return torch.device(f"cuda:{idx}")
        except Exception:
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_best_fold(model_train_dir: str) -> Tuple[str, Optional[dict]]:
    cv_path = os.path.join(model_train_dir, "cv_results.json")
    if not os.path.exists(cv_path):
        return os.path.join(model_train_dir, "fold_1"), None
    try:
        with open(cv_path, "r", encoding="utf-8") as f:
            cv = json.load(f)
    except Exception:
        return os.path.join(model_train_dir, "fold_1"), None

    best_fold = None
    best_score = None
    for it in (cv.get("fold_metrics") or []):
        if not isinstance(it, dict):
            continue
        fold = it.get("fold")
        score = it.get("eval_macro_f1")
        if fold is None or score is None:
            continue
        try:
            fold_i = int(fold)
            score_f = float(score)
        except Exception:
            continue
        if best_score is None or score_f > best_score:
            best_score = score_f
            best_fold = fold_i

    if best_fold is None:
        return os.path.join(model_train_dir, "fold_1"), cv
    return os.path.join(model_train_dir, f"fold_{best_fold}"), cv


def _detect_pos_label_id(model) -> int:
    try:
        id2label = getattr(model.config, "id2label", None) or {}
        for k, v in id2label.items():
            s = str(v).lower()
            if "pos" in s or "positive" in s:
                return int(k)
    except Exception:
        pass
    return 1


def _predict_pos_probs(model, tokenizer, texts, device, max_length: int, batch_size: int, pos_label_id: int):
    model.eval()
    out = []
    for i in range(0, len(texts), int(batch_size)):
        batch = [str(x) for x in texts[i : i + int(batch_size)]]
        enc = tokenizer(batch, truncation=True, max_length=int(max_length), padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            p = probs[:, int(pos_label_id)].detach().cpu().numpy().astype(float).tolist()
            out.extend(p)
    return out


def main(input_file: str | None = None, output_file: str | None = None, mode: str | None = None):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    model_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))

    dataset_dir = os.path.join(project_dir, "dataset")
    estimate_dir = os.path.join(project_dir, "model_estimate", "weibo")
    prediction_dir = os.path.join(project_dir, "model_prediction", "weibo", "prediction")
    train_dir = os.path.join(project_dir, "model_train_outputs", "restaurant_sentiment_speed_sanity")

    input_full = os.path.join(dataset_dir, "weibo_comments_cleaned.csv")
    input_sample = os.path.join(estimate_dir, "sample_input.csv")
    output_full = os.path.join(prediction_dir, "roBERTa_fit_prediction.csv")
    output_sample = os.path.join(estimate_dir, "roBERTa_fit_sample_prediction.csv")

    if input_file or output_file:
        input_file = input_file or input_full
        output_file = output_file or output_full
    else:
        if mode is None:
            if sys.stdin.isatty():
                mode = choose_mode_interactively()
            else:
                mode = "sample"
        if mode is None:
            print("无效输入，程序结束。")
            return
        input_file = input_sample if mode == "sample" else input_full
        output_file = output_sample if mode == "sample" else output_full

    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return

    print("正在读取清洗后的数据...")
    df = load_csv(input_file)
    if df.empty:
        print("输入数据为空，结束。")
        return

    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
    if text_col not in df.columns:
        print("输入文件缺少可用的文本列")
        return

    if "weibo_id" in df.columns:
        df["weibo_id"] = df["weibo_id"].fillna("").astype(str).str.strip()

    doc_topics_df = load_latest_doc_topics(model_dir)
    if doc_topics_df is not None and "weibo_id" in df.columns:
        df = df.merge(doc_topics_df, on="weibo_id", how="left")

    device = get_preferred_device()
    print(f"Device: {device}")

    override_model = os.environ.get("ROBERTA_BASE_MODEL")
    if override_model:
        best_model_dir = str(override_model).strip()
        if not best_model_dir:
            best_model_dir = None
        elif not (best_model_dir.startswith("http") or os.path.exists(best_model_dir)):
            best_model_dir = str(override_model).strip()
    else:
        final_model_dir = os.path.join(train_dir, "final_model")
        best_fold_model_dir = os.path.join(train_dir, "best_fold_model")
        if os.path.exists(final_model_dir):
            best_model_dir = final_model_dir
            cv = None
        elif os.path.exists(best_fold_model_dir):
            best_model_dir = best_fold_model_dir
            cv = None
        else:
            best_model_dir, cv = _pick_best_fold(train_dir)
        if not os.path.exists(best_model_dir):
            raise RuntimeError(f"未找到训练输出目录: {best_model_dir}")

        if cv:
            try:
                best_fold_name = os.path.basename(best_model_dir)
                best_fold_idx = int(best_fold_name.replace("fold_", ""))
                best_fold_score = None
                for it in (cv.get("fold_metrics") or []):
                    if int(it.get("fold")) == best_fold_idx:
                        best_fold_score = it.get("eval_macro_f1")
                        break
                if best_fold_score is not None:
                    print(f"已选择最佳折模型: {best_fold_name} (eval_macro_f1={float(best_fold_score):.6f})")
                else:
                    print(f"已选择最佳折模型: {best_fold_name}")
            except Exception:
                print(f"已选择最佳折模型: {os.path.basename(best_model_dir)}")

    print(f"Model: {best_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)

    pos_label_id = _detect_pos_label_id(model)
    batch_size = 64 if device.type != "cpu" else 16
    max_length = 256
    try:
        positive_threshold = float(os.environ.get("ROBERTA_POS_THRESHOLD", "0.5"))
    except Exception:
        positive_threshold = 0.5
    positive_threshold = max(0.0, min(1.0, positive_threshold))

    texts = df[text_col].astype(str).tolist()
    print(f"待分析文本条数: {len(texts)}")
    probs_pos = _predict_pos_probs(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        pos_label_id=pos_label_id,
    )

    labels = []
    scores = []
    raw_probs = []
    raw_labels = []
    for i, text in enumerate(texts):
        p_pos = float(probs_pos[i]) if i < len(probs_pos) else 0.5
        if p_pos != p_pos:
            p_pos = 0.5
        p_pos = max(0.0, min(1.0, p_pos))
        comment_score = max(0.0, min(1.0, p_pos))
        raw_probs.append(p_pos)
        raw_labels.append("积极" if p_pos >= positive_threshold else "消极")
        aspect = df.at[i, "aspect"] if "aspect" in df.columns else None
        aspect_prob = df.at[i, "probability"] if "probability" in df.columns else None
        lbl, score = combine_comment_with_aspect(
            comment_score=comment_score,
            aspect=aspect,
            aspect_prob=aspect_prob,
            text=text,
        )
        labels.append(lbl)
        scores.append(score)

    out_df = df.copy()
    out_df["model_positive_prob"] = raw_probs
    out_df["model_sentiment_label"] = raw_labels
    out_df["sentiment_label"] = labels
    out_df["sentiment_score"] = scores

    for col in ["topic", "probability", "aspect", "topic_id", "topic_probability"]:
        if col in out_df.columns:
            out_df = out_df.drop(columns=[col])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已输出: {output_file}")


if __name__ == "__main__":
    main()
