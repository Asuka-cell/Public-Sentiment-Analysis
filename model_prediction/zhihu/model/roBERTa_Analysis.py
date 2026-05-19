from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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


def load_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def choose_mode_interactively():
    prompt = (
        "\n请选择运行模式:\n"
        "  1) 样本\n"
        "  2) 全量\n"
        "输入选项(1/2): "
    )
    s = input(prompt).strip()
    if s == "1":
        return "sample"
    if s == "2":
        return "full"
    return None



def get_preferred_device():
    override = os.environ.get("SENTIMENT_DEVICE")
    if override:
        v = override.strip().lower()
        if v in {"cpu", "-1"}:
            return "cpu"
        if v in {"cuda", "gpu", "0"}:
            return "cuda"
        if v in {"mps"}:
            return "mps"
        return v

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_latest_doc_topics(zhihu_dir: str):
    candidates = []
    output_dir_a = os.path.join(zhihu_dir, "prediction", "bertopic_output")
    output_dir_b = os.path.join(zhihu_dir, "prediction", "bertopic_output_sample")
    for output_dir in [output_dir_a, output_dir_b]:
        candidates.extend(glob.glob(os.path.join(output_dir, "doc_topics.csv")))
    if not candidates:
        return None
    latest_path = max(candidates, key=lambda p: os.path.getmtime(p))
    df = load_csv(latest_path)
    if "answer_id" not in df.columns:
        return None
    keep_cols = ["answer_id"]
    for c in ["topic", "probability", "aspect"]:
        if c in df.columns:
            keep_cols.append(c)
    df = df[keep_cols].copy()
    df["answer_id"] = df["answer_id"].fillna("").astype(str).str.strip()
    return df


def combine_comment_with_aspect(comment_score, aspect, aspect_prob, text):
    try:
        s = float(comment_score)
    except Exception:
        s = 0.5
    s = max(0.0, min(1.0, s))

    asp = str(aspect).strip() if aspect is not None else ""
    base_aspect_score = ASPECT_PRIORS.get(asp, ASPECT_PRIORS["其他"])
    try:
        tp = float(aspect_prob)
    except Exception:
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


def infer_pos_label_id(model):
    try:
        id2label = getattr(model.config, "id2label", None) or {}
        for k, v in id2label.items():
            if "positive" in str(v).lower() or "pos" == str(v).lower():
                return int(k)
    except Exception:
        pass
    return 1


def _split_into_token_windows(tokenizer, text: str, max_length: int, overlap: int):
    txt = "" if text is None else str(text)
    txt = txt.strip()
    if not txt:
        return []
    special = 0
    try:
        special = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        special = 2
    window = int(max_length) - special
    if window <= 0:
        return [txt]
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    if len(token_ids) <= window:
        return [txt]
    ov = int(overlap)
    if ov < 0:
        ov = 0
    if ov >= window:
        ov = max(0, window // 4)
    step = max(1, window - ov)
    chunks = []
    for start in range(0, len(token_ids), step):
        ids_slice = token_ids[start : start + window]
        if not ids_slice:
            continue
        chunk = tokenizer.decode(ids_slice, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        chunk = str(chunk).strip()
        if chunk:
            chunks.append(chunk)
        if start + window >= len(token_ids):
            break
    return chunks


def _pool_probs(probs, mode: str):
    if not probs:
        return 0.5
    mode = (mode or "").strip().lower()
    ps = [float(p) for p in probs if p is not None]
    if not ps:
        return 0.5
    if mode == "max":
        return float(max(ps))
    if mode == "mean":
        return float(sum(ps) / float(len(ps)))
    weights = [max(0.05, abs(float(p) - 0.5) * 2.0) for p in ps]
    s = sum(weights)
    if s <= 0:
        return float(sum(ps) / float(len(ps)))
    return float(sum(w * p for w, p in zip(weights, ps)) / s)


def predict_p_pos(tokenizer, model, device, text, pos_label_id):
    pool_mode = os.environ.get("LONGTEXT_POOLING", "weighted")
    try:
        max_length = int(os.environ.get("LONGTEXT_MAX_LENGTH", "512"))
    except Exception:
        max_length = 512
    try:
        overlap = int(os.environ.get("LONGTEXT_OVERLAP", "128"))
    except Exception:
        overlap = 128
    chunks = _split_into_token_windows(tokenizer, text, max_length=max_length, overlap=overlap)
    if not chunks:
        return 0.5
    probs = []
    bs = 16
    for i in range(0, len(chunks), bs):
        batch = chunks[i : i + bs]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, int(pos_label_id)].detach().cpu().numpy().astype(float).tolist()
            probs.extend(p)
    return _pool_probs(probs, pool_mode)


def main(input_file: str | None = None, output_path: str | None = None, mode: str | None = None):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    model_dir = os.path.dirname(os.path.abspath(__file__))
    zhihu_dir = os.path.dirname(model_dir)
    project_dir = os.path.dirname(os.path.dirname(zhihu_dir))
    estimate_dir = os.path.join(project_dir, "model_estimate", "zhihu")
    input_full = os.path.join(project_dir, "dataset", "zhihu_answers_cleaned.csv")
    input_sample = os.path.join(estimate_dir, "sample_input.csv")
    output_full = os.path.join(zhihu_dir, "prediction", "roBERTa_prediction.csv")
    output_sample = os.path.join(estimate_dir, "roBERTa_sample_prediction.csv")

    if input_file or output_path:
        input_file = input_file or input_full
        output_path = output_path or output_full
        if not os.path.exists(input_file):
            print(f"输入文件不存在: {input_file}")
            return
        df = load_csv(input_file)
    else:
        if mode is None:
            if sys.stdin.isatty():
                mode = choose_mode_interactively()
            else:
                mode = "sample"
        if mode is None:
            print("无效输入，程序结束。")
            return
        if mode == "sample":
            if not os.path.exists(input_sample):
                print(f"输入文件不存在: {input_sample}")
                return
            df = load_csv(input_sample)
            output_path = output_sample
        else:
            if not os.path.exists(input_full):
                print(f"输入文件不存在: {input_full}")
                return
            df = load_csv(input_full)
            output_path = output_full

    text_col = "content" if "content" in df.columns else ("cleaned_text" if "cleaned_text" in df.columns else "text")
    if text_col not in df.columns:
        print("输入文件缺少可用的文本列（content/cleaned_text/text）")
        return

    if "answer_id" in df.columns:
        df["answer_id"] = df["answer_id"].fillna("").astype(str).str.strip()

    doc_topics_df = load_latest_doc_topics(zhihu_dir)
    if doc_topics_df is not None and "answer_id" in df.columns:
        df = df.merge(doc_topics_df, on="answer_id", how="left")

    model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    device_name = get_preferred_device()
    device = torch.device(device_name)
    model.to(device)
    model.eval()

    pos_label_id = infer_pos_label_id(model)
    scores = []
    labels = []
    final_scores = []
    final_labels = []

    texts = df[text_col].fillna("").astype(str).tolist()
    for txt in texts:
        p_pos = predict_p_pos(tokenizer, model, device, txt, pos_label_id=pos_label_id)
        scores.append(p_pos)
        labels.append("积极" if p_pos >= 0.5 else "消极")

    df["sentiment_score"] = scores
    df["sentiment_label"] = labels

    if "aspect" in df.columns and "probability" in df.columns:
        for s, asp, prob, txt in zip(df["sentiment_score"], df["aspect"], df["probability"], texts):
            lbl, sc = combine_comment_with_aspect(s, asp, prob, txt)
            final_labels.append(lbl)
            final_scores.append(sc)
        df["sentiment_label"] = final_labels
        df["sentiment_score"] = final_scores

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"输出路径: {output_path}")


if __name__ == "__main__":
    main()
