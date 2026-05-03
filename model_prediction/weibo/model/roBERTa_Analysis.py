import glob
import json
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

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


def map_label(label):
    raw = str(label).strip()
    upper = raw.upper()
    lower = raw.lower()

    if "negative" in lower or upper in {"NEGATIVE", "LABEL_0"}:
        return "消极"
    if "positive" in lower or upper in {"POSITIVE", "LABEL_1"}:
        return "积极"

    return "未知"


def load_latest_doc_topics(model_dir):
    output_dir = os.path.join(model_dir, "bertopic_output")
    candidates = glob.glob(os.path.join(output_dir, "doc_topics.csv"))
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


def get_preferred_device():
    override = os.environ.get("SENTIMENT_DEVICE")
    if override:
        v = override.strip().lower()
        if v in {"cpu", "-1"}:
            return -1
        if v in {"cuda", "gpu", "0"}:
            return 0
        if v in {"mps"}:
            return "mps"
        try:
            return int(v)
        except Exception:
            return -1

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return -1


def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    model_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))

    input_full = os.path.join(project_dir, "dataset", "weibo_comments_cleaned.csv")
    input_sample = os.path.join(project_dir, "model_estimate", "weibo", "sample_input.csv")
    output_full = os.path.join(project_dir, "model_prediction", "roBERTa_prediction.csv")
    output_sample = os.path.join(project_dir, "model_estimate", "weibo", "roBERTa_sample_prediction.csv")

    if sys.stdin.isatty():
        mode = choose_mode_interactively()
    else:
        mode = "sample"
    if mode is None:
        print("无效输入，程序结束。")
        return

    if mode == "sample":
        input_file = input_sample
        output_file = output_sample
    else:
        input_file = input_full
        output_file = output_full

    print("正在读取清洗后的数据...")
    try:
        df = load_csv(input_file)
    except Exception as e:
        print(f"读取CSV失败: {e}")
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

    model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = get_preferred_device()
        clf = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    try:
        def predict(target_df):
            target_df = target_df.copy().reset_index(drop=True)
            texts = target_df[text_col].astype(str).tolist()
            print(f"待分析文本条数: {len(texts)}")
            print("开始情感分析...")

            pos_label_id = None
            try:
                id2label = getattr(model.config, "id2label", None) or {}
                for k, v in id2label.items():
                    if "positive" in str(v).lower():
                        pos_label_id = int(k)
                        break
            except Exception:
                pos_label_id = None
            if pos_label_id is None:
                pos_label_id = 1

            def label_polarity(label):
                s = str(label).strip().lower()
                if not s:
                    return None

                pos_keys = {
                    "positive",
                    "pos",
                    "积极",
                    "喜悦",
                    "快乐",
                    "开心",
                    "高兴",
                    "喜欢",
                    "赞",
                    "love",
                    "like",
                    "happy",
                    "joy",
                }
                neg_keys = {
                    "negative",
                    "neg",
                    "消极",
                    "愤怒",
                    "生气",
                    "厌恶",
                    "悲伤",
                    "低落",
                    "恐惧",
                    "怒",
                    "fear",
                    "sad",
                    "angry",
                    "disgust",
                }
                neu_keys = {
                    "neutral",
                    "neu",
                    "中性",
                    "其他",
                    "surprise",
                    "惊讶",
                }

                if s in pos_keys:
                    return "pos"
                if s in neg_keys:
                    return "neg"
                if s in neu_keys:
                    return "neu"
                for k in pos_keys:
                    if k and k in s:
                        return "pos"
                for k in neg_keys:
                    if k and k in s:
                        return "neg"
                for k in neu_keys:
                    if k and k in s:
                        return "neu"
                return None

            def get_p_pos(text, _pos_label_id=pos_label_id):
                txt = str(text)[:510]

                res = None
                try:
                    res = clf(txt, truncation=True, max_length=512, top_k=None)
                except TypeError:
                    try:
                        res = clf(txt, truncation=True, max_length=512, return_all_scores=True)
                    except TypeError:
                        res = clf(txt, truncation=True, max_length=512)

                score_items = None
                if isinstance(res, list) and res:
                    if isinstance(res[0], list):
                        score_items = res[0]
                    elif isinstance(res[0], dict) and len(res) > 1:
                        score_items = res

                if score_items:
                    pos_acc = 0.0
                    neu_acc = 0.0
                    any_known = False
                    for it in score_items:
                        pol = label_polarity(it.get("label"))
                        if pol is None:
                            continue
                        any_known = True
                        try:
                            p = float(it.get("score"))
                        except Exception:
                            p = 0.0
                        if pol == "pos":
                            pos_acc += p
                        elif pol == "neu":
                            neu_acc += p
                    if any_known:
                        return max(0.0, min(1.0, pos_acc + 0.5 * neu_acc))

                try:
                    enc = tokenizer(txt, truncation=True, max_length=512, return_tensors="pt")
                    enc = {k: v.to(model.device) for k, v in enc.items()}
                    with torch.no_grad():
                        logits = model(**enc).logits[0]
                        probs = torch.softmax(logits, dim=-1)
                    return float(probs[int(_pos_label_id)].item())
                except Exception:
                    if isinstance(res, list) and res and isinstance(res[0], dict):
                        try:
                            lbl = map_label(res[0].get("label"))
                            sc = float(res[0].get("score"))
                        except Exception:
                            return 0.5
                        if lbl == "积极":
                            return sc
                        if lbl == "消极":
                            return 1.0 - sc
                    return 0.5

            labels = []
            scores = []
            for text in texts:
                try:
                    p_pos = max(0.0, min(1.0, float(get_p_pos(text))))
                    comment_score = p_pos

                    aspect = None
                    aspect_prob = None
                    if "aspect" in target_df.columns:
                        aspect = target_df.at[len(labels), "aspect"]
                    if "probability" in target_df.columns:
                        aspect_prob = target_df.at[len(labels), "probability"]

                    lbl, score = combine_comment_with_aspect(
                        comment_score=comment_score,
                        aspect=aspect,
                        aspect_prob=aspect_prob,
                        text=text,
                    )
                except Exception:
                    lbl = "消极"
                    score = 0.0
                labels.append(lbl)
                scores.append(score)

            target_df["sentiment_label"] = labels
            target_df["sentiment_score"] = scores
            for col in ["topic", "probability", "aspect", "topic_id", "topic_probability"]:
                if col in target_df.columns:
                    target_df = target_df.drop(columns=[col])
            return target_df

        pred_df = predict(df)
        print(f"正在保存结果到 {output_file}...")
        pred_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    except Exception:
        print("保存失败")
        return
    print("处理完成！")


if __name__ == "__main__":
    main()
