import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


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
    os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"

    model_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))

    input_full = os.path.join(project_dir, "dataset", "weibo_comments_cleaned.csv")
    input_sample = os.path.join(project_dir, "model_estimate", "weibo", "sample_input.csv")
    output_full = os.path.join(project_dir, "model_prediction", "weibo", "prediction", "roBERTa_origin_prediction.csv")
    output_sample = os.path.join(project_dir, "model_estimate", "weibo", "roBERTa_origin_sample_prediction.csv")

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

    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return

    print("正在读取清洗后的数据...")
    try:
        df = load_csv(input_file)
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return
    if df.empty:
        print("输入数据为空，结束。")
        return

    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"
    if text_col not in df.columns:
        print("输入文件缺少可用的文本列")
        return

    model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = get_preferred_device()
        clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

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

    texts = df[text_col].astype(str).tolist()
    print(f"待分析文本条数: {len(texts)}")
    print("开始情感分析...")
    probs = [max(0.0, min(1.0, float(get_p_pos(t)))) for t in texts]
    labels = ["积极" if p >= 0.5 else "消极" for p in probs]

    out_df = df.copy()
    out_df["sentiment_score"] = probs
    out_df["sentiment_label"] = labels

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已输出: {output_file}")


if __name__ == "__main__":
    main()
