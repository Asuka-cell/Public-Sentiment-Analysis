from __future__ import annotations

import os

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def map_label(label):
    raw = str(label).strip()
    upper = raw.upper()
    lower = raw.lower()

    if "negative" in lower or upper in {"NEGATIVE", "LABEL_0"}:
        return "消极"
    if "positive" in lower or upper in {"POSITIVE", "LABEL_1"}:
        return "积极"

    return "未知"


def main(input_file: str | None = None, output_file: str | None = None):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_file = input_file or os.path.join(base_dir, "weibo_comments_cleaned.csv")
    output_file = output_file or os.path.join(base_dir, "BERT_prediction.csv")

    print("正在读取清洗后的数据...")
    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig")
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    text_col = "cleaned_text" if "cleaned_text" in df.columns else "text"

    if text_col not in df.columns:
        print("输入文件缺少可用的文本列")
        return

    texts = df[text_col].astype(str).tolist()

    print(f"待分析文本条数: {len(texts)}")

    model_name = "google-bert/bert-base-chinese"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        clf = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0,
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print("开始情感分析...")
    labels = []
    scores = []

    for text in texts:
        try:
            result = clf(text[:510], truncation=True, max_length=512)
            lbl = map_label(result[0]["label"])
            conf = float(result[0]["score"])
            if lbl == "积极":
                score = 0.5 + 0.5 * conf
            elif lbl == "消极":
                score = 0.5 - 0.5 * conf
            else:
                score = 0.5
        except Exception:
            lbl = "消极"
            score = 0.0
        labels.append(lbl)
        scores.append(score)

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    print(f"正在保存结果到 {output_file}...")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("处理完成！")


if __name__ == "__main__":
    main()
