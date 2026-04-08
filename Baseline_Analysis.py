import os

import pandas as pd
from snownlp import SnowNLP

def main():


    print("请先安装依赖: /usr/bin/python3 -m pip install --user snownlp")
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weibo_comments_cleaned.csv")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Baseline_prediction.csv")

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

    labels = []
    scores = []

    for text in texts:
        try:
            s = SnowNLP(text)
            score = float(s.sentiments)
            lbl = "积极" if score >= 0.5 else "消极"
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
