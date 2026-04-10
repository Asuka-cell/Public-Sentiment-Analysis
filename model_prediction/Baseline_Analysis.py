import os
import sys

import pandas as pd
from snownlp import SnowNLP

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


def main():
    print("请先安装依赖: /usr/bin/python3 -m pip install --user snownlp")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_full = os.path.join(project_dir, "dataset", "weibo_comments_cleaned.csv")
    input_sample = os.path.join(project_dir, "model_estimate", "sample_input.csv")
    output_full = os.path.join(project_dir, "model_prediction", "Baseline_prediction.csv")
    output_sample = os.path.join(project_dir, "model_estimate", "Baseline_sample_prediction.csv")

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

    def predict(target_df):
        target_df = target_df.copy().reset_index(drop=True)
        texts = target_df[text_col].astype(str).tolist()
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

        target_df["sentiment_label"] = labels
        target_df["sentiment_score"] = scores
        return target_df

    try:
        print("开始情感分析...")
        pred_df = predict(df)
        print(f"正在保存结果到 {output_file}...")
        pred_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    except Exception:
        print("保存失败")
        return
    print("处理完成！")


if __name__ == "__main__":
    main()
