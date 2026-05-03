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
    model_dir = os.path.dirname(os.path.abspath(__file__))
    zhihu_dir = os.path.dirname(model_dir)
    project_dir = os.path.dirname(os.path.dirname(zhihu_dir))
    estimate_dir = os.path.join(project_dir, "model_estimate", "zhihu")
    input_full = os.path.join(project_dir, "dataset", "zhihu_answers_cleaned.csv")
    input_sample = os.path.join(estimate_dir, "sample_input.csv")
    output_full = os.path.join(zhihu_dir, "prediction", "Baseline_prediction.csv")
    output_sample = os.path.join(estimate_dir, "Baseline_sample_prediction.csv")

    if sys.stdin.isatty():
        mode = choose_mode_interactively()
    else:
        mode = "sample"
    if mode is None:
        print("无效输入，程序结束。")
        return

    try:
        if mode == "sample":
            if not os.path.exists(input_sample):
                print(f"输入文件不存在: {input_sample}")
                return
            df = load_csv(input_sample)
            output_file = output_sample
        else:
            if not os.path.exists(input_full):
                print(f"输入文件不存在: {input_full}")
                return
            df = load_csv(input_full)
            output_file = output_full
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    text_col = "content" if "content" in df.columns else ("cleaned_text" if "cleaned_text" in df.columns else "text")
    if text_col not in df.columns:
        print("输入文件缺少可用的文本列（content/cleaned_text/text）")
        return

    texts = df[text_col].fillna("").astype(str).tolist()
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

    out_df = df.copy()
    out_df["sentiment_label"] = labels
    out_df["sentiment_score"] = scores

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"输出路径: {output_file}")


if __name__ == "__main__":
    main()
