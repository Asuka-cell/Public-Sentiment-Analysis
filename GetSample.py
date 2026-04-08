import os

import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "weibo_comments_cleaned.csv")
    output_file = os.path.join(base_dir, "sample_target.csv")

    try:
        df = pd.read_csv(input_file, encoding="utf-8-sig")
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    if df.empty:
        print("输入数据为空，无法抽样")
        return

    sample_size = 100
    n = min(sample_size, len(df))
    sampled_df = df.sample(n=n, random_state=42).reset_index(drop=True)
    sampled_df["sentiment_label"] = "消极"

    try:
        sampled_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"写入CSV失败: {e}")
        return

    print(f"已抽取 {n} 条样本并保存到 {output_file}")


if __name__ == "__main__":
    main()
