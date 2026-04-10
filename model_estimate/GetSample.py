import os

import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    input_file = os.path.join(project_dir, "dataset", "weibo_comments_cleaned.csv")
    output_file = os.path.join(base_dir, "sample_target.csv")
    output_file_no_label = os.path.join(base_dir, "sample_input.csv")

    try:
        if os.path.exists(output_file):
            sampled_df = pd.read_csv(output_file, encoding="utf-8-sig")
        else:
            df = pd.read_csv(input_file, encoding="utf-8-sig")
            if df.empty:
                print("输入数据为空，无法抽样")
                return

            sample_size = 100
            n = min(sample_size, len(df))
            sampled_df = df.sample(n=n, random_state=42).reset_index(drop=True)
            sampled_df["sentiment_label"] = "消极"
            sampled_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    try:
        df_no_label = sampled_df.copy()
        if "sentiment_label" in df_no_label.columns:
            df_no_label = df_no_label.drop(columns=["sentiment_label"])
        df_no_label.to_csv(output_file_no_label, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"写入CSV失败: {e}")
        return

    print(f"样本文件: {output_file}")
    print(f"无标签样本文件: {output_file_no_label}")


if __name__ == "__main__":
    main()
