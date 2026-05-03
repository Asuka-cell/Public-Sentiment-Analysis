import os

import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    input_file = os.path.join(project_dir, "dataset", "weibo_comments_cleaned.csv")
    posts_file = os.path.join(project_dir, "dataset", "weibo_posts_cleaned.csv")
    output_file = os.path.join(base_dir, "sample_target.csv")
    output_file_no_label = os.path.join(base_dir, "sample_input.csv")
    zhihu_input_file = os.path.join(project_dir, "dataset", "zhihu_answers_cleaned.csv")
    zhihu_dir = os.path.join(base_dir, "zhihu")
    zhihu_output_file = os.path.join(zhihu_dir, "sample_target.csv")
    zhihu_output_file_no_label = os.path.join(zhihu_dir, "sample_input.csv")

    try:
        sampled_df = None
        if os.path.exists(output_file):
            sampled_df = pd.read_csv(output_file, encoding="utf-8-sig")
            ok_cols = {"weibo_id", "user_name", "publish_time", "cleaned_text", "post_cleaned_text", "sentiment_label"}
            if not ok_cols.issubset(set(sampled_df.columns)) or len(sampled_df) < 300:
                sampled_df = None

        if sampled_df is None:
            comments_df = pd.read_csv(input_file, encoding="utf-8-sig")
            posts_df = pd.read_csv(posts_file, encoding="utf-8-sig")
            if comments_df.empty or posts_df.empty:
                print("输入数据为空，无法抽样")
                return

            if "weibo_id" not in comments_df.columns or "weibo_id" not in posts_df.columns:
                print("输入文件缺少 weibo_id 列，无法对齐博文与评论")
                return
            if "cleaned_text" not in comments_df.columns:
                print("weibo_comments_cleaned.csv 缺少 cleaned_text 列")
                return
            if "cleaned_text" not in posts_df.columns:
                print("weibo_posts_cleaned.csv 缺少 cleaned_text 列")
                return

            posts_map = posts_df[["weibo_id", "cleaned_text"]].dropna(subset=["weibo_id"]).copy()
            posts_map["weibo_id"] = posts_map["weibo_id"].astype(str).str.strip()
            posts_map["post_cleaned_text"] = posts_map["cleaned_text"].astype(str)
            posts_map = posts_map.drop(columns=["cleaned_text"]).drop_duplicates(subset=["weibo_id"], keep="first")

            comments_df = comments_df.copy()
            comments_df["weibo_id"] = comments_df["weibo_id"].astype(str).str.strip()
            merged = comments_df.merge(posts_map, on="weibo_id", how="inner")
            merged = merged[merged["cleaned_text"].astype(str).str.strip() != ""]
            merged = merged[merged["post_cleaned_text"].astype(str).str.strip() != ""]

            sample_size = 300
            n = min(sample_size, len(merged))
            sampled_df = merged.sample(n=n, random_state=42).reset_index(drop=True)
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

    try:
        zhihu_sampled_df = None
        if os.path.exists(zhihu_output_file):
            zhihu_sampled_df = pd.read_csv(zhihu_output_file, encoding="utf-8-sig")
            ok_cols = {"answer_id", "question_id", "author_name", "created_time", "content", "sentiment_label"}
            if not ok_cols.issubset(set(zhihu_sampled_df.columns)) or len(zhihu_sampled_df) < 100:
                zhihu_sampled_df = None

        if zhihu_sampled_df is None:
            if not os.path.exists(zhihu_input_file):
                print("知乎输入文件不存在，跳过知乎抽样")
                return

            try:
                zhihu_df = pd.read_csv(zhihu_input_file, encoding="utf-8-sig")
            except Exception:
                zhihu_df = pd.read_csv(zhihu_input_file)

            if zhihu_df.empty:
                print("知乎输入数据为空，无法抽样")
                return

            if "answer_id" not in zhihu_df.columns:
                print("知乎输入文件缺少 answer_id 列")
                return

            text_col = "content" if "content" in zhihu_df.columns else ("cleaned_text" if "cleaned_text" in zhihu_df.columns else "text")
            if text_col not in zhihu_df.columns:
                print("知乎输入文件缺少可用的文本列（content/cleaned_text/text）")
                return

            zhihu_df = zhihu_df.copy()
            zhihu_df["answer_id"] = zhihu_df["answer_id"].fillna("").astype(str).str.strip()
            zhihu_df[text_col] = zhihu_df[text_col].fillna("").astype(str)
            zhihu_df = zhihu_df[(zhihu_df["answer_id"] != "") & (zhihu_df[text_col].str.strip() != "")]

            sample_size = 100
            n = min(sample_size, len(zhihu_df))
            zhihu_sampled_df = zhihu_df.sample(n=n, random_state=42).reset_index(drop=True)
            if text_col != "content":
                zhihu_sampled_df = zhihu_sampled_df.rename(columns={text_col: "content"})
            zhihu_sampled_df["sentiment_label"] = "消极"
            os.makedirs(zhihu_dir, exist_ok=True)
            zhihu_sampled_df.to_csv(zhihu_output_file, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"知乎抽样失败: {e}")
        return

    try:
        zhihu_df_no_label = zhihu_sampled_df.copy()
        if "sentiment_label" in zhihu_df_no_label.columns:
            zhihu_df_no_label = zhihu_df_no_label.drop(columns=["sentiment_label"])
        os.makedirs(zhihu_dir, exist_ok=True)
        zhihu_df_no_label.to_csv(zhihu_output_file_no_label, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"写入知乎CSV失败: {e}")
        return

    print(f"知乎样本文件: {zhihu_output_file}")
    print(f"知乎无标签样本文件: {zhihu_output_file_no_label}")


if __name__ == "__main__":
    main()
