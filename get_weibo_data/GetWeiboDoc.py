from __future__ import annotations

import io
import json
import os

import pandas as pd


def safe_str(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def load_csv(path: str) -> pd.DataFrame:
    try:
        raw = b""
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        raw = b""

    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = raw.decode(enc)
            return pd.read_csv(io.StringIO(text))
        except Exception:
            continue

    try:
        text = raw.decode("latin1", errors="ignore")
        return pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()


def main(posts_path: str | None = None, comments_path: str | None = None, output_path: str | None = None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    posts_path = posts_path or os.path.join(project_dir, "dataset/weibo_posts_cleaned.csv")
    comments_path = comments_path or os.path.join(project_dir, "dataset/weibo_comments_cleaned.csv")
    output_path = output_path or os.path.join(project_dir, "dataset/weibo_doc.jsonl")

    posts_df = load_csv(posts_path)
    comments_df = load_csv(comments_path)
    if posts_df.empty:
        print(f"读取CSV失败或为空: {posts_path}")
        return
    if comments_df.empty:
        print(f"读取CSV失败或为空: {comments_path}")
        return

    if "weibo_id" not in posts_df.columns:
        print("weibo_posts_cleaned.csv 缺少 weibo_id 列")
        return
    if "weibo_id" not in comments_df.columns:
        print("weibo_comments_cleaned.csv 缺少 weibo_id 列")
        return

    post_text_col = "cleaned_text" if "cleaned_text" in posts_df.columns else "text"
    comment_text_col = "cleaned_text" if "cleaned_text" in comments_df.columns else "text"
    if post_text_col not in posts_df.columns:
        print("weibo_posts_cleaned.csv 缺少 cleaned_text/text 列")
        return
    if comment_text_col not in comments_df.columns:
        print("weibo_comments_cleaned.csv 缺少 cleaned_text/text 列")
        return

    posts_df["weibo_id"] = posts_df["weibo_id"].astype(str)
    comments_df["weibo_id"] = comments_df["weibo_id"].astype(str)

    posts_df[post_text_col] = posts_df[post_text_col].astype(str)
    comments_df[comment_text_col] = comments_df[comment_text_col].astype(str)

    posts_map = dict(zip(posts_df["weibo_id"], posts_df[post_text_col]))
    grouped_comments = comments_df.groupby("weibo_id")[comment_text_col].apply(list).to_dict()

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for weibo_id, post_text in posts_map.items():
            comments_list = grouped_comments.get(weibo_id, [])
            doc_text = "\n".join(
                [safe_str(post_text)] + [safe_str(t) for t in comments_list if safe_str(t)]
            ).strip()
            if not doc_text:
                continue
            record = {
                "weibo_id": weibo_id,
                "text": doc_text,
                "post": safe_str(post_text),
                "comments": [safe_str(t) for t in comments_list if safe_str(t)],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"已生成 {written} 条 document，保存到 {output_path}")


if __name__ == "__main__":
    main()
