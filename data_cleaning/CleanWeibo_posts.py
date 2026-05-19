from __future__ import annotations

import os
import re
import pandas as pd


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(r"@[^\s]+", "", text)
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main(input_file: str | None = None, output_file: str | None = None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = input_file or os.path.join(base_dir, "weibo_posts.csv")
    output_file = output_file or os.path.join(base_dir, "weibo_posts_cleaned.csv")

    print("正在读取数据...")
    try:
        df = _read_csv_any(input_file)
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    if "text" not in df.columns:
        print("输入文件缺少 text 列")
        return

    expected_cols = ["weibo_id", "user_name", "publish_time", "text"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 输入文件缺少以下列: {missing_cols}，可能会影响后续处理")
    else:
        print("输入文件列名检查通过")

    print(f"原始数据条数: {len(df)}")

    print("正在进行数据清洗...")
    df.dropna(subset=["text"], inplace=True)
    # 以 weibo_id 优先去重，其次按文本去重
    if "weibo_id" in df.columns:
        df.drop_duplicates(subset=["weibo_id"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)

    df["cleaned_text"] = df["text"].apply(clean_text).astype(str)
    df = df[df["cleaned_text"].str.len() > 1]

    for col in ["sentiment_label", "sentiment_score"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if "text" in df.columns:
        df.drop(columns=["text"], inplace=True)

    print(f"清洗后数据条数: {len(df)}")

    print(f"正在保存结果到 {output_file}...")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("处理完成！")


if __name__ == "__main__":
    main()
