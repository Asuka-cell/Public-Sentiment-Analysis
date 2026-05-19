from __future__ import annotations

import argparse
import os
import re
from datetime import datetime

import pandas as pd


def _read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    except Exception:
        return pd.read_csv(path, dtype=str)


def _write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    if text.lower() in {"nan", "none"}:
        return ""
    text = text.replace("\u200b", " ")
    text = text.replace("[图片]", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_questions(df):
    df = df.copy()
    for c in [
        "question_id",
        "title",
        "excerpt",
        "publish_time",
        "answer_count",
        "comment_count",
        "follower_count",
    ]:
        if c not in df.columns:
            df[c] = ""

    df["question_id"] = df["question_id"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["excerpt"] = df["excerpt"].fillna("").astype(str).apply(clean_text)
    df["publish_time"] = df["publish_time"].fillna("").astype(str).str.strip()

    df = df[df["question_id"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["question_id"], keep="first")

    df.loc[df["title"].astype(str).str.strip().isin({"进入知乎"}), "title"] = ""
    df = df[df["title"].astype(str).str.len() > 0]

    dt = pd.to_datetime(df["publish_time"], errors="coerce")
    df["publish_time"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df[df["publish_time"].notna() & (df["publish_time"] != "NaT")]

    for c in ["answer_count", "comment_count", "follower_count"]:
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )

    df = df[
        [
            "question_id",
            "title",
            "excerpt",
            "publish_time",
            "answer_count",
            "comment_count",
            "follower_count",
        ]
    ].sort_values("publish_time")

    return df


def _clean_answers(df, max_content_chars):
    df = df.copy()
    for c in [
        "answer_id",
        "question_id",
        "author_id",
        "author_name",
        "created_time",
        "voteup_count",
        "comment_count",
        "content",
    ]:
        if c not in df.columns:
            df[c] = ""

    df["answer_id"] = df["answer_id"].fillna("").astype(str).str.strip()
    df["question_id"] = df["question_id"].fillna("").astype(str).str.strip()
    df = df[df["answer_id"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["answer_id"], keep="first")

    df["content"] = df["content"].fillna("").astype(str).apply(clean_text)
    if max_content_chars and int(max_content_chars) > 0:
        n = int(max_content_chars)
        df["content"] = df["content"].astype(str).str.slice(0, n)
    df = df[df["content"].astype(str).str.len() > 1]

    df["voteup_count"] = pd.to_numeric(df["voteup_count"], errors="coerce").fillna(0).astype(int)
    df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int)

    out = df[
        [
            "answer_id",
            "question_id",
            "author_id",
            "author_name",
            "created_time",
            "voteup_count",
            "comment_count",
            "content",
        ]
    ].copy()

    out["created_time"] = out["created_time"].fillna("").astype(str).str.strip()
    dt = pd.to_datetime(out["created_time"], errors="coerce")
    out["created_time"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    out.loc[out["created_time"].isna() | (out["created_time"] == "NaT"), "created_time"] = ""

    out = out.sort_values(["voteup_count", "answer_id"], ascending=[False, True])
    return out


def main(
    questions_in: str | None = None,
    answers_in: str | None = None,
    questions_out: str | None = None,
    answers_out: str | None = None,
    max_content_chars: int = 8000,
):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_dir, "dataset")

    if questions_in is None and answers_in is None and questions_out is None and answers_out is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--questions_in", default=os.path.join(dataset_dir, "zhihu_questions.csv"))
        parser.add_argument("--answers_in", default=os.path.join(dataset_dir, "zhihu_answers.csv"))
        parser.add_argument("--questions_out", default=os.path.join(dataset_dir, "zhihu_questions_cleaned.csv"))
        parser.add_argument("--answers_out", default=os.path.join(dataset_dir, "zhihu_answers_cleaned.csv"))
        parser.add_argument("--max_content_chars", type=int, default=8000)
        args = parser.parse_args()
        questions_in = args.questions_in
        answers_in = args.answers_in
        questions_out = args.questions_out
        answers_out = args.answers_out
        max_content_chars = int(args.max_content_chars)
    else:
        questions_in = questions_in or os.path.join(dataset_dir, "zhihu_questions.csv")
        answers_in = answers_in or os.path.join(dataset_dir, "zhihu_answers.csv")
        questions_out = questions_out or os.path.join(dataset_dir, "zhihu_questions_cleaned.csv")
        answers_out = answers_out or os.path.join(dataset_dir, "zhihu_answers_cleaned.csv")
        max_content_chars = int(max_content_chars)

    if not os.path.exists(questions_in):
        print(f"问题输入文件不存在: {questions_in}")
        return
    if not os.path.exists(answers_in):
        print(f"回答输入文件不存在: {answers_in}")
        return

    t0 = datetime.now()

    q_raw = _read_csv(questions_in)
    a_raw = _read_csv(answers_in)

    q_clean = _clean_questions(q_raw)
    a_clean = _clean_answers(a_raw, max_content_chars=max_content_chars)

    _write_csv(q_clean, questions_out)
    _write_csv(a_clean, answers_out)

    print(f"问题原始: {len(q_raw)} -> 清洗后: {len(q_clean)}")
    print(f"回答原始: {len(a_raw)} -> 清洗后: {len(a_clean)}")
    print(f"问题输出: {questions_out}")
    print(f"回答输出: {answers_out}")
    print(f"耗时: {(datetime.now() - t0).total_seconds():.1f}s")


if __name__ == "__main__":
    main()
