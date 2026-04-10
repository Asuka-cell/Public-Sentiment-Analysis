import argparse
import json
import os
import re


def safe_str(value):
    if value is None:
        return ""
    return str(value)


def normalize_space(text):
    text = text.replace("\u200b", " ")
    text = text.replace("&quot;", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"@[^\s]+", " ", text)
    text = re.sub(r"#([^#\s]+)#", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_low_value_line(line):
    if not line:
        return True
    s = line.strip()
    if not s:
        return True
    low_value_phrases = {
        "转发微博",
        "网页链接",
        "全文",
        "展开",
        "视频",
        "图片",
        "评论",
        "点赞",
        "回复",
        "分享",
        "收藏",
        "查看",
    }
    if s in low_value_phrases:
        return True
    if len(s) <= 1:
        return True
    if s.isdigit():
        return True
    return False


def count_cjk(text):
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def count_alpha(text):
    return len(re.findall(r"[A-Za-z]", text))


def clean_lines(text):
    text = normalize_space(text)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[\r\n]+", text) if p is not None]
    out = []
    for p in parts:
        p = p.strip()
        if is_low_value_line(p):
            continue
        out.append(p)
    return out


def build_document(post, comments, max_comments, min_comment_len, max_doc_chars, post_repeat):
    post_lines = clean_lines(post)
    post_text = " ".join(post_lines).strip()
    if not post_text:
        post_text = normalize_space(safe_str(post)).strip()

    cleaned_comments = []
    seen = set()
    for c in comments or []:
        lines = clean_lines(safe_str(c))
        t = " ".join(lines).strip()
        if not t:
            t = normalize_space(safe_str(c)).strip()
        if not t:
            continue
        if len(t) < min_comment_len:
            continue
        if t in seen:
            continue
        seen.add(t)
        cleaned_comments.append(t)

    cleaned_comments.sort(key=len, reverse=True)
    if max_comments is not None and max_comments > 0:
        cleaned_comments = cleaned_comments[:max_comments]

    doc_parts = []
    if post_text:
        for _ in range(max(1, int(post_repeat))):
            doc_parts.append(post_text)
    doc_parts.extend(cleaned_comments)
    doc_text = "\n".join([p for p in doc_parts if p]).strip()

    if max_doc_chars is not None and max_doc_chars > 0 and len(doc_text) > max_doc_chars:
        doc_text = doc_text[:max_doc_chars]

    return post_text, cleaned_comments, doc_text


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_dir, "dataset")
    parser.add_argument("--input", default=os.path.join(dataset_dir, "weibo_doc.jsonl"))
    parser.add_argument("--output", default=os.path.join(dataset_dir, "weibo_doc_cleaned.jsonl"))
    parser.add_argument("--max_comments", type=int, default=50)
    parser.add_argument("--min_comment_len", type=int, default=4)
    parser.add_argument("--max_doc_chars", type=int, default=4000)
    parser.add_argument("--post_repeat", type=int, default=1)
    parser.add_argument("--min_cjk_or_alpha", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"输入文件不存在: {args.input}")
        return

    records = read_jsonl(args.input)
    if not records:
        print(f"输入为空或无法解析: {args.input}")
        return

    cleaned_records = []
    dropped = 0
    for r in records:
        weibo_id = safe_str(r.get("weibo_id", "")).strip()
        post = r.get("post", "")
        comments = r.get("comments", [])
        if not weibo_id:
            dropped += 1
            continue

        post_text, comment_texts, doc_text = build_document(
            post=post,
            comments=comments,
            max_comments=args.max_comments,
            min_comment_len=args.min_comment_len,
            max_doc_chars=args.max_doc_chars,
            post_repeat=args.post_repeat,
        )

        if not doc_text:
            dropped += 1
            continue

        if (count_cjk(doc_text) + count_alpha(doc_text)) < int(args.min_cjk_or_alpha):
            dropped += 1
            continue

        cleaned_records.append(
            {
                "weibo_id": weibo_id,
                "text": doc_text,
                "post": post_text,
                "comments": comment_texts,
                "meta": {
                    "n_comments": len(comments) if isinstance(comments, list) else 0,
                    "kept_comments": len(comment_texts),
                },
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_jsonl(args.output, cleaned_records)
    print(f"输入条数: {len(records)}")
    print(f"输出条数: {len(cleaned_records)}")
    print(f"丢弃条数: {dropped}")
    print(f"输出路径: {args.output}")


if __name__ == "__main__":
    main()
