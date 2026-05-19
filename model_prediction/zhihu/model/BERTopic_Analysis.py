from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_ASPECT_KEYWORDS = {
    "食品安全与卫生": {
        "食品安全",
        "卫生",
        "不卫生",
        "干净",
        "脏",
        "变质",
        "过期",
        "拉肚子",
        "食物中毒",
        "添加剂",
        "安全",
        "健康",
    },
    "预制菜与食材": {
        "预制",
        "预制菜",
        "食材",
        "原料",
        "冷冻",
        "解冻",
        "料理包",
        "中央厨房",
        "现炒",
        "锅气",
        "新鲜",
        "科技",
    },
    "价格与性价比": {
        "价格",
        "涨价",
        "降价",
        "贵",
        "太贵",
        "便宜",
        "性价比",
        "不值",
        "划算",
        "优惠",
        "折扣",
        "人均",
        "收费",
    },
    "服务与体验": {
        "服务",
        "态度",
        "服务态度",
        "上菜",
        "排队",
        "等位",
        "体验",
        "投诉",
        "差评",
        "管理",
        "店员",
    },
    "口味与品质": {
        "口味",
        "难吃",
        "好吃",
        "味道",
        "咸",
        "淡",
        "品质",
        "分量",
        "份量",
    },
    "营销与公关": {
        "营销",
        "公关",
        "洗地",
        "热搜",
        "道歉",
        "回应",
        "声明",
        "舆论",
        "炒作",
        "危机",
    },
    "企业责任与诚信": {
        "责任",
        "担当",
        "诚信",
        "真诚",
        "敷衍",
        "认错",
        "整改",
        "改进",
        "承诺",
        "透明",
        "知情权",
    },
    "用工与合规": {
        "员工",
        "工资",
        "社保",
        "劳动",
        "合规",
        "监管",
        "处罚",
        "执法",
    },
    "其他": set(),
}


def load_stop_words(path):
    words = set()
    if not path:
        return words
    if not os.path.exists(path):
        return words
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            words.add(w)
    return words


STOP_WORDS = set()


def jieba_tokenizer(text):
    import jieba

    tokens = []
    for w in jieba.lcut(str(text), cut_all=False):
        w = str(w).strip().lower()
        if not w:
            continue
        if any(ch.isdigit() for ch in w):
            continue
        if re.fullmatch(r"[_\W]+", w, flags=re.UNICODE):
            continue
        if w in STOP_WORDS:
            continue
        if len(w) <= 1:
            continue
        tokens.append(w)
    return tokens


def normalize_doc_text(text):
    s = "" if text is None else str(text)
    s = s.replace("\u200b", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("[图片]", " ")
    s = re.sub(r"http[s]?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_output_dir(output_dir):
    keep_names = {
        "bertopic_model",
        "topic_info.csv",
        "doc_topics.csv",
        "topics.json",
        "aspect_map.json",
        "aspect_stats.csv",
    }
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if name not in keep_names:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except Exception:
                pass


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


def main(input_path: str | None = None, output_dir: str | None = None, mode: str | None = None):
    model_dir = os.path.dirname(os.path.abspath(__file__))
    zhihu_dir = os.path.dirname(model_dir)
    project_dir = os.path.dirname(os.path.dirname(zhihu_dir))
    estimate_dir = os.path.join(project_dir, "model_estimate", "zhihu")
    stop_dir = os.path.join(os.path.dirname(zhihu_dir), "stopwords")
    default_common = os.path.join(stop_dir, "common_stopwords.txt")
    default_event = os.path.join(stop_dir, "event_stopwords.txt")

    if input_path is None and output_dir is None and mode is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", default=os.path.join(project_dir, "dataset", "zhihu_answers_cleaned.csv"))
        parser.add_argument("--output_dir", default=os.path.join(zhihu_dir, "prediction", "bertopic_output"))
        parser.add_argument("--embedding_model", default="shibing624/text2vec-base-chinese")
        parser.add_argument("--min_topic_size", type=int, default=15)
        parser.add_argument("--nr_topics", default=None)
        parser.add_argument("--stopwords_common", default=default_common)
        parser.add_argument("--stopwords_event", default=default_event)
        parser.add_argument("--hf_endpoint", default="https://hf-mirror.com")
        parser.add_argument("--cache_dir", default=None)
        parser.add_argument("--local_files_only", action="store_true")
        parser.add_argument("--device", default=None)
        parser.add_argument("--keep_history", action="store_true")
        args = parser.parse_args()

        if sys.stdin.isatty():
            mode = choose_mode_interactively()
        else:
            mode = "sample"
        if mode is None:
            print("无效输入，程序结束。")
            return

        input_path = args.input
        output_dir = args.output_dir
    else:
        class _Args:
            pass

        args = _Args()
        args.embedding_model = "shibing624/text2vec-base-chinese"
        args.min_topic_size = 15
        args.nr_topics = None
        args.stopwords_common = default_common
        args.stopwords_event = default_event
        args.hf_endpoint = "https://hf-mirror.com"
        args.cache_dir = None
        args.local_files_only = False
        args.device = None
        args.keep_history = False
        mode = mode or "full"
        input_path = input_path or os.path.join(project_dir, "dataset", "zhihu_answers_cleaned.csv")
        output_dir = output_dir or os.path.join(zhihu_dir, "prediction", "bertopic_output")

    try:
        from bertopic import BERTopic
    except Exception as e:
        print(f"导入 bertopic 失败: {e}")
        print("请安装: /usr/bin/python3 -m pip install --user bertopic sentence-transformers")
        return

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"导入 sentence-transformers 失败: {e}")
        print("请安装: /usr/bin/python3 -m pip install --user sentence-transformers")
        return

    try:
        import jieba
    except Exception as e:
        print(f"导入 jieba 失败: {e}")
        print("请安装: /usr/bin/python3 -m pip install --user jieba")
        return

    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception as e:
        print(f"导入 scikit-learn 失败: {e}")
        print("请安装: /usr/bin/python3 -m pip install --user scikit-learn")
        return

    if args.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    input_path = os.path.join(estimate_dir, "sample_input.csv") if mode == "sample" else input_path
    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        return

    output_dir = output_dir if mode == "full" else f"{output_dir}_sample"
    ensure_dir(output_dir)
    if not args.keep_history:
        clear_output_dir(output_dir)

    try:
        import pandas as pd
    except Exception as e:
        print(f"导入 pandas 失败: {e}")
        return

    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(input_path)

    if "answer_id" not in df.columns:
        print("输入文件缺少 answer_id 列")
        return
    text_col = "content" if "content" in df.columns else ("cleaned_text" if "cleaned_text" in df.columns else "text")
    if text_col not in df.columns:
        print("输入文件缺少可用的文本列（content/cleaned_text/text）")
        return

    answer_ids = df["answer_id"].fillna("").astype(str).str.strip().tolist()
    question_ids = df["question_id"].fillna("").astype(str).str.strip().tolist() if "question_id" in df.columns else [""] * len(df)
    docs = [normalize_doc_text(x) for x in df[text_col].fillna("").astype(str).tolist()]
    keep = [(aid, qid, d) for aid, qid, d in zip(answer_ids, question_ids, docs) if aid and d]
    if not keep:
        print("没有可用的文本用于建模")
        return

    answer_ids = [x[0] for x in keep]
    question_ids = [x[1] for x in keep]
    docs = [x[2] for x in keep]

    nr_topics = args.nr_topics
    if isinstance(nr_topics, str) and nr_topics.lower() == "none":
        nr_topics = None

    device = args.device
    if not device:
        device = os.environ.get("EMBEDDING_DEVICE")
    if not device:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    embedding_model = SentenceTransformer(
        args.embedding_model,
        cache_folder=args.cache_dir,
        local_files_only=bool(args.local_files_only),
        device=device,
    )

    global STOP_WORDS
    STOP_WORDS = set()
    STOP_WORDS |= load_stop_words(args.stopwords_common)
    STOP_WORDS |= load_stop_words(args.stopwords_event)

    vectorizer_model = CountVectorizer(
        tokenizer=jieba_tokenizer,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=1.0,
        max_features=8000,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="chinese",
        vectorizer_model=vectorizer_model,
        min_topic_size=int(args.min_topic_size),
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    topic_info_path = os.path.join(output_dir, f"topic_info_{ts}.csv") if args.keep_history else os.path.join(output_dir, "topic_info.csv")
    topic_model.get_topic_info().to_csv(topic_info_path, index=False, encoding="utf-8-sig")

    topics_path = os.path.join(output_dir, f"topics_{ts}.json") if args.keep_history else os.path.join(output_dir, "topics.json")
    topic_words = {}
    for topic_id in topic_model.get_topic_info()["Topic"].tolist():
        try:
            tid = int(topic_id)
        except Exception:
            continue
        if tid == -1:
            continue
        words = topic_model.get_topic(tid) or []
        topic_words[str(tid)] = [{"word": w, "weight": float(s)} for w, s in words]

    def score_aspect_for_word(word):
        s = str(word).strip().lower()
        if not s:
            return {}
        scores = {}
        for aspect, kws in DEFAULT_ASPECT_KEYWORDS.items():
            if not kws or aspect == "其他":
                continue
            hit = 0
            for kw in kws:
                kw2 = str(kw).strip().lower()
                if not kw2:
                    continue
                if kw2 == s or kw2 in s:
                    hit = 1
                    break
            if hit:
                scores[aspect] = 1
        return scores

    def map_topic_to_aspect(words):
        acc = {}
        for it in words[:30]:
            if not isinstance(it, dict):
                continue
            w = it.get("word", "")
            try:
                weight = float(it.get("weight", 0.0))
            except Exception:
                weight = 0.0
            for aspect, v in score_aspect_for_word(w).items():
                acc[aspect] = acc.get(aspect, 0.0) + abs(weight) * float(v)
        if not acc:
            return "其他"
        return max(acc.items(), key=lambda kv: kv[1])[0]

    topic_aspect_map = {}
    for tid_str, words in topic_words.items():
        try:
            tid = int(tid_str)
        except Exception:
            continue
        if tid < 0:
            continue
        if not isinstance(words, list):
            continue
        topic_aspect_map[str(tid)] = map_topic_to_aspect(words)

    doc_topics_path = os.path.join(output_dir, f"doc_topics_{ts}.csv") if args.keep_history else os.path.join(output_dir, "doc_topics.csv")
    rows = []
    for aid, qid, topic, prob in zip(answer_ids, question_ids, topics, probs if probs is not None else [None] * len(topics)):
        p = None
        if prob is not None:
            try:
                p = float(max(prob))
            except Exception:
                p = None
        rows.append(
            {
                "answer_id": str(aid),
                "question_id": str(qid),
                "topic": int(topic) if topic is not None else None,
                "probability": p,
            }
        )

    doc_df = pd.DataFrame(rows)
    doc_df["topic"] = doc_df["topic"].astype("Int64")
    doc_df["aspect"] = doc_df["topic"].astype(str).map(topic_aspect_map).fillna("其他")
    doc_df.to_csv(doc_topics_path, index=False, encoding="utf-8-sig")

    aspect_stats_path = (
        os.path.join(output_dir, f"aspect_stats_{ts}.csv")
        if args.keep_history
        else os.path.join(output_dir, "aspect_stats.csv")
    )
    stats = (
        doc_df.groupby("aspect", dropna=False)
        .agg(doc_count=("answer_id", "count"), mean_topic_prob=("probability", "mean"))
        .reset_index()
        .sort_values(["doc_count", "aspect"], ascending=[False, True])
    )
    stats.to_csv(aspect_stats_path, index=False, encoding="utf-8-sig")

    aspect_map_path = os.path.join(output_dir, f"aspect_map_{ts}.json") if args.keep_history else os.path.join(output_dir, "aspect_map.json")
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topic_words, f, ensure_ascii=False, indent=2)
    with open(aspect_map_path, "w", encoding="utf-8") as f:
        json.dump(topic_aspect_map, f, ensure_ascii=False, indent=2)

    print(f"主题概览: {topic_info_path}")
    print(f"文档主题: {doc_topics_path}")
    print(f"主题词表: {topics_path}")
    print(f"主题维度映射: {aspect_map_path}")


if __name__ == "__main__":
    main()
