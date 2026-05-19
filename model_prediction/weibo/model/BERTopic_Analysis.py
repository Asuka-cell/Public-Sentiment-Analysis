from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime

_STOP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords")
DEFAULT_COMMON_STOPWORDS_PATH = os.path.join(_STOP_DIR, "common_stopwords.txt")
DEFAULT_EVENT_STOPWORDS_PATH = os.path.join(_STOP_DIR, "event_stopwords.txt")
STOP_WORDS = set()

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
    t = "" if text is None else str(text)
    t = re.sub(r"\d+\s*天后", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_output_dir(output_dir):
    keep_names = {
        "bertopic_model",
        "topic_info.csv",
        "doc_topics.csv",
        "topics.json",
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


def main(input_path: str | None = None, output_dir: str | None = None):
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )

    if input_path is None and output_dir is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input",
            default=os.path.join(project_root, "dataset", "weibo_doc_cleaned.jsonl"),
        )
        parser.add_argument(
            "--output_dir",
            default=os.path.join(project_root, "model_prediction", "weibo", "prediction", "bertopic_output"),
        )
        parser.add_argument("--embedding_model", default="shibing624/text2vec-base-chinese")
        parser.add_argument("--min_topic_size", type=int, default=15)
        parser.add_argument("--nr_topics", default=None)
        parser.add_argument("--stopwords_common", default=DEFAULT_COMMON_STOPWORDS_PATH)
        parser.add_argument("--stopwords_event", default=DEFAULT_EVENT_STOPWORDS_PATH)
        parser.add_argument("--hf_endpoint", default="https://hf-mirror.com")
        parser.add_argument("--cache_dir", default=None)
        parser.add_argument("--local_files_only", action="store_true")
        parser.add_argument("--device", default=None)
        parser.add_argument("--keep_history", action="store_true")
        args = parser.parse_args()
        input_path = args.input
        output_dir = args.output_dir
    else:
        class _Args:
            pass

        args = _Args()
        input_path = input_path or os.path.join(project_root, "dataset", "weibo_doc_cleaned.jsonl")
        output_dir = output_dir or os.path.join(project_root, "model_prediction", "weibo", "prediction", "bertopic_output")
        args.embedding_model = "shibing624/text2vec-base-chinese"
        args.min_topic_size = 15
        args.nr_topics = None
        args.stopwords_common = DEFAULT_COMMON_STOPWORDS_PATH
        args.stopwords_event = DEFAULT_EVENT_STOPWORDS_PATH
        args.hf_endpoint = "https://hf-mirror.com"
        args.cache_dir = None
        args.local_files_only = False
        args.device = None
        args.keep_history = False

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

    ensure_dir(output_dir)
    if not args.keep_history:
        clear_output_dir(output_dir)

    if args.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        return

    records = read_jsonl(input_path)
    if not records:
        print(f"输入文件为空或无法解析: {input_path}")
        return

    weibo_ids = []
    docs = []
    for r in records:
        wid = str(r.get("weibo_id", "")).strip()
        text = r.get("text", "")
        if not wid:
            continue
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = text.strip()
        text = normalize_doc_text(text)
        if not text:
            continue
        weibo_ids.append(wid)
        docs.append(text)

    if not docs:
        print("没有可用的文本用于建模")
        return

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
                if not kw:
                    continue
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

    aspect_map_path = os.path.join(output_dir, f"aspect_map_{ts}.json") if args.keep_history else os.path.join(output_dir, "aspect_map.json")

    doc_topics_path = os.path.join(output_dir, f"doc_topics_{ts}.csv") if args.keep_history else os.path.join(output_dir, "doc_topics.csv")
    rows = []
    for wid, topic, prob in zip(weibo_ids, topics, probs if probs is not None else [None] * len(topics)):
        p = None
        if prob is not None:
            try:
                p = float(max(prob))
            except Exception:
                p = None
        rows.append({"weibo_id": wid, "topic": int(topic) if topic is not None else None, "probability": p})

    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None:
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
            .agg(doc_count=("weibo_id", "count"), mean_topic_prob=("probability", "mean"))
            .reset_index()
            .sort_values(["doc_count", "aspect"], ascending=[False, True])
        )
        stats.to_csv(aspect_stats_path, index=False, encoding="utf-8-sig")
    else:
        with open(doc_topics_path, "w", encoding="utf-8") as f:
            f.write("weibo_id,topic,probability,aspect\n")
            for r in rows:
                aspect = topic_aspect_map.get(str(r.get("topic")), "其他")
                f.write(f"{r['weibo_id']},{r['topic']},{'' if r['probability'] is None else r['probability']},{aspect}\n")

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
