from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def choose_mode_interactively() -> Optional[str]:
    prompt = "\n请选择运行模式:\n  1) 样本\n  2) 全量\n输入选项(1/2): "
    s = input(prompt).strip()
    if s == "1":
        return "sample"
    if s == "2":
        return "full"
    return None


def get_preferred_device():
    override = os.environ.get("SENTIMENT_DEVICE")
    if override:
        v = override.strip().lower()
        if v in {"cpu", "-1"}:
            return torch.device("cpu")
        if v in {"mps"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if v in {"cuda", "gpu", "0"} and torch.cuda.is_available():
            return torch.device("cuda:0")
        try:
            idx = int(v)
            if torch.cuda.is_available():
                return torch.device(f"cuda:{idx}")
        except Exception:
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_best_fold(model_train_dir: str) -> Tuple[str, Optional[dict]]:
    cv_path = os.path.join(model_train_dir, "cv_results.json")
    if not os.path.exists(cv_path):
        return os.path.join(model_train_dir, "fold_1"), None
    try:
        with open(cv_path, "r", encoding="utf-8") as f:
            cv = json.load(f)
    except Exception:
        return os.path.join(model_train_dir, "fold_1"), None

    best_fold = None
    best_score = None
    for it in (cv.get("fold_metrics") or []):
        if not isinstance(it, dict):
            continue
        fold = it.get("fold")
        score = it.get("eval_macro_f1")
        if fold is None or score is None:
            continue
        try:
            fold_i = int(fold)
            score_f = float(score)
        except Exception:
            continue
        if best_score is None or score_f > best_score:
            best_score = score_f
            best_fold = fold_i

    if best_fold is None:
        return os.path.join(model_train_dir, "fold_1"), cv
    return os.path.join(model_train_dir, f"fold_{best_fold}"), cv


def _detect_pos_label_id(model) -> int:
    try:
        id2label = getattr(model.config, "id2label", None) or {}
        for k, v in id2label.items():
            s = str(v).lower()
            if "pos" in s or "positive" in s:
                return int(k)
    except Exception:
        pass
    return 1


def _predict_pos_probs(model, tokenizer, texts, device, max_length: int, batch_size: int, pos_label_id: int):
    model.eval()
    out = []
    for i in range(0, len(texts), int(batch_size)):
        batch = [str(x) for x in texts[i : i + int(batch_size)]]
        enc = tokenizer(batch, truncation=True, max_length=int(max_length), padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            p = probs[:, int(pos_label_id)].detach().cpu().numpy().astype(float).tolist()
            out.extend(p)
    return out


def _split_text_to_chunks(tokenizer, text: str, max_length: int, overlap: int):
    txt = "" if text is None else str(text)
    txt = txt.strip()
    if not txt:
        return []
    special = 0
    try:
        special = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        special = 2
    window = int(max_length) - special
    if window <= 0:
        return [txt]
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    if len(token_ids) <= window:
        return [txt]
    ov = int(overlap)
    if ov < 0:
        ov = 0
    if ov >= window:
        ov = max(0, window // 4)
    step = max(1, window - ov)
    chunks = []
    for start in range(0, len(token_ids), step):
        ids_slice = token_ids[start : start + window]
        if not ids_slice:
            continue
        chunk = tokenizer.decode(ids_slice, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        chunk = str(chunk).strip()
        if chunk:
            chunks.append(chunk)
        if start + window >= len(token_ids):
            break
    return chunks


def _pool_probs(probs, mode: str):
    if not probs:
        return 0.5
    mode = (mode or "").strip().lower()
    ps = [float(p) for p in probs if p is not None]
    if not ps:
        return 0.5
    if mode == "max":
        return float(max(ps))
    if mode == "mean":
        return float(sum(ps) / float(len(ps)))
    weights = [max(0.05, abs(float(p) - 0.5) * 2.0) for p in ps]
    s = sum(weights)
    if s <= 0:
        return float(sum(ps) / float(len(ps)))
    return float(sum(w * p for w, p in zip(weights, ps)) / s)


def _split_to_segments(text: str):
    t = "" if text is None else str(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if not t:
        return []

    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    segs = []
    for p in paras:
        if len(p) <= 220:
            segs.append(p)
            continue
        parts = re.split(r"(?<=[。！？!?；;])\s*", p)
        parts = [x.strip() for x in parts if x and x.strip()]
        if not parts:
            segs.append(p)
            continue
        cur = ""
        for s in parts:
            if not cur:
                cur = s
                continue
            if len(cur) + len(s) <= 260:
                cur = cur + s
            else:
                segs.append(cur)
                cur = s
        if cur:
            segs.append(cur)

    uniq = []
    seen = set()
    for s in segs:
        s2 = s.strip()
        if not s2:
            continue
        key = s2[:200]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s2)
    return uniq


def _load_question_lookup(dataset_dir: str):
    q_path = os.path.join(dataset_dir, "zhihu_questions_cleaned.csv")
    if not os.path.exists(q_path):
        return {}
    try:
        qdf = pd.read_csv(q_path, encoding="utf-8-sig")
    except Exception:
        try:
            qdf = pd.read_csv(q_path)
        except Exception:
            return {}
    if qdf.empty or "question_id" not in qdf.columns:
        return {}
    qdf = qdf.copy()
    qdf["question_id"] = qdf["question_id"].fillna("").astype(str).str.strip()
    title_col = "title" if "title" in qdf.columns else None
    excerpt_col = "excerpt" if "excerpt" in qdf.columns else None
    out = {}
    for _, r in qdf.iterrows():
        qid = str(r.get("question_id", "")).strip()
        if not qid:
            continue
        title = str(r.get(title_col, "")).strip() if title_col else ""
        excerpt = str(r.get(excerpt_col, "")).strip() if excerpt_col else ""
        if title and excerpt:
            out[qid] = f"{title}\n\n{excerpt}"
        elif title:
            out[qid] = title
        elif excerpt:
            out[qid] = excerpt
    return out


def _build_context_texts(df: pd.DataFrame, text_col: str, dataset_dir: str):
    q_lookup = _load_question_lookup(dataset_dir)
    texts = df[text_col].fillna("").astype(str).tolist()
    if not q_lookup:
        return texts
    if "question_id" not in df.columns:
        return texts
    qids = df["question_id"].fillna("").astype(str).str.strip().tolist()
    out = []
    for qid, ans in zip(qids, texts):
        qtxt = q_lookup.get(str(qid).strip(), "")
        if qtxt:
            out.append(f"问题：{qtxt}\n\n回答：{ans}")
        else:
            out.append(ans)
    return out


def _predict_pos_probs_longtexts(model, tokenizer, texts, device, max_length: int, batch_size: int, pos_label_id: int):
    try:
        overlap = int(os.environ.get("LONGTEXT_OVERLAP", "64"))
    except Exception:
        overlap = 64
    try:
        top_k = int(os.environ.get("LONGTEXT_TOPK", "3"))
    except Exception:
        top_k = 3
    top_k = max(1, min(10, int(top_k)))

    seg_pool_mode = os.environ.get("LONGTEXT_SEG_POOL", "max")
    doc_rank_mode = os.environ.get("LONGTEXT_DOC_RANK", "positive")
    doc_pool_mode = os.environ.get("LONGTEXT_DOC_POOL", "mean")

    all_chunks = []
    chunk_owner_doc = []
    chunk_owner_seg = []
    seg_counts = []
    for di, t in enumerate(texts):
        segs = _split_to_segments(t)
        if not segs:
            segs = [""]
        seg_counts.append(len(segs))
        for si, seg in enumerate(segs):
            chunks = _split_text_to_chunks(tokenizer, seg, max_length=max_length, overlap=overlap)
            if not chunks:
                chunks = [""]
            for c in chunks:
                all_chunks.append(c)
                chunk_owner_doc.append(di)
                chunk_owner_seg.append(si)

    probs = _predict_pos_probs(
        model=model,
        tokenizer=tokenizer,
        texts=all_chunks,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        pos_label_id=pos_label_id,
    )

    doc_seg_probs = []
    for _ in range(len(texts)):
        doc_seg_probs.append([])
    seg_bucket = {}
    for p, di, si in zip(probs, chunk_owner_doc, chunk_owner_seg):
        key = (int(di), int(si))
        seg_bucket.setdefault(key, []).append(float(p))

    for di, nseg in enumerate(seg_counts):
        seg_probs = []
        for si in range(int(nseg)):
            ps = seg_bucket.get((int(di), int(si)), [])
            seg_probs.append(_pool_probs(ps, seg_pool_mode))
        doc_seg_probs[int(di)] = seg_probs

    out = []
    for seg_probs in doc_seg_probs:
        if not seg_probs:
            out.append(0.5)
            continue
        rm = (doc_rank_mode or "").strip().lower()
        if rm == "confidence":
            ranked = sorted(seg_probs, key=lambda x: abs(float(x) - 0.5), reverse=True)
        elif rm == "negative":
            ranked = sorted(seg_probs, key=lambda x: float(x))
        else:
            ranked = sorted(seg_probs, key=lambda x: float(x), reverse=True)
        chosen = ranked[:top_k] if len(ranked) > top_k else ranked
        pm = (doc_pool_mode or "").strip().lower()
        if not chosen:
            out.append(0.5)
        elif pm == "max":
            out.append(float(max(float(x) for x in chosen)))
        else:
            out.append(float(sum(float(x) for x in chosen) / float(len(chosen))))
    return out


def _normalize_binary_label(v: str):
    s = "" if v is None else str(v).strip()
    if s in {"积极", "正面", "正向"}:
        return 1
    if s in {"消极", "负面", "负向"}:
        return 0
    if "positive" in s.lower():
        return 1
    if "negative" in s.lower():
        return 0
    return None


def _find_best_threshold(y_true, scores):
    pairs = []
    for yt, sc in zip(y_true, scores):
        try:
            yt_i = int(yt)
        except Exception:
            continue
        if yt_i not in (0, 1):
            continue
        try:
            sc_f = float(sc)
        except Exception:
            continue
        if sc_f != sc_f:
            continue
        pairs.append((yt_i, max(0.0, min(1.0, sc_f))))
    if not pairs:
        return None

    ys = [p[0] for p in pairs]
    ss = [p[1] for p in pairs]
    candidates = sorted(set(ss + [0.5]))
    if len(candidates) > 400:
        step = max(1, len(candidates) // 400)
        candidates = candidates[::step]

    best = None
    for th in candidates:
        tp = tn = fp = fn = 0
        for y, s in zip(ys, ss):
            pred = 1 if s >= th else 0
            if y == 1 and pred == 1:
                tp += 1
            elif y == 0 and pred == 0:
                tn += 1
            elif y == 0 and pred == 1:
                fp += 1
            elif y == 1 and pred == 0:
                fn += 1
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        p_pos = tp / (tp + fp) if (tp + fp) else 0.0
        r_pos = tp / (tp + fn) if (tp + fn) else 0.0
        f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) else 0.0
        p_neg = tn / (tn + fn) if (tn + fn) else 0.0
        r_neg = tn / (tn + fp) if (tn + fp) else 0.0
        f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) else 0.0
        macro_f1 = (f1_pos + f1_neg) / 2.0

        cand = (macro_f1, acc, th)
        if best is None:
            best = cand
            continue
        if macro_f1 > best[0] + 1e-12:
            best = cand
            continue
        if abs(macro_f1 - best[0]) <= 1e-12 and acc > best[1] + 1e-12:
            best = cand
            continue
        if abs(macro_f1 - best[0]) <= 1e-12 and abs(acc - best[1]) <= 1e-12 and th > best[2]:
            best = cand

    if best is None:
        return None
    return float(best[2]), float(best[1]), float(best[0])


def main(input_file: str | None = None, output_file: str | None = None, mode: str | None = None):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    model_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))

    dataset_dir = os.path.join(project_dir, "dataset")
    prediction_dir = os.path.join(project_dir, "model_prediction", "zhihu", "prediction")
    estimate_dir = os.path.join(project_dir, "model_estimate", "zhihu")
    train_dir = os.path.join(project_dir, "model_train_outputs", "restaurant_sentiment_speed_sanity")

    input_full = os.path.join(dataset_dir, "zhihu_answers_cleaned.csv")
    input_sample = os.path.join(estimate_dir, "sample_input.csv")
    output_full = os.path.join(prediction_dir, "roBERTa_fit_prediction.csv")
    output_sample = os.path.join(estimate_dir, "roBERTa_fit_sample_prediction.csv")

    print("正在读取清洗后的数据...")
    if input_file or output_file:
        input_file = input_file or input_full
        output_file = output_file or output_full
        if not os.path.exists(input_file):
            print(f"输入文件不存在: {input_file}")
            return
        df = load_csv(input_file)
    else:
        if mode is None:
            if sys.stdin.isatty():
                mode = choose_mode_interactively()
            else:
                mode = "sample"
        if mode is None:
            print("无效输入，程序结束。")
            return
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
    if df.empty:
        print("输入数据为空，结束。")
        return

    text_col = "content" if "content" in df.columns else ("cleaned_text" if "cleaned_text" in df.columns else "text")
    if text_col not in df.columns:
        print("输入文件缺少可用的文本列（content/cleaned_text/text）")
        return

    if "answer_id" in df.columns:
        df["answer_id"] = df["answer_id"].fillna("").astype(str).str.strip()

    device = get_preferred_device()
    print(f"Device: {device}")

    override_model = os.environ.get("ROBERTA_BASE_MODEL")
    if override_model:
        best_model_dir = str(override_model).strip()
        if not best_model_dir:
            best_model_dir = None
        elif not (best_model_dir.startswith("http") or os.path.exists(best_model_dir)):
            best_model_dir = str(override_model).strip()
    else:
        final_model_dir = os.path.join(train_dir, "final_model")
        best_fold_model_dir = os.path.join(train_dir, "best_fold_model")
        if os.path.exists(final_model_dir):
            best_model_dir = final_model_dir
            cv = None
        elif os.path.exists(best_fold_model_dir):
            best_model_dir = best_fold_model_dir
            cv = None
        else:
            best_model_dir, cv = _pick_best_fold(train_dir)
        if not os.path.exists(best_model_dir):
            raise RuntimeError(f"未找到训练输出目录: {best_model_dir}")

        if cv:
            try:
                best_fold_name = os.path.basename(best_model_dir)
                best_fold_idx = int(best_fold_name.replace("fold_", ""))
                best_fold_score = None
                for it in (cv.get("fold_metrics") or []):
                    if int(it.get("fold")) == best_fold_idx:
                        best_fold_score = it.get("eval_macro_f1")
                        break
                if best_fold_score is not None:
                    print(f"已选择最佳折模型: {best_fold_name} (eval_macro_f1={float(best_fold_score):.6f})")
                else:
                    print(f"已选择最佳折模型: {best_fold_name}")
            except Exception:
                print(f"已选择最佳折模型: {os.path.basename(best_model_dir)}")

    print(f"Model: {best_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)

    pos_label_id = _detect_pos_label_id(model)
    batch_size = 64 if device.type != "cpu" else 16
    max_length = 512
    positive_threshold = None
    override_th = os.environ.get("ROBERTA_POS_THRESHOLD")
    if override_th is not None and str(override_th).strip() != "":
        try:
            positive_threshold = float(override_th)
        except Exception:
            positive_threshold = None
    if positive_threshold is not None:
        positive_threshold = max(0.0, min(1.0, float(positive_threshold)))

    texts = _build_context_texts(df=df, text_col=text_col, dataset_dir=dataset_dir)
    print(f"待分析文本条数: {len(texts)}")
    probs_pos = _predict_pos_probs_longtexts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        pos_label_id=pos_label_id,
    )

    raw_probs = []
    raw_labels = []
    for i, _text in enumerate(texts):
        p_pos = float(probs_pos[i]) if i < len(probs_pos) else 0.5
        if p_pos != p_pos:
            p_pos = 0.5
        p_pos = max(0.0, min(1.0, p_pos))
        raw_probs.append(p_pos)

    if positive_threshold is None:
        positive_threshold = 0.5
    positive_threshold = max(0.0, min(1.0, float(positive_threshold)))
    for p_pos in raw_probs:
        raw_labels.append("积极" if p_pos >= positive_threshold else "消极")

    out_df = df.copy()
    out_df["model_positive_prob"] = raw_probs
    out_df["model_sentiment_label"] = raw_labels
    out_df["sentiment_label"] = raw_labels
    out_df["sentiment_score"] = raw_probs

    for col in ["topic", "probability", "aspect", "topic_id", "topic_probability"]:
        if col in out_df.columns:
            out_df = out_df.drop(columns=[col])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已输出: {output_file}")


if __name__ == "__main__":
    main()
