import os
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st


def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, path)


def _pick_key_cols(df: pd.DataFrame, dataset: str) -> List[str]:
    if str(dataset).strip().lower() == "zhihu":
        candidates = ["answer_id"]
    else:
        candidates = ["weibo_id", "user_name", "publish_time", "cleaned_text"]
    return [c for c in candidates if c in df.columns]


def _make_key(df: pd.DataFrame, key_cols: List[str]) -> pd.Series:
    if not key_cols:
        return pd.Series(range(len(df)), index=df.index, dtype="int64")
    parts = []
    for c in key_cols:
        s = df[c]
        if c == "weibo_id":
            s = s.fillna("").astype(str).str.strip()
        else:
            s = s.fillna("").astype(str)
        parts.append(s)
    out = parts[0]
    for p in parts[1:]:
        out = out + "||" + p
    return out


def _ensure_target_schema(input_df: pd.DataFrame, target_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if target_df is None or target_df.empty:
        out = input_df.copy()
        out["sentiment_label"] = ""
        return out

    out = target_df.copy()
    if "sentiment_label" not in out.columns:
        out["sentiment_label"] = ""

    for c in input_df.columns:
        if c not in out.columns:
            out[c] = pd.NA

    extra_cols = [c for c in out.columns if c not in list(input_df.columns) + ["sentiment_label"]]
    ordered = list(input_df.columns) + ["sentiment_label"] + extra_cols
    out = out[ordered]
    return out


def _get_label_for_row(target_df: pd.DataFrame, target_key: pd.Series, row_key: str) -> str:
    if "sentiment_label" not in target_df.columns:
        return ""
    mask = target_key == row_key
    if not bool(mask.any()):
        return ""
    v = target_df.loc[mask, "sentiment_label"].iloc[0]
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s in {"积极", "消极"}:
        return s
    return ""


def _set_label_for_row(
    input_df: pd.DataFrame,
    input_key: pd.Series,
    row_key: str,
    target_df: pd.DataFrame,
    target_key: pd.Series,
    label: str,
) -> pd.DataFrame:
    label = str(label).strip()
    if label not in {"积极", "消极"}:
        raise ValueError("label must be 积极/消极")

    out = target_df.copy()
    in_mask = input_key == row_key
    if not bool(in_mask.any()):
        return out

    row = input_df.loc[in_mask].iloc[0].to_dict()

    t_mask = target_key == row_key
    if bool(t_mask.any()):
        for c, v in row.items():
            if c not in out.columns:
                out[c] = pd.NA
            out.loc[t_mask, c] = v
        out.loc[t_mask, "sentiment_label"] = label
        return out

    new_row = {c: pd.NA for c in out.columns}
    for c, v in row.items():
        if c in out.columns:
            new_row[c] = v
    new_row["sentiment_label"] = label
    out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
    return out


def _render_dataset_labeling(base_dir: str, dataset: str) -> None:
    dataset = str(dataset).strip().lower()
    if dataset not in {"weibo", "zhihu"}:
        dataset = "weibo"
    prefix = f"{dataset}_"
    row_state_key = f"{prefix}row_key"
    question_state_key = f"{prefix}question"

    input_path = os.path.join(base_dir, "model_estimate", dataset, "sample_input.csv")
    target_path = os.path.join(base_dir, "model_estimate", dataset, "sample_target.csv")

    if not os.path.exists(input_path):
        st.error(f"未找到待标注文件：model_estimate/{dataset}/sample_input.csv")
        return

    input_df = _load_csv(input_path)
    if input_df.empty:
        st.info("sample_input.csv 为空")
        return

    target_df = _load_csv(target_path) if os.path.exists(target_path) else None
    target_df = _ensure_target_schema(input_df=input_df, target_df=target_df)

    key_cols = _pick_key_cols(input_df, dataset=dataset)
    input_key = _make_key(input_df, key_cols)
    target_key = _make_key(target_df, key_cols)

    idx_key = f"{prefix}label_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    idx = int(st.session_state[idx_key])
    idx = max(0, min(idx, len(input_df) - 1))
    st.session_state[idx_key] = idx

    current_row_key = str(input_key.iloc[idx])
    current_label = _get_label_for_row(target_df=target_df, target_key=target_key, row_key=current_row_key)
    default_choice = 0 if current_label == "积极" else 1
    default_label = "积极" if default_choice == 0 else "消极"

    prev_row_key = st.session_state.get(row_state_key)
    if prev_row_key != current_row_key:
        st.session_state[row_state_key] = current_row_key
        st.session_state[f"{prefix}goto"] = int(idx + 1)
        st.session_state[f"{prefix}label"] = default_label

    labeled_count = int(target_df["sentiment_label"].astype(str).str.strip().isin({"积极", "消极"}).sum())
    st.caption(f"进度: {idx + 1}/{len(input_df)}  |  已标注: {labeled_count}/{len(input_df)}")
    st.progress((idx + 1) / float(len(input_df)))

    if dataset == "zhihu":
        meta_cols = [c for c in ["answer_id", "question_id", "author_name", "created_time"] if c in input_df.columns]
    else:
        meta_cols = [c for c in ["weibo_id", "user_name", "publish_time"] if c in input_df.columns]
    if meta_cols:
        st.write(input_df.loc[idx, meta_cols].to_frame("值"))

    if dataset == "zhihu":
        questions_path = os.path.join(base_dir, "dataset", "zhihu_questions_cleaned.csv")
        answers_path = os.path.join(base_dir, "dataset", "zhihu_answers_cleaned.csv")

        q_lookup_key = "zhihu_question_lookup_v1"
        a2q_lookup_key = "zhihu_answer_to_question_v1"
        if q_lookup_key not in st.session_state:
            try:
                qdf = pd.read_csv(questions_path, encoding="utf-8-sig")
            except Exception:
                try:
                    qdf = pd.read_csv(questions_path)
                except Exception:
                    qdf = pd.DataFrame()
            if not qdf.empty and "question_id" in qdf.columns:
                qdf = qdf.copy()
                qdf["question_id"] = qdf["question_id"].fillna("").astype(str).str.strip()
                title_col = "title" if "title" in qdf.columns else None
                excerpt_col = "excerpt" if "excerpt" in qdf.columns else None
                payload = {}
                for _, r in qdf.iterrows():
                    qid = str(r.get("question_id", "")).strip()
                    if not qid:
                        continue
                    title = str(r.get(title_col, "")).strip() if title_col else ""
                    excerpt = str(r.get(excerpt_col, "")).strip() if excerpt_col else ""
                    if title and excerpt:
                        payload[qid] = f"{title}\n\n{excerpt}"
                    elif title:
                        payload[qid] = title
                    elif excerpt:
                        payload[qid] = excerpt
                st.session_state[q_lookup_key] = payload
            else:
                st.session_state[q_lookup_key] = {}

        if a2q_lookup_key not in st.session_state:
            try:
                adf = pd.read_csv(answers_path, encoding="utf-8-sig", usecols=["answer_id", "question_id"])
            except Exception:
                try:
                    adf = pd.read_csv(answers_path, usecols=["answer_id", "question_id"])
                except Exception:
                    adf = pd.DataFrame()
            if not adf.empty and "answer_id" in adf.columns and "question_id" in adf.columns:
                adf = adf.copy()
                adf["answer_id"] = adf["answer_id"].fillna("").astype(str).str.strip()
                adf["question_id"] = adf["question_id"].fillna("").astype(str).str.strip()
                st.session_state[a2q_lookup_key] = dict(zip(adf["answer_id"].tolist(), adf["question_id"].tolist()))
            else:
                st.session_state[a2q_lookup_key] = {}

        q_lookup = st.session_state.get(q_lookup_key) or {}
        a2q_lookup = st.session_state.get(a2q_lookup_key) or {}

        qid = ""
        if "question_id" in input_df.columns:
            qid = str(input_df.at[idx, "question_id"] if "question_id" in input_df.columns else "").strip()
        if not qid and "answer_id" in input_df.columns:
            aid = str(input_df.at[idx, "answer_id"]).strip()
            qid = str(a2q_lookup.get(aid, "")).strip()
        question_text = str(q_lookup.get(qid, "")).strip()
        st.session_state[question_state_key] = question_text

        content_col = "content" if "content" in input_df.columns else ("cleaned_text" if "cleaned_text" in input_df.columns else "text")
        content_val = str(input_df.at[idx, content_col]) if content_col in input_df.columns else ""
        st.session_state[f"{prefix}content"] = content_val

        left, right = st.columns(2)
        with left:
            st.subheader("问题")
            st.text_area(
                "question",
                key=question_state_key,
                height=360,
                disabled=True,
                label_visibility="collapsed",
            )
        with right:
            st.subheader("回答")
            st.text_area(
                "content",
                key=f"{prefix}content",
                height=360,
                disabled=True,
                label_visibility="collapsed",
            )
    else:
        post_col = "post_cleaned_text" if "post_cleaned_text" in input_df.columns else None
        comment_col = "cleaned_text" if "cleaned_text" in input_df.columns else ("text" if "text" in input_df.columns else None)
        post_val = str(input_df.at[idx, post_col]) if post_col else ""
        comment_val = str(input_df.at[idx, comment_col]) if comment_col else ""
        st.session_state[f"{prefix}post"] = post_val
        st.session_state[f"{prefix}comment"] = comment_val

        left, right = st.columns(2)
        with left:
            st.subheader("博文")
            st.text_area(
                "post",
                key=f"{prefix}post",
                height=280,
                disabled=True,
                label_visibility="collapsed",
            )
        with right:
            st.subheader("评论")
            st.text_area(
                "comment",
                key=f"{prefix}comment",
                height=280,
                disabled=True,
                label_visibility="collapsed",
            )

    choice = st.radio("标签", ["积极", "消极"], index=default_choice, horizontal=True, key=f"{prefix}label")

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])
    with nav1:
        if st.button("上一条", use_container_width=True, disabled=(idx <= 0), key=f"{prefix}prev"):
            st.session_state[idx_key] = max(0, idx - 1)
            st.rerun()
    with nav2:
        if st.button("下一条", use_container_width=True, disabled=(idx >= len(input_df) - 1), key=f"{prefix}next"):
            st.session_state[idx_key] = min(len(input_df) - 1, idx + 1)
            st.rerun()
    with nav3:
        if st.button("保存标签", use_container_width=True, key=f"{prefix}save"):
            updated = _set_label_for_row(
                input_df=input_df,
                input_key=input_key,
                row_key=current_row_key,
                target_df=target_df,
                target_key=target_key,
                label=choice,
            )
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            _atomic_write_csv(updated, target_path)
            st.success("已写入 sample_target.csv")
            st.session_state[idx_key] = min(len(input_df) - 1, idx + 1)
            st.rerun()
    with nav4:
        goto = st.number_input(
            "跳转",
            min_value=1,
            max_value=int(len(input_df)),
            value=int(idx + 1),
            step=1,
            key=f"{prefix}goto",
        )
        if st.button("跳转到该条", use_container_width=True, key=f"{prefix}go"):
            st.session_state[idx_key] = int(goto) - 1
            st.rerun()


def render_labeling(base_dir: str) -> None:
    st.title("🏷️ 样本标注")
    tab_weibo, tab_zhihu = st.tabs(["微博", "知乎"])
    with tab_weibo:
        _render_dataset_labeling(base_dir=base_dir, dataset="weibo")
    with tab_zhihu:
        _render_dataset_labeling(base_dir=base_dir, dataset="zhihu")
