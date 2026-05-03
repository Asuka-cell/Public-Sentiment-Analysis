import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import pyplot as plt


def normalize_eval_label(value):
    if pd.isna(value):
        return "未知"
    s = str(value).strip()
    if s in {"积极", "正面", "正向"}:
        return "积极"
    if s in {"消极", "负面", "负向"}:
        return "消极"
    if "positive" in s.lower():
        return "积极"
    if "negative" in s.lower():
        return "消极"
    return "未知"


def _label_to_int(value):
    s = normalize_eval_label(value)
    if s == "积极":
        return 1
    if s == "消极":
        return 0
    return None


def _to_float_or_none(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:
        return None
    return x


def _compute_pr_curve(y_true: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    pos = int((y_true == 1).sum())
    if pos <= 0:
        return None

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(pos)

    change = np.empty_like(s, dtype=bool)
    change[0] = True
    change[1:] = s[1:] != s[:-1]
    idx = np.where(change)[0]

    precision = precision[idx]
    recall = recall[idx]

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    auprc = float(np.trapz(precision, recall))
    return recall, precision, auprc


def _compute_roc_curve(y_true: np.ndarray, scores: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    s = scores[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos <= 0 or neg <= 0:
        return None

    tpr = tp / float(pos)
    fpr = fp / float(neg)

    change = np.empty_like(s, dtype=bool)
    change[0] = True
    change[1:] = s[1:] != s[:-1]
    idx = np.where(change)[0]

    tpr = tpr[idx]
    fpr = fpr[idx]

    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def compute_eval_metrics(truth, pred):
    truth = truth.map(normalize_eval_label)
    pred = pred.map(normalize_eval_label)
    valid = truth.isin({"积极", "消极"})
    truth = truth[valid]
    pred = pred[valid]

    tp = int(((truth == "积极") & (pred == "积极")).sum())
    tn = int(((truth == "消极") & (pred == "消极")).sum())
    fp = int(((truth == "消极") & (pred == "积极")).sum())
    fn = int(((truth == "积极") & (pred == "消极")).sum())
    total = int(len(truth))

    accuracy = (tp + tn) / total if total else 0.0

    precision_pos = tp / (tp + fp) if (tp + fp) else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = (
        2 * precision_pos * recall_pos / (precision_pos + recall_pos)
        if (precision_pos + recall_pos)
        else 0.0
    )

    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = (
        2 * precision_neg * recall_neg / (precision_neg + recall_neg)
        if (precision_neg + recall_neg)
        else 0.0
    )

    macro_f1 = (f1_pos + f1_neg) / 2

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def render_model_report(base_dir: str):
    st.title("📑 模型评估报告")

    def _load_csv(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path)

    def _render_one(dataset_name: str, estimate_dir: str, key_candidates):
        sample_path = os.path.join(estimate_dir, "sample_target.csv")
        roberta_pred_path = os.path.join(estimate_dir, "roBERTa_sample_prediction.csv")
        roberta_origin_pred_path = os.path.join(estimate_dir, "roBERTa_origin_sample_prediction.csv")
        roberta_fit_pred_path = os.path.join(estimate_dir, "roBERTa_fit_sample_prediction.csv")
        baseline_pred_path = os.path.join(estimate_dir, "Baseline_sample_prediction.csv")

        if not (os.path.exists(sample_path) and os.path.exists(roberta_pred_path) and os.path.exists(baseline_pred_path)):
            st.info("未找到 sample_target.csv / roBERTa_sample_prediction.csv / Baseline_sample_prediction.csv，无法绘制评估图表")
            return

        sample_df = _load_csv(sample_path)
        roberta_pred_df = _load_csv(roberta_pred_path)
        baseline_pred_df = _load_csv(baseline_pred_path)

        roberta_origin_pred_df = None
        if os.path.exists(roberta_origin_pred_path):
            roberta_origin_pred_df = _load_csv(roberta_origin_pred_path)

        roberta_fit_pred_df = None
        if os.path.exists(roberta_fit_pred_path):
            roberta_fit_pred_df = _load_csv(roberta_fit_pred_path)

        keys = [
            c
            for c in key_candidates
            if c in sample_df.columns and c in roberta_pred_df.columns and c in baseline_pred_df.columns
        ]

        if not (
            keys
            and "sentiment_label" in sample_df.columns
            and "sentiment_label" in roberta_pred_df.columns
            and "sentiment_label" in baseline_pred_df.columns
        ):
            st.info("评估数据列不完整，无法绘制评估图表")
            return

        eval_df = (
            sample_df[keys + ["sentiment_label"]]
            .merge(
                roberta_pred_df[keys + ["sentiment_label"]].rename(columns={"sentiment_label": "roBERTa_label"}),
                on=keys,
                how="left",
            )
            .merge(
                roberta_pred_df[keys + ([c for c in ["sentiment_score"] if c in roberta_pred_df.columns])].rename(
                    columns={"sentiment_score": "roBERTa_score"}
                )
                if "sentiment_score" in roberta_pred_df.columns
                else roberta_pred_df[keys].assign(roBERTa_score=pd.NA),
                on=keys,
                how="left",
            )
            .merge(
                baseline_pred_df[keys + ["sentiment_label"]].rename(columns={"sentiment_label": "SnowNLP_label"}),
                on=keys,
                how="left",
            )
            .merge(
                baseline_pred_df[keys + ([c for c in ["sentiment_score"] if c in baseline_pred_df.columns])].rename(
                    columns={"sentiment_score": "SnowNLP_score"}
                )
                if "sentiment_score" in baseline_pred_df.columns
                else baseline_pred_df[keys].assign(SnowNLP_score=pd.NA),
                on=keys,
                how="left",
            )
        )

        if roberta_origin_pred_df is not None:
            origin_keys = [k for k in keys if k in roberta_origin_pred_df.columns]
            if origin_keys and "sentiment_label" in roberta_origin_pred_df.columns:
                eval_df = eval_df.merge(
                    roberta_origin_pred_df[origin_keys + ["sentiment_label"]].rename(
                        columns={"sentiment_label": "roBERTa_origin_label"}
                    ),
                    on=origin_keys,
                    how="left",
                )
            if origin_keys and "sentiment_score" in roberta_origin_pred_df.columns:
                eval_df = eval_df.merge(
                    roberta_origin_pred_df[origin_keys + ["sentiment_score"]].rename(
                        columns={"sentiment_score": "roBERTa_origin_score"}
                    ),
                    on=origin_keys,
                    how="left",
                )

        roberta_fit_label_source = None
        roberta_fit_score_source = None
        if roberta_fit_pred_df is not None:
            if "sentiment_label" in roberta_fit_pred_df.columns:
                roberta_fit_label_source = "sentiment_label"
            elif "model_sentiment_label" in roberta_fit_pred_df.columns:
                roberta_fit_label_source = "model_sentiment_label"

            if "sentiment_score" in roberta_fit_pred_df.columns:
                roberta_fit_score_source = "sentiment_score"
            elif "model_positive_prob" in roberta_fit_pred_df.columns:
                roberta_fit_score_source = "model_positive_prob"

        fit_keys = (
            [k for k in keys if roberta_fit_pred_df is not None and k in roberta_fit_pred_df.columns]
            if roberta_fit_pred_df is not None
            else []
        )
        if roberta_fit_pred_df is not None and roberta_fit_label_source is not None and fit_keys:
            eval_df = eval_df.merge(
                roberta_fit_pred_df[fit_keys + [roberta_fit_label_source]].rename(
                    columns={roberta_fit_label_source: "roBERTa_fit_label"}
                ),
                on=fit_keys,
                how="left",
            )
            if roberta_fit_score_source is not None:
                eval_df = eval_df.merge(
                    roberta_fit_pred_df[fit_keys + [roberta_fit_score_source]].rename(
                        columns={roberta_fit_score_source: "roBERTa_fit_score"}
                    ),
                    on=fit_keys,
                    how="left",
                )
            else:
                eval_df["roBERTa_fit_score"] = pd.NA

        st.caption(f"数据源：{dataset_name} · 评估样本数：{int(len(eval_df))}")

        roberta_m = compute_eval_metrics(eval_df["sentiment_label"], eval_df["roBERTa_label"])
        snow_m = compute_eval_metrics(eval_df["sentiment_label"], eval_df["SnowNLP_label"])
        roberta_origin_m = None
        if "roBERTa_origin_label" in eval_df.columns:
            roberta_origin_m = compute_eval_metrics(eval_df["sentiment_label"], eval_df["roBERTa_origin_label"])
        roberta_fit_m = None
        if "roBERTa_fit_label" in eval_df.columns:
            roberta_fit_m = compute_eval_metrics(eval_df["sentiment_label"], eval_df["roBERTa_fit_label"])

        st.subheader("📈 P-R 曲线 / ROC 曲线（多模型对比）")

        truth_int = eval_df["sentiment_label"].map(_label_to_int)
        models = []

        def _collect_model(name: str, label_col: str, score_col: str):
            if label_col not in eval_df.columns:
                return
            pred_int = eval_df[label_col].map(_label_to_int)
            score_raw = eval_df[score_col] if score_col in eval_df.columns else pd.Series([pd.NA] * len(eval_df))
            score = score_raw.map(_to_float_or_none)
            if score.isna().all():
                score = pred_int.map(lambda x: float(x) if x is not None else None)
            dfm = pd.DataFrame({"y": truth_int, "pred": pred_int, "score": score})
            dfm = dfm.dropna(subset=["y", "score"])
            if dfm.empty:
                return
            y = dfm["y"].astype(int).to_numpy()
            s = dfm["score"].astype(float).to_numpy()
            pr = _compute_pr_curve(y, s)
            roc = _compute_roc_curve(y, s)
            if pr is None or roc is None:
                return
            recall, precision, auprc = pr
            fpr, tpr, auc = roc
            models.append(
                {
                    "name": name,
                    "recall": recall,
                    "precision": precision,
                    "auprc": float(auprc),
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": float(auc),
                }
            )

        _collect_model("roBERTa", "roBERTa_label", "roBERTa_score")
        if roberta_origin_m is not None:
            _collect_model("roBERTa (Origin)", "roBERTa_origin_label", "roBERTa_origin_score")
        if roberta_fit_m is not None:
            _collect_model("roBERTa (Fine-tuned)", "roBERTa_fit_label", "roBERTa_fit_score")
        _collect_model("SnowNLP", "SnowNLP_label", "SnowNLP_score")

        if not models:
            st.info("无法绘制曲线：缺少可用的预测分数（sentiment_score）或有效标签。")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
            for m in models:
                ax1.plot(m["recall"], m["precision"], linewidth=2, label=f'{m["name"]} (AUPRC={m["auprc"]:.3f})')
            ax1.set_title("Precision-Recall")
            ax1.set_xlabel("Recall")
            ax1.set_ylabel("Precision")
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8, loc="lower left")

            for m in models:
                ax2.plot(m["fpr"], m["tpr"], linewidth=2, label=f'{m["name"]} (AUC={m["auc"]:.3f})')
            ax2.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
            ax2.set_title("ROC")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, loc="lower right")

            st.pyplot(fig, clear_figure=True)

        st.subheader("🧾 混淆矩阵")
        truth = eval_df["sentiment_label"].map(normalize_eval_label)
        roberta_pred = eval_df["roBERTa_label"].map(normalize_eval_label)
        roberta_origin_pred = (
            eval_df["roBERTa_origin_label"].map(normalize_eval_label)
            if "roBERTa_origin_label" in eval_df.columns
            else None
        )
        snow_pred = eval_df["SnowNLP_label"].map(normalize_eval_label)
        roberta_fit_pred = (
            eval_df["roBERTa_fit_label"].map(normalize_eval_label)
            if "roBERTa_fit_label" in eval_df.columns
            else None
        )

        valid_truth = truth.isin({"积极", "消极"})
        valid_roberta = roberta_pred.isin({"积极", "消极"})
        valid_origin = (
            roberta_origin_pred.isin({"积极", "消极"}) if roberta_origin_pred is not None else None
        )
        valid_snow = snow_pred.isin({"积极", "消极"})
        valid_fit = roberta_fit_pred.isin({"积极", "消极"}) if roberta_fit_pred is not None else None

        cm_roberta = pd.crosstab(
            truth[valid_truth & valid_roberta],
            roberta_pred[valid_truth & valid_roberta],
            rownames=["真值"],
            colnames=["roBERTa 预测"],
            dropna=False,
        ).reindex(index=["积极", "消极"], columns=["积极", "消极"], fill_value=0)

        cm_snow = pd.crosstab(
            truth[valid_truth & valid_snow],
            snow_pred[valid_truth & valid_snow],
            rownames=["真值"],
            colnames=["SnowNLP 预测"],
            dropna=False,
        ).reindex(index=["积极", "消极"], columns=["积极", "消极"], fill_value=0)

        cm_origin = None
        if roberta_origin_pred is not None and valid_origin is not None:
            cm_origin = pd.crosstab(
                truth[valid_truth & valid_origin],
                roberta_origin_pred[valid_truth & valid_origin],
                rownames=["真值"],
                colnames=["roBERTa (Origin) 预测"],
                dropna=False,
            ).reindex(index=["积极", "消极"], columns=["积极", "消极"], fill_value=0)

        if cm_origin is not None and roberta_fit_pred is not None and valid_fit is not None:
            cm_fit = pd.crosstab(
                truth[valid_truth & valid_fit],
                roberta_fit_pred[valid_truth & valid_fit],
                rownames=["真值"],
                colnames=["roBERTa (Fine-tuned) 预测"],
                dropna=False,
            ).reindex(index=["积极", "消极"], columns=["积极", "消极"], fill_value=0)

            cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
            with cm_col1:
                st.write("roBERTa")
                st.dataframe(cm_roberta, use_container_width=True)
            with cm_col2:
                st.write("roBERTa (Origin)")
                st.dataframe(cm_origin, use_container_width=True)
            with cm_col3:
                st.write("roBERTa (Fine-tuned)")
                st.dataframe(cm_fit, use_container_width=True)
            with cm_col4:
                st.write("SnowNLP")
                st.dataframe(cm_snow, use_container_width=True)
        elif cm_origin is not None:
            cm_col1, cm_col2, cm_col3 = st.columns(3)
            with cm_col1:
                st.write("roBERTa")
                st.dataframe(cm_roberta, use_container_width=True)
            with cm_col2:
                st.write("roBERTa (Origin)")
                st.dataframe(cm_origin, use_container_width=True)
            with cm_col3:
                st.write("SnowNLP")
                st.dataframe(cm_snow, use_container_width=True)
        elif roberta_fit_pred is not None and valid_fit is not None:
            cm_fit = pd.crosstab(
                truth[valid_truth & valid_fit],
                roberta_fit_pred[valid_truth & valid_fit],
                rownames=["真值"],
                colnames=["roBERTa (Fine-tuned) 预测"],
                dropna=False,
            ).reindex(index=["积极", "消极"], columns=["积极", "消极"], fill_value=0)

            cm_col1, cm_col2, cm_col3 = st.columns(3)
            with cm_col1:
                st.write("roBERTa")
                st.dataframe(cm_roberta, use_container_width=True)
            with cm_col2:
                st.write("roBERTa (Fine-tuned)")
                st.dataframe(cm_fit, use_container_width=True)
            with cm_col3:
                st.write("SnowNLP")
                st.dataframe(cm_snow, use_container_width=True)
        else:
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                st.write("roBERTa")
                st.dataframe(cm_roberta, use_container_width=True)
            with cm_col2:
                st.write("SnowNLP")
                st.dataframe(cm_snow, use_container_width=True)

        report_path = os.path.join(estimate_dir, "estimate_report.html")
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_html = f.read()
                components.html(report_html, height=900, scrolling=True)
            except Exception as e:
                _ = e
                st.error("读取评估报告失败。")

    tab_weibo, tab_zhihu = st.tabs(["微博", "知乎"])
    with tab_weibo:
        _render_one(
            dataset_name="微博",
            estimate_dir=os.path.join(base_dir, "model_estimate", "weibo"),
            key_candidates=["weibo_id", "user_name", "publish_time", "cleaned_text"],
        )
    with tab_zhihu:
        _render_one(
            dataset_name="知乎",
            estimate_dir=os.path.join(base_dir, "model_estimate", "zhihu"),
            key_candidates=["answer_id"],
        )
