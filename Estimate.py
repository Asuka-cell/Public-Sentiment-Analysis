import json
import os
from datetime import datetime

import pandas as pd


def normalize_sentiment_label(value):
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


def compute_binary_metrics(truth, pred, positive_label="积极", negative_label="消极"):
    truth = truth.map(normalize_sentiment_label)
    pred = pred.map(normalize_sentiment_label)

    valid_truth = truth.isin({positive_label, negative_label})
    truth = truth[valid_truth]
    pred = pred[valid_truth]

    tp = int(((truth == positive_label) & (pred == positive_label)).sum())
    tn = int(((truth == negative_label) & (pred == negative_label)).sum())
    fp = int(((truth == negative_label) & (pred == positive_label)).sum())
    fn = int(((truth == positive_label) & (pred == negative_label)).sum())
    unk = int((~pred.isin({positive_label, negative_label})).sum())

    total = int(len(truth))
    correct = tp + tn

    accuracy = correct / total if total else 0.0

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
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "unknown_pred": unk,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "macro_f1": macro_f1,
    }


def load_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def merge_predictions(sample_df, pred_df, pred_label_col, pred_score_col, prefix):
    key_cols = ["weibo_id", "user_name", "publish_time", "cleaned_text"]
    keys = [c for c in key_cols if c in sample_df.columns and c in pred_df.columns]
    if not keys:
        raise ValueError("无法找到可用于对齐样本与预测结果的共同键列")

    cols = keys + [c for c in [pred_label_col, pred_score_col] if c in pred_df.columns]
    pred_df = pred_df[cols].drop_duplicates(subset=keys, keep="first")

    rename_map = {}
    if pred_label_col in pred_df.columns:
        rename_map[pred_label_col] = f"{prefix}_label"
    if pred_score_col in pred_df.columns:
        rename_map[pred_score_col] = f"{prefix}_score"
    pred_df = pred_df.rename(columns=rename_map)

    merged = sample_df.merge(pred_df, on=keys, how="left")
    return merged, keys


def generate_html_report(
    merged_df,
    keys,
    truth_col,
    bert_metrics,
    baseline_metrics,
    report_path,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    view_cols = keys + [truth_col, "BERT_label", "BERT_score", "SnowNLP_label", "SnowNLP_score"]
    existing_view_cols = [c for c in view_cols if c in merged_df.columns]
    df_view = merged_df[existing_view_cols].copy()

    if truth_col in df_view.columns:
        df_view[truth_col] = df_view[truth_col].map(normalize_sentiment_label)
    if "BERT_label" in df_view.columns:
        df_view["BERT_label"] = df_view["BERT_label"].map(normalize_sentiment_label)
    if "SnowNLP_label" in df_view.columns:
        df_view["SnowNLP_label"] = df_view["SnowNLP_label"].map(normalize_sentiment_label)

    def is_error_row(row):
        truth = row.get(truth_col, "未知")
        b = row.get("BERT_label", "未知")
        s = row.get("SnowNLP_label", "未知")
        if truth not in {"积极", "消极"}:
            return True
        return (b != truth) or (s != truth)

    error_mask = df_view.apply(is_error_row, axis=1)
    error_rows = df_view[error_mask].fillna("").to_dict(orient="records")

    def fmt_pct(v):
        return f"{v*100:.2f}%"

    def fmt_float(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return ""

    bert_summary = {
        "准确率": fmt_pct(bert_metrics["accuracy"]),
        "Macro-F1": fmt_float(bert_metrics["macro_f1"]),
        "积极F1": fmt_float(bert_metrics["f1_pos"]),
        "消极F1": fmt_float(bert_metrics["f1_neg"]),
        "未知预测数": str(bert_metrics["unknown_pred"]),
        "TP": str(bert_metrics["tp"]),
        "FP": str(bert_metrics["fp"]),
        "FN": str(bert_metrics["fn"]),
        "TN": str(bert_metrics["tn"]),
    }

    baseline_summary = {
        "准确率": fmt_pct(baseline_metrics["accuracy"]),
        "Macro-F1": fmt_float(baseline_metrics["macro_f1"]),
        "积极F1": fmt_float(baseline_metrics["f1_pos"]),
        "消极F1": fmt_float(baseline_metrics["f1_neg"]),
        "未知预测数": str(baseline_metrics["unknown_pred"]),
        "TP": str(baseline_metrics["tp"]),
        "FP": str(baseline_metrics["fp"]),
        "FN": str(baseline_metrics["fn"]),
        "TN": str(baseline_metrics["tn"]),
    }

    data_json = json.dumps(
        {
            "timestamp": timestamp,
            "total_samples": int(len(merged_df)),
            "error_rows": error_rows,
            "bert_summary": bert_summary,
            "snownlp_summary": baseline_summary,
        },
        ensure_ascii=False,
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>情感分析对比评估报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; margin: 20px; color: #111827; }}
    .muted {{ color: #6b7280; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; background: #ffffff; }}
    .title {{ font-size: 20px; font-weight: 700; margin: 0 0 8px 0; }}
    .h2 {{ font-size: 16px; font-weight: 700; margin: 14px 0 8px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #f9fafb; position: sticky; top: 0; }}
    .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    input[type="text"] {{ padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; min-width: 260px; }}
    select {{ padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 8px; }}
    label {{ font-size: 13px; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid #e5e7eb; }}
    .pos {{ background: #ecfdf5; border-color: #a7f3d0; }}
    .neg {{ background: #eff6ff; border-color: #bfdbfe; }}
    .unk {{ background: #fff7ed; border-color: #fed7aa; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="title">情感分析对比评估报告</div>
    <div class="muted">生成时间：<span id="ts"></span> · 样本数：<span id="n"></span></div>
  </div>

  <div class="grid" style="margin-top: 14px;">
    <div class="card">
      <div class="h2">BERT</div>
      <table id="tbl_bert"></table>
    </div>
    <div class="card">
      <div class="h2">SnowNLP</div>
      <table id="tbl_snownlp"></table>
    </div>
  </div>

  <div class="card" style="margin-top: 14px;">
    <div class="h2">错误样本查看（可搜索/过滤）</div>
    <div class="controls">
      <input id="q" type="text" placeholder="搜索：文本/用户/ID/标签..." />
      <select id="model">
        <option value="all">显示全部</option>
        <option value="bert">仅看 BERT 错误</option>
        <option value="snownlp">仅看 SnowNLP 错误</option>
      </select>
      <label><input id="only_err" type="checkbox" checked /> 只显示预测错误（真值为积极/消极）</label>
      <span class="muted">共 <span id="err_count"></span> 行</span>
    </div>
    <div style="margin-top: 10px; overflow: auto; max-height: 520px;">
      <table>
        <thead>
          <tr id="thead"></tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const payload = {data_json};
    document.getElementById("ts").textContent = payload.timestamp;
    document.getElementById("n").textContent = payload.total_samples;

    function renderSummary(tableId, titleMap) {{
      const tbl = document.getElementById(tableId);
      const rows = Object.entries(titleMap).map(([k,v]) => `<tr><th style="width: 140px;">${{k}}</th><td class="mono">${{v}}</td></tr>`).join("");
      tbl.innerHTML = `<tbody>${{rows}}</tbody>`;
    }}

    renderSummary("tbl_bert", payload.bert_summary);
    renderSummary("tbl_snownlp", payload.snownlp_summary);

    const rows = payload.error_rows || [];
    const cols = rows.length ? Object.keys(rows[0]) : [];

    const thead = document.getElementById("thead");
    thead.innerHTML = cols.map(c => `<th>${{c}}</th>`).join("");

    function badge(val) {{
      if (val === "积极") return `<span class="badge pos">${{val}}</span>`;
      if (val === "消极") return `<span class="badge neg">${{val}}</span>`;
      if (!val) return "";
      return `<span class="badge unk">${{val}}</span>`;
    }}

    function norm(v) {{
      return (v ?? "").toString().toLowerCase();
    }}

    function isBinaryTruth(v) {{
      return v === "积极" || v === "消极";
    }}

    function renderTable() {{
      const q = norm(document.getElementById("q").value);
      const model = document.getElementById("model").value;
      const onlyErr = document.getElementById("only_err").checked;

      const filtered = rows.filter(r => {{
        const truth = r["{truth_col}"];
        const bert = r["BERT_label"];
        const snow = r["SnowNLP_label"];

        if (onlyErr && isBinaryTruth(truth)) {{
          const bertOk = (bert === truth);
          const snowOk = (snow === truth);
          if (model === "bert" && bertOk) return false;
          if (model === "snownlp" && snowOk) return false;
          if (model === "all" && bertOk && snowOk) return false;
        }} else if (onlyErr && !isBinaryTruth(truth)) {{
          return false;
        }} else {{
          if (model === "bert" && bert === truth) return false;
          if (model === "snownlp" && snow === truth) return false;
        }}

        if (!q) return true;
        return cols.some(c => norm(r[c]).includes(q));
      }});

      document.getElementById("err_count").textContent = filtered.length;

      const tbody = document.getElementById("tbody");
      tbody.innerHTML = filtered.map(r => {{
        const tds = cols.map(c => {{
          const v = r[c];
          if (c.endsWith("_label") || c === "{truth_col}") return `<td>${{badge(v)}}</td>`;
          return `<td>${{(v ?? "").toString()}}</td>`;
        }}).join("");
        return `<tr>${{tds}}</tr>`;
      }}).join("");
    }}

    document.getElementById("q").addEventListener("input", renderTable);
    document.getElementById("model").addEventListener("change", renderTable);
    document.getElementById("only_err").addEventListener("change", renderTable);
    renderTable();
  </script>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(base_dir, "sample_target.csv")
    bert_path = os.path.join(base_dir, "BERT_prediction.csv")
    baseline_path = os.path.join(base_dir, "Baseline_prediction.csv")
    report_path = os.path.join(base_dir, "estimate_report.html")

    for p in [sample_path, bert_path, baseline_path]:
        if not os.path.exists(p):
            print(f"文件不存在: {p}")
            return

    sample_df = load_csv(sample_path)
    bert_df = load_csv(bert_path)
    baseline_df = load_csv(baseline_path)

    truth_col = "sentiment_label"
    if truth_col not in sample_df.columns:
        print(f"样本文件缺少列: {truth_col}")
        return

    merged, keys = merge_predictions(
        sample_df,
        bert_df,
        pred_label_col="sentiment_label",
        pred_score_col="sentiment_score",
        prefix="BERT",
    )

    merged, _ = merge_predictions(
        merged,
        baseline_df,
        pred_label_col="sentiment_label",
        pred_score_col="sentiment_score",
        prefix="SnowNLP",
    )

    if "BERT_label" not in merged.columns or "SnowNLP_label" not in merged.columns:
        print("预测文件缺少 sentiment_label 列，无法评估")
        return

    bert_metrics = compute_binary_metrics(merged[truth_col], merged["BERT_label"])
    baseline_metrics = compute_binary_metrics(merged[truth_col], merged["SnowNLP_label"])

    generate_html_report(
        merged_df=merged,
        keys=keys,
        truth_col=truth_col,
        bert_metrics=bert_metrics,
        baseline_metrics=baseline_metrics,
        report_path=report_path,
    )

    print(f"评估报告已生成: {report_path}")


if __name__ == "__main__":
    main()
