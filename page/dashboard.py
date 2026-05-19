import os
from collections import Counter

import jieba
import pandas as pd
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, WordCloud

from page.common import load_data, st_pyecharts


def _read_csv(path: str):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def _find_peak_band_indices(values: pd.Series, peak_idx: int, ratio: float):
    if values.empty:
        return 0, -1
    peak_v = float(values.iloc[int(peak_idx)])
    if peak_v <= 0:
        return int(peak_idx), int(peak_idx)
    band = values >= (peak_v * float(ratio))
    left = int(peak_idx)
    while left - 1 >= 0 and bool(band.iloc[left - 1]):
        left -= 1
    right = int(peak_idx)
    while right + 1 < len(band) and bool(band.iloc[right + 1]):
        right += 1
    return left, right


def _compute_opinion_cycle(daily_counts: pd.Series, smooth_window: int, growth_ratio: float, min_threshold: int):
    if daily_counts is None or daily_counts.empty:
        return {"stages": [], "summary": {}}

    daily_counts = daily_counts.astype(float)
    smooth = daily_counts.rolling(window=int(smooth_window), min_periods=1, center=True).mean()

    smooth_arr = smooth.to_numpy()
    peak_idx = int(smooth_arr.argmax()) if len(smooth_arr) else 0
    peak_value = float(smooth.iloc[peak_idx]) if len(smooth_arr) else 0.0
    n = int(len(smooth))

    head_n = max(3, int(n * 0.2))
    baseline = float(smooth.iloc[:head_n].median()) if n else 0.0
    threshold = max(float(min_threshold), baseline * float(growth_ratio))

    stages = []
    if peak_value < threshold or n < 3:
        stages.append({"name": "平稳期", "start_i": 0, "end_i": n - 1, "color": "#93c5fd"})
    else:
        peak_start, peak_end = _find_peak_band_indices(smooth, peak_idx=peak_idx, ratio=0.9)

        outbreak_candidates = (smooth.iloc[: peak_start + 1] >= threshold).to_numpy()
        outbreak_start = int(outbreak_candidates.argmax()) if bool(outbreak_candidates.any()) else 0

        if outbreak_start > 0:
            stages.append(
                {"name": "潜伏期", "start_i": 0, "end_i": outbreak_start - 1, "color": "#cbd5e1"}
            )

        if outbreak_start <= peak_start - 1:
            stages.append(
                {
                    "name": "爆发期",
                    "start_i": outbreak_start,
                    "end_i": peak_start - 1,
                    "color": "#fdba74",
                }
            )

        stages.append({"name": "高峰期", "start_i": peak_start, "end_i": peak_end, "color": "#f87171"})

        decline_start = peak_end + 1
        if decline_start <= n - 1:
            decline_candidates = (smooth.iloc[decline_start:] >= threshold).to_numpy()
            if bool(decline_candidates.any()):
                last_true_rel = int(len(decline_candidates) - 1 - decline_candidates[::-1].argmax())
                decline_end = int(decline_start + last_true_rel)
            else:
                decline_end = int(n - 1)
            stages.append(
                {"name": "衰退期", "start_i": decline_start, "end_i": decline_end, "color": "#86efac"}
            )

            if decline_end < n - 1:
                stages.append(
                    {"name": "长尾期", "start_i": decline_end + 1, "end_i": n - 1, "color": "#a5b4fc"}
                )

    last_i = n - 1
    current_stage = None
    for s in stages:
        if int(s["start_i"]) <= last_i <= int(s["end_i"]):
            current_stage = s["name"]
            break

    summary = {
        "start_day": str(daily_counts.index[0].date()),
        "end_day": str(daily_counts.index[-1].date()),
        "days": int(n),
        "peak_day": str(daily_counts.index[peak_idx].date()),
        "peak_count": int(daily_counts.iloc[peak_idx]),
        "baseline": float(baseline),
        "threshold": float(threshold),
        "current_stage": current_stage or "",
    }
    return {"stages": stages, "summary": summary, "smooth": smooth, "raw": daily_counts}


def render_dashboard(base_dir: str):
    st.sidebar.header("⚙️ 平台选择")
    platform = st.sidebar.radio("选择平台", options=["微博 (Weibo)", "知乎 (Zhihu)"])

    st.sidebar.header("📦 数据来源")
    data_source = st.sidebar.radio("选择数据来源", options=["采集数据", "导入数据"], index=0)

    st.sidebar.header("🧠 模型选择")
    model_choice = st.sidebar.selectbox(
        "选择情感分析模型",
        options=["基线 (Baseline)", "roBERTa", "roBERTa (Fine-tuned)"],
        index=1,
    )

    if platform == "微博 (Weibo)":
        prediction_dir = os.path.join(base_dir, "model_prediction", "weibo", "prediction")
        default_bertopic_dir = (
            os.path.join(prediction_dir, "bertopic_output")
            if data_source == "采集数据"
            else os.path.join(prediction_dir, "bertopic_output_import.csv")
        )
        title_suffix = "微博"
    else:
        prediction_dir = os.path.join(base_dir, "model_prediction", "zhihu", "prediction")
        default_bertopic_dir = (
            os.path.join(prediction_dir, "bertopic_output")
            if data_source == "采集数据"
            else os.path.join(prediction_dir, "bertopic_output_import.csv")
        )
        title_suffix = "知乎"

    file_suffix = "" if data_source == "采集数据" else "_import"
    model_file_map = {
        "基线 (Baseline)": f"Baseline_prediction{file_suffix}.csv",
        "roBERTa": f"roBERTa_prediction{file_suffix}.csv",
        "roBERTa (Fine-tuned)": f"roBERTa_fit_prediction{file_suffix}.csv",
    }
    data_path = os.path.join(
        prediction_dir,
        model_file_map.get(model_choice, f"roBERTa_prediction{file_suffix}.csv"),
    )

    if platform == "知乎 (Zhihu)":
        df = _read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame()
    else:
        df = load_data(
            data_path,
            os.path.getmtime(data_path) if os.path.exists(data_path) else 0,
        )

    if not df.empty:
        if "author_name" in df.columns and "user_name" not in df.columns:
            df = df.rename(columns={"author_name": "user_name"})
        if "content" in df.columns and "cleaned_text" not in df.columns:
            df = df.rename(columns={"content": "cleaned_text"})

    if platform == "知乎 (Zhihu)" and not df.empty:
        try:
            if "question_id" in df.columns:
                df = df.copy()
                df["question_id"] = df["question_id"].fillna("").astype(str).str.strip()
        except Exception:
            pass

        need_publish_time = ("publish_time" not in df.columns) or df["publish_time"].isna().all()
        if need_publish_time:
            try:
                questions_path = (
                    os.path.join(base_dir, "dataset", "zhihu_questions_cleaned.csv")
                    if data_source == "采集数据"
                    else os.path.join(base_dir, "dataset", "zhihu_questions_import.csv")
                )
                if os.path.exists(questions_path) and "question_id" in df.columns:
                    q = _read_csv(questions_path)
                    if "publish_time" in q.columns and "question_id" in q.columns:
                        q = q.copy()
                        q["question_id"] = q["question_id"].fillna("").astype(str).str.strip()
                        q["publish_time"] = pd.to_datetime(q["publish_time"], errors="coerce")
                        df = df.merge(q[["question_id", "publish_time"]], on="question_id", how="left")
            except Exception:
                pass

        if "publish_time" in df.columns:
            df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

        if "publish_time" not in df.columns or df["publish_time"].isna().all():
            if "created_time" in df.columns:
                df = df.copy()
                df["publish_time"] = pd.to_datetime(df["created_time"], errors="coerce")

        if "publish_time" in df.columns and df["publish_time"].notna().any():
            df = df.dropna(subset=["publish_time"])

    if df.empty:
        st.warning("暂无数据，请检查数据文件是否存在且格式正确。")
        st.stop()

    st.sidebar.header("🔍 筛选条件")

    min_date = df["publish_time"].min().date()
    max_date = df["publish_time"].max().date()

    if min_date == max_date:
        start_date, end_date = min_date, max_date
        st.sidebar.info(f"数据仅包含日期: {min_date}")
    else:
        date_range = st.sidebar.date_input(
            "选择日期范围",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = date_range[0], date_range[0]

    sentiment_options = df["sentiment_label"].unique().tolist()
    selected_sentiments = st.sidebar.multiselect(
        "情感倾向",
        options=sentiment_options,
        default=sentiment_options,
    )

    search_keyword = st.sidebar.text_input("关键词搜索 (在评论内容中)")

    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df["publish_time"].dt.date >= start_date)
        & (filtered_df["publish_time"].dt.date <= end_date)
    ]

    if selected_sentiments:
        filtered_df = filtered_df[filtered_df["sentiment_label"].isin(selected_sentiments)]

    if search_keyword:
        filtered_df = filtered_df[
            filtered_df["cleaned_text"].astype(str).str.contains(search_keyword, na=False)
        ]

    if "sentiment_label" in filtered_df.columns and "sentiment_score" in filtered_df.columns:
        raw_scores = (
            pd.to_numeric(filtered_df["sentiment_score"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
        labels = filtered_df["sentiment_label"].astype(str)

        pos_mask = labels.eq("积极")
        neg_mask = labels.eq("消极")
        directional_ok = 0.0
        if int(pos_mask.sum()) > 0:
            directional_ok += float((raw_scores[pos_mask] >= 0.5).mean())
        if int(neg_mask.sum()) > 0:
            directional_ok += float((raw_scores[neg_mask] <= 0.5).mean())
        directional_ok = directional_ok / max(
            1,
            int((int(pos_mask.sum()) > 0) + (int(neg_mask.sum()) > 0)),
        )

        if directional_ok >= 0.9:
            filtered_df["sentiment_score_display"] = raw_scores
        else:
            display_scores = raw_scores.copy()
            display_scores[pos_mask] = 0.5 + 0.5 * raw_scores[pos_mask]
            display_scores[neg_mask] = 0.5 - 0.5 * raw_scores[neg_mask]
            display_scores[~(pos_mask | neg_mask)] = 0.5
            filtered_df["sentiment_score_display"] = display_scores.clip(0.0, 1.0)

    st.title(f"📊 {title_suffix}舆情分析可视化大屏")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    total_comments = len(filtered_df)
    score_col = (
        "sentiment_score_display"
        if "sentiment_score_display" in filtered_df.columns
        else "sentiment_score"
    )
    avg_sentiment = (
        filtered_df[score_col].mean()
        if not filtered_df.empty and score_col in filtered_df.columns
        else 0
    )
    pos_count = len(filtered_df[filtered_df["sentiment_label"] == "积极"])
    neg_count = len(filtered_df[filtered_df["sentiment_label"] == "消极"])

    with col1:
        st.metric(label="总评论数", value=f"{total_comments:,}")
    with col2:
        st.metric(label="平均情感得分 (0-1)", value=f"{avg_sentiment:.3f}")
    with col3:
        st.metric(
            label="积极评论数",
            value=f"{pos_count:,}",
            delta=f"{pos_count/total_comments*100:.1f}%" if total_comments > 0 else "0%",
        )
    with col4:
        st.metric(
            label="消极评论数",
            value=f"{neg_count:,}",
            delta=f"-{neg_count/total_comments*100:.1f}%" if total_comments > 0 else "0%",
            delta_color="inverse",
        )

    st.markdown("---")

    row1_col1, row1_col2 = st.columns([3, 2])

    with row1_col1:
        st.subheader("📈 情感趋势变化 (每天)")
        if not filtered_df.empty:
            trend_data = (
                filtered_df.set_index("publish_time").resample("D")[score_col].mean().reset_index()
            )
            trend_count = (
                filtered_df.set_index("publish_time").resample("D").size().reset_index(name="count")
            )

            line = (
                Line(init_opts=opts.InitOpts(width="100%", height="400px"))
                .add_xaxis(trend_data["publish_time"].dt.strftime("%Y-%m-%d").tolist())
                .add_yaxis(
                    "平均情感分",
                    trend_data[score_col].round(3).tolist(),
                    is_smooth=True,
                    areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
                    label_opts=opts.LabelOpts(is_show=False),
                    markpoint_opts=opts.MarkPointOpts(
                        data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]
                    ),
                )
                .add_yaxis(
                    "评论数量",
                    trend_count["count"].tolist(),
                    yaxis_index=1,
                    is_smooth=True,
                    linestyle_opts=opts.LineStyleOpts(type_="dashed"),
                )
                .set_global_opts(
                    tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                    xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                    yaxis_opts=opts.AxisOpts(name="情感得分", min_=0, max_=1),
                    datazoom_opts=[opts.DataZoomOpts()],
                    legend_opts=opts.LegendOpts(pos_top="5%"),
                )
                .extend_axis(yaxis=opts.AxisOpts(name="评论数量", position="right"))
            )
            st_pyecharts(line, height="400px")
        else:
            st.info("无数据可显示趋势")

    with row1_col2:
        st.subheader("🍰 情感分布")
        if not filtered_df.empty:
            sentiment_counts = filtered_df["sentiment_label"].value_counts()
            data_pair = [
                list(z)
                for z in zip(
                    sentiment_counts.index.tolist(),
                    sentiment_counts.values.tolist(),
                )
            ]

            pie = (
                Pie(init_opts=opts.InitOpts(width="100%", height="400px"))
                .add("", data_pair, radius=["35%", "65%"], center=["50%", "45%"])
                .set_global_opts(
                    legend_opts=opts.LegendOpts(
                        orient="horizontal",
                        pos_bottom="0%",
                        pos_left="center",
                    ),
                )
                .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
            )
            st_pyecharts(pie, height="400px")
        else:
            st.info("无数据")

    st.markdown("---")
    st.subheader("🌀 舆论周期（基于评论量）")

    if filtered_df.empty:
        st.info("无数据，无法计算舆论周期")
    else:
        smooth_window = 3
        growth_ratio = 1.5
        min_threshold = 5

        daily_counts = filtered_df.set_index("publish_time").resample("D").size()
        cycle = _compute_opinion_cycle(
            daily_counts=daily_counts,
            smooth_window=int(smooth_window),
            growth_ratio=float(growth_ratio),
            min_threshold=int(min_threshold),
        )

        summary = cycle.get("summary") or {}
        stages = cycle.get("stages") or []
        raw = cycle.get("raw")
        smooth = cycle.get("smooth")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("当前阶段", summary.get("current_stage", ""))
        with m2:
            st.metric("峰值日", summary.get("peak_day", ""))
        with m3:
            st.metric("峰值评论量", f"{int(summary.get('peak_count', 0)):,}")
        with m4:
            st.metric("周期跨度(天)", str(summary.get("days", "")))

        if raw is not None and not raw.empty:
            x = [str(d.date()) for d in raw.index]
            y_raw = [int(v) for v in raw.tolist()]
            y_smooth = [float(v) for v in (smooth.tolist() if smooth is not None else raw.tolist())]

            chart = Line(init_opts=opts.InitOpts(width="100%", height="420px"))
            chart.add_xaxis(x)
            chart.add_yaxis(
                "评论量(原始)",
                y_raw,
                is_smooth=False,
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=1, color="#94a3b8"),
            )

            for s in stages:
                start_i = int(s["start_i"])
                end_i = int(s["end_i"])
                data = [None] * len(y_smooth)
                for i in range(start_i, end_i + 1):
                    if 0 <= i < len(data):
                        data[i] = float(y_smooth[i])
                chart.add_yaxis(
                    f"评论量({s['name']})",
                    data,
                    is_smooth=True,
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=3, color=str(s.get("color") or "#60a5fa")),
                )

            chart.set_global_opts(
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False, axislabel_opts=opts.LabelOpts(rotate=35)),
                yaxis_opts=opts.AxisOpts(type_="value", name="评论量"),
                datazoom_opts=[opts.DataZoomOpts()],
                legend_opts=opts.LegendOpts(pos_top="2%", type_="scroll"),
            )
            st_pyecharts(chart, height="420px")

        if stages:
            stage_rows = []
            for s in stages:
                si = int(s["start_i"])
                ei = int(s["end_i"])
                stage_rows.append(
                    {
                        "阶段": s["name"],
                        "开始": str(daily_counts.index[si].date()),
                        "结束": str(daily_counts.index[ei].date()),
                        "天数": int(ei - si + 1),
                    }
                )
            st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, height=220)

    st.markdown("---")

    st.subheader("🏢 企业建议面板")
    if data_source == "导入数据":
        st.info("导入数据暂不支持企业建议面板。")
    else:
        insights_dir = os.path.join(base_dir, "dataset", "enterprise_insights")

        platform_key = "weibo" if platform == "微博 (Weibo)" else "zhihu"
        demands_path = os.path.join(insights_dir, f"enterprise_negative_demands_{platform_key}.csv")
        topics_path = os.path.join(insights_dir, f"enterprise_negative_topics_{platform_key}.csv")
        trend_path = os.path.join(insights_dir, f"enterprise_negative_demand_trend_daily_{platform_key}.csv")

        if not os.path.exists(demands_path):
            st.info(
                "未找到企业洞察 CSV。请先运行 data_analysis/generate_enterprise_insights.py 生成。"
            )
        else:
            demands_df = _read_csv(demands_path)
            topics_df = _read_csv(topics_path) if os.path.exists(topics_path) else pd.DataFrame()
            trend_df = _read_csv(trend_path) if os.path.exists(trend_path) else pd.DataFrame()

            top_k_demands = 12
            top_k_topics = 15

            if not demands_df.empty:
                demands_df = demands_df.copy()
                for c in ["priority_score", "weighted_negative", "negative_share", "negative_count"]:
                    if c in demands_df.columns:
                        demands_df[c] = pd.to_numeric(demands_df[c], errors="coerce")
                demands_df = demands_df.sort_values(["priority_score", "negative_count"], ascending=False).head(int(top_k_demands))

                c1, c2 = st.columns([2, 3])
                with c1:
                    show_demands = demands_df.set_index("demand")[["priority_score"]]
                    st.bar_chart(show_demands)
                with c2:
                    st.dataframe(
                        demands_df[
                            [
                                "demand",
                                "negative_count",
                                "negative_share",
                                "weighted_negative",
                                "priority_score",
                                "suggested_actions",
                                "evidence_1",
                                "evidence_2",
                                "evidence_3",
                            ]
                        ],
                        use_container_width=True,
                        height=360,
                    )

            if not trend_df.empty and {"day", "demand"}.issubset(set(trend_df.columns)):
                trend_df = trend_df.copy()
                trend_df["day"] = trend_df["day"].astype(str)
                if "weighted_neg" in trend_df.columns:
                    trend_df["weighted_neg"] = pd.to_numeric(trend_df["weighted_neg"], errors="coerce").fillna(0.0)
                else:
                    trend_df["weighted_neg"] = 0.0

                days = sorted(trend_df["day"].dropna().unique().tolist())
                if days:
                    st.subheader("📉 负面诉求趋势（Top 诉求）")
                    chart = Line(init_opts=opts.InitOpts(width="100%", height="360px"))
                    chart.add_xaxis(days)
                    for dname, g in trend_df.groupby("demand"):
                        g2 = g.set_index("day").reindex(days).fillna(0.0)
                        y = [float(v) for v in g2["weighted_neg"].tolist()]
                        chart.add_yaxis(
                            str(dname),
                            y,
                            is_smooth=True,
                            label_opts=opts.LabelOpts(is_show=False),
                        )
                    chart.set_global_opts(
                        tooltip_opts=opts.TooltipOpts(trigger="axis"),
                        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False, axislabel_opts=opts.LabelOpts(rotate=35)),
                        yaxis_opts=opts.AxisOpts(type_="value", name="加权负面"),
                        datazoom_opts=[opts.DataZoomOpts()],
                        legend_opts=opts.LegendOpts(pos_top="2%", type_="scroll"),
                    )
                    st_pyecharts(chart, height="360px")

            if not topics_df.empty and {"priority_score", "aspect"}.issubset(set(topics_df.columns)):
                topics_df = topics_df.copy()
                for c in ["priority_score", "negative_count", "total_count"]:
                    if c in topics_df.columns:
                        topics_df[c] = pd.to_numeric(topics_df[c], errors="coerce")
                topics_df = topics_df.sort_values(["priority_score", "negative_count"], ascending=False).head(int(top_k_topics))
                st.subheader("🧩 负面议题 Top 榜（按方面/主题聚类）")
                st.dataframe(
                    topics_df[
                        [
                            "aspect",
                            "topic",
                            "topic_keywords",
                            "total_count",
                            "negative_count",
                            "negative_ratio",
                            "priority_score",
                            "top_demand_1",
                            "top_demand_2",
                            "top_demand_3",
                            "evidence_1",
                        ]
                    ],
                    use_container_width=True,
                    height=320,
                )

    st.markdown("---")

    st.subheader("🧩 主题聚类可视化（BERTopic）")
    output_dir = default_bertopic_dir
    doc_topics_path = os.path.join(output_dir, "doc_topics.csv")
    topics_path = os.path.join(output_dir, "topics.json")
    topic_info_path = os.path.join(output_dir, "topic_info.csv")

    if not os.path.exists(doc_topics_path):
        st.info("未找到 doc_topics.csv，请先运行 model_prediction/BERTopic_Analysis.py 生成主题产物")
    else:
        doc_topics = _read_csv(doc_topics_path)
        doc_topics = doc_topics.dropna(subset=["topic"]).copy()
        if doc_topics.empty:
            st.info("doc_topics.csv 无有效主题数据（topic 为空）")
        else:
            doc_topics["topic"] = pd.to_numeric(doc_topics["topic"], errors="coerce").astype("Int64")
            doc_topics = doc_topics.dropna(subset=["topic"])
            topic_counts = doc_topics["topic"].astype(int).value_counts()

            top_n = 20
            top_topics = topic_counts.head(top_n).reset_index()
            top_topics.columns = ["topic", "count"]
            top_topics["topic"] = top_topics["topic"].astype(int)

            if "aspect" in doc_topics.columns:
                topic_aspect = (
                    doc_topics.groupby("topic")["aspect"]
                    .apply(lambda s: s.astype(str).value_counts().index[0] if len(s) else "其他")
                    .to_dict()
                )
                top_topics["aspect"] = top_topics["topic"].map(topic_aspect).fillna("其他")
            else:
                top_topics["aspect"] = "其他"

            top_words = {}
            if os.path.exists(topics_path):
                try:
                    import json

                    with open(topics_path, "r", encoding="utf-8") as f:
                        topic_words = json.load(f)
                    for k, v in (topic_words or {}).items():
                        if isinstance(v, list):
                            ws = [x.get("word") for x in v if isinstance(x, dict) and x.get("word")]
                            top_words[int(k)] = "、".join(ws[:10])
                except Exception:
                    top_words = {}

            top_topics["top_words"] = top_topics["topic"].map(top_words).fillna("")
            top_topics["topic"] = top_topics["topic"].map(lambda x: f"T{x}")

            topic_col1, topic_col2 = st.columns([3, 2])
            with topic_col1:
                st.dataframe(
                    top_topics[["topic", "count", "aspect", "top_words"]],
                    use_container_width=True,
                    height=380,
                )
            with topic_col2:
                st.bar_chart(top_topics.set_index("topic")["count"])

            if os.path.exists(topic_info_path):
                with st.expander("topic_info.csv", expanded=False):
                    st.dataframe(_read_csv(topic_info_path), use_container_width=True, height=320)

    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        st.subheader("☁️ 评论词云")
        if not filtered_df.empty:
            text_content = " ".join(filtered_df["cleaned_text"].astype(str).tolist())
            words = jieba.cut(text_content)
            stop_words = {
                "的",
                "了",
                "是",
                "在",
                "我",
                "有",
                "和",
                "就",
                "不",
                "人",
                "都",
                "一",
                "一个",
                "上",
                "也",
                "很",
                "到",
                "说",
                "要",
                "去",
                "你",
                "会",
                "着",
                "没有",
                "看",
                "好",
                "自己",
                "这",
            }
            filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
            word_counts = Counter(filtered_words)
            data_pair = word_counts.most_common(100)

            wordcloud = (
                WordCloud(init_opts=opts.InitOpts(width="100%", height="400px"))
                .add("", data_pair, word_size_range=[20, 100])
                .set_global_opts(title_opts=opts.TitleOpts(title="高频词汇"))
            )
            st_pyecharts(wordcloud, height="400px")
        else:
            st.info("无数据生成词云")

    with row2_col2:
        st.subheader("🏆 活跃用户 Top 10")
        if not filtered_df.empty:
            user_counts = (
                filtered_df["user_name"].value_counts().head(10).sort_values(ascending=True)
            )

            bar = (
                Bar(init_opts=opts.InitOpts(width="100%", height="400px"))
                .add_xaxis(user_counts.index.tolist())
                .add_yaxis("评论数", user_counts.values.tolist())
                .reversal_axis()
                .set_series_opts(label_opts=opts.LabelOpts(position="right"))
                .set_global_opts(
                    yaxis_opts=opts.AxisOpts(name="用户"),
                    xaxis_opts=opts.AxisOpts(name="评论数"),
                )
            )
            st_pyecharts(bar, height="400px")
        else:
            st.info("无数据")

    st.markdown("---")
    st.subheader("📋 详细评论列表")
    st.dataframe(
        filtered_df[["publish_time", "user_name", "sentiment_label", score_col, "cleaned_text"]].rename(
            columns={score_col: "sentiment_score"}
        ),
        use_container_width=True,
        height=300,
    )
