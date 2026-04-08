import os
import streamlit as st
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line, Pie, WordCloud, Bar
import streamlit.components.v1 as components
import jieba
from collections import Counter

# --- Helper Function for Pyecharts ---
def st_pyecharts(chart, height="400px", width="100%"):
    """
    Renders a Pyecharts chart in Streamlit using HTML components.
    This replaces streamlit-echarts to avoid compatibility issues.
    """
    # 1. Render the chart to a standalone HTML snippet (div + script)
    #    render_embed() returns the minimal HTML/JS to render the chart.
    chart_html = chart.render_embed()
    
    # 2. Wrap it in a full HTML structure with necessary CDNs
    #    We include ECharts and ECharts-WordCloud CDNs.
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/echarts-wordcloud@2.1.0/dist/echarts-wordcloud.min.js"></script>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: auto; 
            }}
            body > div {{
                width: 100% !important;
            }}
        </style>
    </head>
    <body>
        {chart_html}
    </body>
    </html>
    """
    
    # 3. Display using Streamlit's html component
    #    Adjust height to match the chart's height
    h = int(height.replace("px", "")) if isinstance(height, str) else height
    w = width if isinstance(width, int) else None
    components.html(full_html, height=h, width=w, scrolling=False)

# --- Page Configuration ---
st.set_page_config(
    page_title="微博舆情分析可视化平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Industrial" Look ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stDataFrame {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Parse timestamp: "Thu Dec 25 13:06:29 +0800 2025"
        # Using 'mixed' or explicit format
        df['publish_time'] = pd.to_datetime(df['publish_time'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['publish_time'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

DATA_PATH = "BERT_prediction.csv"
df = load_data(DATA_PATH)

if df.empty:
    st.warning("暂无数据，请检查 BERT_prediction.csv 文件是否存在且格式正确。")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("🔍 筛选条件")

# Date Filter
min_date = df['publish_time'].min().date()
max_date = df['publish_time'].max().date()

# If min and max are the same, allow selecting that single day
if min_date == max_date:
    start_date, end_date = min_date, max_date
    st.sidebar.info(f"数据仅包含日期: {min_date}")
else:
    date_range = st.sidebar.date_input(
        "选择日期范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = date_range[0], date_range[0]

# Sentiment Filter
sentiment_options = df['sentiment_label'].unique().tolist()
selected_sentiments = st.sidebar.multiselect(
    "情感倾向",
    options=sentiment_options,
    default=sentiment_options
)

# Keyword Search
search_keyword = st.sidebar.text_input("关键词搜索 (在评论内容中)")

# --- Data Filtering Logic ---
filtered_df = df.copy()

# Date filter
filtered_df = filtered_df[
    (filtered_df['publish_time'].dt.date >= start_date) & 
    (filtered_df['publish_time'].dt.date <= end_date)
]

# Sentiment filter
if selected_sentiments:
    filtered_df = filtered_df[filtered_df['sentiment_label'].isin(selected_sentiments)]

# Keyword filter
if search_keyword:
    filtered_df = filtered_df[filtered_df['cleaned_text'].astype(str).str.contains(search_keyword, na=False)]

# --- Main Dashboard ---
st.title("📊 微博舆情分析可视化大屏")
st.markdown("---")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

total_comments = len(filtered_df)
avg_sentiment = filtered_df['sentiment_score'].mean() if not filtered_df.empty else 0
pos_count = len(filtered_df[filtered_df['sentiment_label'] == '积极'])
neg_count = len(filtered_df[filtered_df['sentiment_label'] == '消极'])

with col1:
    st.metric(label="总评论数", value=f"{total_comments:,}")
with col2:
    st.metric(label="平均情感得分 (0-1)", value=f"{avg_sentiment:.3f}")
with col3:
    st.metric(label="积极评论数", value=f"{pos_count:,}", delta=f"{pos_count/total_comments*100:.1f}%" if total_comments > 0 else "0%")
with col4:
    st.metric(label="消极评论数", value=f"{neg_count:,}", delta=f"-{neg_count/total_comments*100:.1f}%" if total_comments > 0 else "0%", delta_color="inverse")

st.markdown("---")

# --- Layout: Row 1 (Trend & Distribution) ---
row1_col1, row1_col2 = st.columns([3, 2])

with row1_col1:
    st.subheader("📈 情感趋势变化 (每小时)")
    if not filtered_df.empty:
        # Resample data by hour
        trend_data = filtered_df.set_index('publish_time').resample('h')['sentiment_score'].mean().reset_index()
        trend_count = filtered_df.set_index('publish_time').resample('h').size().reset_index(name='count')
        
        # Line Chart for Sentiment Score
        line = (
            Line(init_opts=opts.InitOpts(width="100%", height="400px"))
            .add_xaxis(trend_data['publish_time'].dt.strftime('%Y-%m-%d %H:%M').tolist())
            .add_yaxis("平均情感分", trend_data['sentiment_score'].round(3).tolist(), 
                       is_smooth=True, 
                       areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
                       label_opts=opts.LabelOpts(is_show=False),
                       markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]))
            .add_yaxis("评论数量", trend_count['count'].tolist(), yaxis_index=1, is_smooth=True, 
                       linestyle_opts=opts.LineStyleOpts(type_="dashed"))
            .set_global_opts(
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                yaxis_opts=opts.AxisOpts(name="情感得分", min_=0, max_=1),
                datazoom_opts=[opts.DataZoomOpts()],
                legend_opts=opts.LegendOpts(pos_top="5%")
            )
            .extend_axis(yaxis=opts.AxisOpts(name="评论数量", position="right"))
        )
        st_pyecharts(line, height="400px")
    else:
        st.info("无数据可显示趋势")

with row1_col2:
    st.subheader("🍰 情感分布")
    if not filtered_df.empty:
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        data_pair = [list(z) for z in zip(sentiment_counts.index.tolist(), sentiment_counts.values.tolist())]
        
        pie = (
            Pie(init_opts=opts.InitOpts(width="100%", height="400px"))
            .add("", data_pair, radius=["35%", "65%"], center=["50%", "45%"])
            .set_global_opts(
                legend_opts=opts.LegendOpts(orient="horizontal", pos_bottom="0%", pos_left="center"),
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
        )
        st_pyecharts(pie, height="400px")
    else:
        st.info("无数据")

# --- Layout: Row 2 (Word Cloud & Top Users) ---
row2_col1, row2_col2 = st.columns([1, 1])

with row2_col1:
    st.subheader("☁️ 评论词云")
    if not filtered_df.empty:
        # Concatenate all text
        text_content = " ".join(filtered_df['cleaned_text'].astype(str).tolist())
        # Tokenize
        words = jieba.cut(text_content)
        # Filter stop words (basic list)
        stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'} 
        filtered_words = [word for word in words if len(word) > 1 and word not in stop_words]
        word_counts = Counter(filtered_words)
        
        # Word Cloud Data
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
        user_counts = filtered_df['user_name'].value_counts().head(10).sort_values(ascending=True)
        
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

# --- Layout: Data Table ---
st.markdown("---")
st.subheader("📋 详细评论列表")
st.dataframe(
    filtered_df[['publish_time', 'user_name', 'sentiment_label', 'sentiment_score', 'cleaned_text']], 
    use_container_width=True,
    height=300
)

st.markdown("---")
st.subheader("📑 模型评估报告")

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


sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_target.csv")
bert_pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BERT_prediction.csv")
baseline_pred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Baseline_prediction.csv")

if os.path.exists(sample_path) and os.path.exists(bert_pred_path) and os.path.exists(baseline_pred_path):
    try:
        sample_df = pd.read_csv(sample_path, encoding="utf-8-sig")
    except Exception:
        sample_df = pd.read_csv(sample_path)

    try:
        bert_pred_df = pd.read_csv(bert_pred_path, encoding="utf-8-sig")
    except Exception:
        bert_pred_df = pd.read_csv(bert_pred_path)

    try:
        baseline_pred_df = pd.read_csv(baseline_pred_path, encoding="utf-8-sig")
    except Exception:
        baseline_pred_df = pd.read_csv(baseline_pred_path)

    key_cols = ["weibo_id", "user_name", "publish_time", "cleaned_text"]
    keys = [c for c in key_cols if c in sample_df.columns and c in bert_pred_df.columns and c in baseline_pred_df.columns]

    if keys and "sentiment_label" in sample_df.columns and "sentiment_label" in bert_pred_df.columns and "sentiment_label" in baseline_pred_df.columns:
        eval_df = sample_df[keys + ["sentiment_label"]].merge(
            bert_pred_df[keys + ["sentiment_label"]].rename(columns={"sentiment_label": "BERT_label"}),
            on=keys,
            how="left",
        ).merge(
            baseline_pred_df[keys + ["sentiment_label"]].rename(columns={"sentiment_label": "SnowNLP_label"}),
            on=keys,
            how="left",
        )

        bert_m = compute_eval_metrics(eval_df["sentiment_label"], eval_df["BERT_label"])
        snow_m = compute_eval_metrics(eval_df["sentiment_label"], eval_df["SnowNLP_label"])

        metric_df = pd.DataFrame(
            [
                {"模型": "BERT", "准确率": bert_m["accuracy"], "Macro-F1": bert_m["macro_f1"], "积极F1": bert_m["f1_pos"], "消极F1": bert_m["f1_neg"]},
                {"模型": "SnowNLP", "准确率": snow_m["accuracy"], "Macro-F1": snow_m["macro_f1"], "积极F1": snow_m["f1_pos"], "消极F1": snow_m["f1_neg"]},
            ]
        ).set_index("模型")

        st.subheader("📊 评估指标对比")
        st.bar_chart(metric_df[["准确率", "Macro-F1"]])
        st.bar_chart(metric_df[["积极F1", "消极F1"]])

        st.subheader("🧾 混淆矩阵")
        truth = eval_df["sentiment_label"].map(normalize_eval_label)
        bert_pred = eval_df["BERT_label"].map(normalize_eval_label)
        snow_pred = eval_df["SnowNLP_label"].map(normalize_eval_label)

        cm_bert = pd.crosstab(truth, bert_pred, rownames=["真值"], colnames=["BERT 预测"], dropna=False)
        cm_snow = pd.crosstab(truth, snow_pred, rownames=["真值"], colnames=["SnowNLP 预测"], dropna=False)
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.write("BERT")
            st.dataframe(cm_bert, use_container_width=True)
        with cm_col2:
            st.write("SnowNLP")
            st.dataframe(cm_snow, use_container_width=True)
    else:
        st.info("评估数据列不完整，无法绘制评估图表")
else:
    st.info("未找到 sample_target.csv / BERT_prediction.csv / Baseline_prediction.csv，无法绘制评估图表")

report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estimate_report.html")
if os.path.exists(report_path):
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_html = f.read()
        components.html(report_html, height=900, scrolling=True)
    except Exception as e:
        st.error(f"读取评估报告失败: {e}")
else:
    st.info("未找到 estimate_report.html，请先运行：/usr/bin/python3 Estimate.py")
