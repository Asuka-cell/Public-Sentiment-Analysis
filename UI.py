import os
import streamlit as st
from page.dashboard import render_dashboard
from page.data_collection import render_data_collection
from page.labeling import render_labeling
from page.model_report import render_model_report

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

page = st.sidebar.radio(
    "页面",
    [
        "舆情分析数据采集",
        "舆情分析可视化大屏",
        "抽样样本标注",
        "情感分析模型评估报告",
    ],
)

base_dir = os.path.dirname(os.path.abspath(__file__))

if page == "舆情分析可视化大屏":
    render_dashboard(base_dir=base_dir)
elif page == "情感分析模型评估报告":
    render_model_report(base_dir=base_dir)
elif page == "抽样样本标注":
    render_labeling(base_dir=base_dir)
else:
    render_data_collection(base_dir=base_dir)
