import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import os


def st_pyecharts(chart, height="400px", width="100%"):
    chart_html = chart.render_embed()
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
    h = int(height.replace("px", "")) if isinstance(height, str) else height
    w = width if isinstance(width, int) else None
    components.html(full_html, height=h, width=w, scrolling=False)


@st.cache_data
def load_data(file_path, file_mtime):
    try:
        _ = file_mtime
        df = pd.read_csv(file_path)

        time_col = "publish_time" if "publish_time" in df.columns else ("created_time" if "created_time" in df.columns else None)
        
        if not time_col:
            st.error("Error loading data: Cannot find 'publish_time' or 'created_time'.")
            return pd.DataFrame()
            
        dt = pd.to_datetime(
            df[time_col],
            format="%a %b %d %H:%M:%S %z %Y",
            errors="coerce",
        )
        if dt.notna().sum() < max(1, int(len(df) * 0.8)):
            dt = pd.to_datetime(df[time_col], errors="coerce")

        df["publish_time"] = dt
        df = df.dropna(subset=["publish_time"])
        return df
    except Exception as e:
        _ = e
        st.error("Error loading data.")
        return pd.DataFrame()
