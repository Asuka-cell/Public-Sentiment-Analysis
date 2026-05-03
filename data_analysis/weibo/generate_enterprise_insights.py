import argparse
import os
import sys

import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from data_analysis.insights_common import (
    DEMAND_COLUMNS,
    TOPIC_COLUMNS,
    TREND_COLUMNS,
    PlatformConfig,
    aggregate_demand_summary,
    aggregate_demand_trend_daily,
    aggregate_topic_insights,
    ensure_publish_time,
    load_topics_keywords,
    read_csv,
    safe_mkdir,
    standardize_prediction_df,
)


def compute_weight(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index, dtype=float)


def build_config(base_dir: str) -> PlatformConfig:
    return PlatformConfig(
        name="weibo",
        id_col="weibo_id",
        prediction_path=os.path.join(base_dir, "model_prediction", "weibo", "prediction", "roBERTa_prediction.csv"),
        bertopic_dir=os.path.join(base_dir, "model_prediction", "weibo", "prediction", "bertopic_output"),
    )


def generate(out_dir: str, base_dir: str, top_n_topics: int, top_n_demands: int):
    cfg = build_config(base_dir)
    if not os.path.exists(cfg.prediction_path):
        raise FileNotFoundError(cfg.prediction_path)

    df = read_csv(cfg.prediction_path)
    df = standardize_prediction_df(df)
    df = ensure_publish_time(df, cfg)

    doc_topics_path = os.path.join(cfg.bertopic_dir, "doc_topics.csv")
    if os.path.exists(doc_topics_path):
        dt = read_csv(doc_topics_path)
        if cfg.id_col in dt.columns:
            dt = dt[[cfg.id_col, "topic", "probability", "aspect"]].copy()
            df = df.merge(dt, on=cfg.id_col, how="left")

    topics_json_path = os.path.join(cfg.bertopic_dir, "topics.json")
    topic_keywords = load_topics_keywords(topics_json_path, top_k=8)

    topic_insights = aggregate_topic_insights(
        df,
        cfg=cfg,
        topic_keywords=topic_keywords,
        compute_weight_fn=compute_weight,
        top_n=top_n_topics,
    )
    demand_summary = aggregate_demand_summary(df, cfg=cfg, compute_weight_fn=compute_weight, top_n=top_n_demands)

    top_demands = demand_summary["demand"].head(8).tolist() if not demand_summary.empty else []
    demand_trend = aggregate_demand_trend_daily(
        df,
        cfg=cfg,
        compute_weight_fn=compute_weight,
        top_demands=top_demands,
    )

    safe_mkdir(out_dir)
    topic_out_path = os.path.join(out_dir, "enterprise_negative_topics_weibo.csv")
    demand_out_path = os.path.join(out_dir, "enterprise_negative_demands_weibo.csv")
    trend_out_path = os.path.join(out_dir, "enterprise_negative_demand_trend_daily_weibo.csv")

    if topic_insights.empty:
        topic_insights = pd.DataFrame(columns=TOPIC_COLUMNS)
    if demand_summary.empty:
        demand_summary = pd.DataFrame(columns=DEMAND_COLUMNS)
    if demand_trend.empty:
        demand_trend = pd.DataFrame(columns=TREND_COLUMNS)

    topic_insights.to_csv(topic_out_path, index=False, encoding="utf-8-sig")
    demand_summary.to_csv(demand_out_path, index=False, encoding="utf-8-sig")
    demand_trend.to_csv(trend_out_path, index=False, encoding="utf-8-sig")

    return topic_out_path, demand_out_path, trend_out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=os.getcwd())
    parser.add_argument("--out_dir", default=os.path.join(os.getcwd(), "dataset", "enterprise_insights"))
    parser.add_argument("--top_n_topics", type=int, default=30)
    parser.add_argument("--top_n_demands", type=int, default=20)
    args = parser.parse_args()

    t, d, tr = generate(args.out_dir, args.base_dir, args.top_n_topics, args.top_n_demands)
    print(t)
    print(d)
    print(tr)


if __name__ == "__main__":
    main()
