import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlatformConfig:
    name: str
    id_col: str
    prediction_path: str
    bertopic_dir: str
    questions_path: Optional[str] = None


TOPIC_COLUMNS = [
    "platform",
    "aspect",
    "topic",
    "topic_keywords",
    "total_count",
    "negative_count",
    "negative_ratio",
    "weighted_negative",
    "avg_negative_intensity",
    "priority_score",
    "top_demand_1",
    "top_demand_1_count",
    "top_demand_2",
    "top_demand_2_count",
    "top_demand_3",
    "top_demand_3_count",
    "evidence_1",
    "evidence_2",
    "evidence_3",
]

DEMAND_COLUMNS = [
    "platform",
    "demand",
    "negative_count",
    "negative_share",
    "weighted_negative",
    "priority_score",
    "evidence_1",
    "evidence_2",
    "evidence_3",
    "suggested_actions",
]

TREND_COLUMNS = ["platform", "day", "demand", "neg_count", "weighted_neg"]


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path)


def load_topics_keywords(topics_json_path: str, top_k: int = 8) -> Dict[int, str]:
    if not os.path.exists(topics_json_path):
        return {}
    with open(topics_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f) or {}
    out: Dict[int, str] = {}
    for k, v in obj.items():
        try:
            topic_id = int(k)
        except Exception:
            continue
        if not isinstance(v, list):
            continue
        words = []
        for item in v:
            if isinstance(item, dict) and item.get("word"):
                words.append(str(item["word"]))
        out[topic_id] = "、".join(words[: int(top_k)])
    return out


def standardize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "author_name" in df.columns and "user_name" not in df.columns:
        df = df.rename(columns={"author_name": "user_name"})
    if "content" in df.columns and "cleaned_text" not in df.columns:
        df = df.rename(columns={"content": "cleaned_text"})
    return df


def ensure_publish_time(df: pd.DataFrame, cfg: PlatformConfig) -> pd.DataFrame:
    df = df.copy()

    time_col = None
    if "publish_time" in df.columns:
        time_col = "publish_time"
    elif "created_time" in df.columns:
        time_col = "created_time"

    if time_col:
        dt = pd.to_datetime(df[time_col], errors="coerce")
        df["publish_time"] = dt

    if "publish_time" in df.columns and df["publish_time"].notna().any():
        df = df.dropna(subset=["publish_time"])
        return df

    if cfg.questions_path and os.path.exists(cfg.questions_path) and "question_id" in df.columns:
        q = read_csv(cfg.questions_path)
        if "publish_time" in q.columns and "question_id" in q.columns:
            q = q[["question_id", "publish_time"]].copy()
            q = q.rename(columns={"publish_time": "question_publish_time"})
            q["question_publish_time"] = pd.to_datetime(q["question_publish_time"], errors="coerce")
            df = df.merge(q, on="question_id", how="left")
            df["publish_time"] = pd.to_datetime(df.get("publish_time"), errors="coerce").combine_first(
                df["question_publish_time"]
            )
            df = df.drop(columns=["question_publish_time"], errors="ignore")
            df = df.dropna(subset=["publish_time"])
            return df

    return df


def sentiment_negative_mask(labels: pd.Series) -> pd.Series:
    s = labels.astype(str).str.strip()
    return s.isin(["消极", "负面", "negative", "neg", "-1"])


def compute_intensity(df: pd.DataFrame, score_col: str = "sentiment_score") -> pd.Series:
    scores = pd.to_numeric(df.get(score_col), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    neg = sentiment_negative_mask(df.get("sentiment_label", pd.Series(index=df.index, dtype=object)))
    intensity = pd.Series(0.0, index=df.index, dtype=float)
    intensity.loc[neg] = (1.0 - scores.loc[neg]).clip(0.0, 1.0)
    return intensity


def truncate_text(s: str, max_len: int = 140) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def compile_patterns() -> List[Tuple[str, re.Pattern]]:
    rules: List[Tuple[str, str]] = [
        ("退款退货", r"(退款|退钱|退货|退回|赔偿|补偿|赔钱|退我钱)"),
        ("售后客服", r"(售后|客服|人工客服|客服态度|投诉|工单|维权|反馈无人|不回复)"),
        ("质量问题", r"(质量|坏了|故障|出问题|不耐用|漏|裂|异味|生锈|变质|卡顿|闪退|崩溃|掉线)"),
        ("虚假宣传", r"(虚假|夸大|欺骗|割韭菜|宣传|营销|骗|诱导|套路)"),
        ("价格与性价比", r"(太贵|贵|涨价|降价|不值|性价比|价格|收费|会员|续费|扣费)"),
        ("物流与配送", r"(物流|快递|配送|发货|到货|延迟|丢件|破损|签收|运费)"),
        ("食品安全与卫生", r"(卫生|食品安全|吃坏|拉肚子|中毒|异物|虫|发霉|过期)"),
        ("安全与隐私", r"(隐私|泄露|信息泄露|安全问题|盗号|诈骗|风控|账号异常)"),
        ("政策与规则", r"(霸王条款|规则|政策|强制|不合理|封号|禁言|限制|审核|下架)"),
        ("官方回应与态度", r"(道歉|回应|解释|声明|公关|态度|敷衍|装死|不作为)"),
        ("体验与性能", r"(体验|难用|不好用|界面|bug|卡|慢|延迟|发热|耗电|适配)"),
    ]
    return [(name, re.compile(pat)) for name, pat in rules]


PATTERNS = compile_patterns()


def classify_demands(text: str) -> List[str]:
    hits: List[str] = []
    for name, pat in PATTERNS:
        if pat.search(str(text) or ""):
            hits.append(name)
    return hits or ["其他诉求"]


def action_library() -> Dict[str, List[str]]:
    return {
        "退款退货": ["明确退款/退货路径与时限", "对批量投诉设立快速处理通道", "对高影响用户优先回访并给出补偿策略边界"],
        "售后客服": ["优化客服排队与转人工入口", "建立投诉闭环（受理-处理-回访）", "对高频问题输出统一口径与解决脚本"],
        "质量问题": ["按模块归因排查并发布修复计划", "对同批次/同型号问题启动召回或换新策略", "用数据披露减少信息不对称（故障率、修复进度）"],
        "虚假宣传": ["对外澄清关键事实并统一传播口径", "整改宣传物料与落地页，避免夸大表述", "建立合规审核流程与红线清单"],
        "价格与性价比": ["解释定价逻辑与成本构成，避免对立叙事", "推出限时补偿/老用户保价/阶梯优惠", "对核心套餐做价值重塑（加量/增服务）"],
        "物流与配送": ["按地区/仓配节点定位异常并公布恢复时间", "对延误订单自动触发赔付/补券", "提升物流状态可视化与主动通知"],
        "食品安全与卫生": ["立即排查供应链与门店卫生流程并公开结果", "对疑似事件启动第三方检测与公示", "建立高风险批次追溯与快速下架机制"],
        "安全与隐私": ["发布安全事件说明与补救措施（改密/冻结/风控）", "补齐最小权限与数据脱敏策略", "上线安全公告与漏洞响应流程"],
        "政策与规则": ["将规则写清楚并给出申诉通道", "对误伤用户提供快速恢复机制", "关键规则变更提前公告并留缓冲期"],
        "官方回应与态度": ["第一时间给出事实+行动+时间表三要素回应", "减少对抗性表达，聚焦解决方案", "对高热度议题用短频快更新降低猜测空间"],
        "体验与性能": ["建立问题复现清单与版本修复节奏", "对核心路径做可用性测试并迭代", "对高频痛点上线灰度与回滚机制"],
        "其他诉求": ["补充收集问题样本并进行二次归类", "建立 FAQ 与自助解决入口", "对未覆盖诉求扩充动作库"],
    }


ACTIONS = action_library()


def pick_evidence_rows(df_neg: pd.DataFrame, id_col: str, k: int = 3) -> List[str]:
    if df_neg.empty:
        return []
    score = df_neg["weight"].astype(float) * df_neg["intensity"].astype(float)
    top = df_neg.assign(_score=score).sort_values("_score", ascending=False).head(int(k))
    evidences: List[str] = []
    for _, r in top.iterrows():
        rid = r.get(id_col)
        txt = truncate_text(r.get("cleaned_text", ""))
        evidences.append(f"{rid}: {txt}")
    return evidences


def aggregate_topic_insights(
    df: pd.DataFrame,
    cfg: PlatformConfig,
    topic_keywords: Dict[int, str],
    compute_weight_fn: Callable[[pd.DataFrame], pd.Series],
    top_n: int = 30,
) -> pd.DataFrame:
    df = df.copy()
    df["is_negative"] = sentiment_negative_mask(df.get("sentiment_label", pd.Series(index=df.index, dtype=object)))
    df["intensity"] = compute_intensity(df)
    df["weight"] = compute_weight_fn(df)

    if "topic" not in df.columns:
        df["topic"] = -1
    if "aspect" not in df.columns:
        df["aspect"] = "其他"

    rows = []
    for (aspect, topic), g in df.groupby(["aspect", "topic"], dropna=False):
        total = int(len(g))
        neg_g = g[g["is_negative"]].copy()
        neg_cnt = int(len(neg_g))
        if total == 0:
            continue
        neg_ratio = float(neg_cnt / total)
        weighted_neg = float((neg_g["weight"] * (0.5 + 0.5 * neg_g["intensity"])).sum()) if neg_cnt else 0.0
        avg_intensity = float(neg_g["intensity"].mean()) if neg_cnt else 0.0
        priority = float(
            np.log1p(total) * (0.6 * neg_ratio + 0.4 * avg_intensity) * (1.0 + np.log1p(weighted_neg))
        )

        demand_counter: Dict[str, int] = {}
        if neg_cnt:
            for t in neg_g["cleaned_text"].astype(str).tolist():
                for d in classify_demands(t):
                    demand_counter[d] = demand_counter.get(d, 0) + 1
        top_demands = sorted(demand_counter.items(), key=lambda x: (-x[1], x[0]))[:3]
        evidences = pick_evidence_rows(neg_g, cfg.id_col, k=3)

        try:
            topic_int = int(topic)
        except Exception:
            topic_int = -1

        rows.append(
            {
                "platform": cfg.name,
                "aspect": str(aspect) if pd.notna(aspect) else "其他",
                "topic": topic_int,
                "topic_keywords": topic_keywords.get(topic_int, ""),
                "total_count": total,
                "negative_count": neg_cnt,
                "negative_ratio": round(neg_ratio, 4),
                "weighted_negative": round(weighted_neg, 4),
                "avg_negative_intensity": round(avg_intensity, 4),
                "priority_score": round(priority, 6),
                "top_demand_1": top_demands[0][0] if len(top_demands) > 0 else "",
                "top_demand_1_count": top_demands[0][1] if len(top_demands) > 0 else 0,
                "top_demand_2": top_demands[1][0] if len(top_demands) > 1 else "",
                "top_demand_2_count": top_demands[1][1] if len(top_demands) > 1 else 0,
                "top_demand_3": top_demands[2][0] if len(top_demands) > 2 else "",
                "top_demand_3_count": top_demands[2][1] if len(top_demands) > 2 else 0,
                "evidence_1": evidences[0] if len(evidences) > 0 else "",
                "evidence_2": evidences[1] if len(evidences) > 1 else "",
                "evidence_3": evidences[2] if len(evidences) > 2 else "",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values(["priority_score", "negative_count", "total_count"], ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def aggregate_demand_summary(
    df: pd.DataFrame,
    cfg: PlatformConfig,
    compute_weight_fn: Callable[[pd.DataFrame], pd.Series],
    top_n: int = 20,
) -> pd.DataFrame:
    df = df.copy()
    df["is_negative"] = sentiment_negative_mask(df.get("sentiment_label", pd.Series(index=df.index, dtype=object)))
    df = df[df["is_negative"]].copy()
    if df.empty:
        return pd.DataFrame()

    df["intensity"] = compute_intensity(df)
    df["weight"] = compute_weight_fn(df)

    demand_rows = []
    for _, r in df.iterrows():
        ds = classify_demands(r.get("cleaned_text", ""))
        for d in ds:
            demand_rows.append(
                {
                    "demand": d,
                    "id": r.get(cfg.id_col),
                    "text": r.get("cleaned_text", ""),
                    "weight": float(r.get("weight", 1.0)),
                    "intensity": float(r.get("intensity", 0.0)),
                    "publish_time": r.get("publish_time"),
                }
            )

    dd = pd.DataFrame(demand_rows)
    if dd.empty:
        return pd.DataFrame()

    total_neg = float(len(df))
    out_rows = []
    for demand, g in dd.groupby("demand"):
        cnt = int(len(g))
        share = float(cnt / total_neg) if total_neg > 0 else 0.0
        evidences = pick_evidence_rows(
            g.rename(columns={"id": cfg.id_col, "text": "cleaned_text"}),
            id_col=cfg.id_col,
            k=3,
        )
        actions = "；".join(ACTIONS.get(demand, ACTIONS["其他诉求"]))
        weighted = float((g["weight"] * (0.5 + 0.5 * g["intensity"])).sum())
        priority = float(np.log1p(cnt) * (1.0 + np.log1p(weighted)))
        out_rows.append(
            {
                "platform": cfg.name,
                "demand": demand,
                "negative_count": cnt,
                "negative_share": round(share, 4),
                "weighted_negative": round(weighted, 4),
                "priority_score": round(priority, 6),
                "evidence_1": evidences[0] if len(evidences) > 0 else "",
                "evidence_2": evidences[1] if len(evidences) > 1 else "",
                "evidence_3": evidences[2] if len(evidences) > 2 else "",
                "suggested_actions": actions,
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    return (
        out.sort_values(["priority_score", "negative_count"], ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def aggregate_demand_trend_daily(df: pd.DataFrame, cfg: PlatformConfig, compute_weight_fn: Callable[[pd.DataFrame], pd.Series], top_demands: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    df["is_negative"] = sentiment_negative_mask(df.get("sentiment_label", pd.Series(index=df.index, dtype=object)))
    df = df[df["is_negative"]].copy()
    if df.empty or "publish_time" not in df.columns:
        return pd.DataFrame()

    df["intensity"] = compute_intensity(df)
    df["weight"] = compute_weight_fn(df)
    df["day"] = pd.to_datetime(df["publish_time"], errors="coerce").dt.date.astype(str)

    rows = []
    top_set = set([str(x) for x in top_demands])
    for _, r in df.iterrows():
        ds = classify_demands(r.get("cleaned_text", ""))
        ds = [d for d in ds if d in top_set] or []
        for d in ds:
            rows.append(
                {
                    "platform": cfg.name,
                    "day": r.get("day"),
                    "demand": d,
                    "neg_count": 1,
                    "weighted_neg": float(r.get("weight", 1.0) * (0.5 + 0.5 * float(r.get("intensity", 0.0)))),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = (
        out.groupby(["platform", "day", "demand"], as_index=False)
        .agg({"neg_count": "sum", "weighted_neg": "sum"})
        .sort_values(["day", "weighted_neg"], ascending=[True, False])
        .reset_index(drop=True)
    )
    out["weighted_neg"] = out["weighted_neg"].round(4)
    return out

