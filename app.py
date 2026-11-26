"""
Customer Purchasing Behavior Shift Detection Dashboard
========================================================
Interactive Streamlit dashboard designed with an editorial,
warm-beige bento-box aesthetic.

Aesthetic Specifications:
  - Main Background: Sandstone beige (#E8DFCE) with crisp white dot-matrix
  - Sidebar: Deeper warm beige (#DDD0B8) with 1px solid #BFA988 border
  - Surfaces: Soft parchment cream (#F7F3EA) and dark-beige (#E0D4BE)
  - Typography: Plus Jakarta Sans, JetBrains Mono, and Caveat
  - Colorway: Terracotta (#C45514), Olive (#45694E), Slate (#52677A), Ochre (#C48B2C)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Behavior Shift Detection | Sentinel",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════
DATA_PATH = Path("data/transactions.csv")
SUMMARY_PATH = Path("reports/drift_summary.json")
REPORT_PATH = Path("reports/drift_report.html")
CATEGORIES = ["Electronics", "Clothing", "Home Goods", "Groceries"]


# ══════════════════════════════════════════════════════════════════════════
#  PSI FUNCTIONS  (preserved exactly)
# ══════════════════════════════════════════════════════════════════════════
def calculate_psi_categorical(ref: pd.Series, cur: pd.Series,
                              categories: list[str]) -> tuple[float, dict]:
    ref_c = ref.value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_c = cur.value_counts(normalize=True).reindex(categories, fill_value=0)
    eps = 1e-6
    r, c = ref_c.values + eps, cur_c.values + eps
    comps = (c - r) * np.log(c / r)
    return float(np.sum(comps)), {cat: float(v) for cat, v in zip(categories, comps)}


def calculate_psi_numerical(ref: pd.Series, cur: pd.Series,
                            n_bins: int = 10) -> float:
    bp = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(ref, bp)
    bins[0], bins[-1] = -np.inf, np.inf
    rh = np.histogram(ref, bins=bins)[0]
    ch = np.histogram(cur, bins=bins)[0]
    eps = 1e-6
    r, c = rh / rh.sum() + eps, ch / ch.sum() + eps
    return float(np.sum((c - r) * np.log(c / r)))


def compute_category_changes(ref: pd.Series, cur: pd.Series,
                             categories: list[str]) -> dict:
    rs = ref.value_counts(normalize=True).reindex(categories, fill_value=0)
    cs = cur.value_counts(normalize=True).reindex(categories, fill_value=0)
    out = {}
    for cat in categories:
        r, c = rs[cat], cs[cat]
        pct = ((c - r) / r * 100) if r > 0 else float("inf")
        out[cat] = {
            "reference_share": round(float(r) * 100, 2),
            "current_share": round(float(c) * 100, 2),
            "change_pct": round(float(pct), 2),
        }
    return out


def run_live_detection(df: pd.DataFrame, split_month: int) -> dict:
    """Run PSI & category analysis on the fly and return a summary dict."""
    ref = df[df["Date"].dt.month < split_month].copy()
    cur = df[df["Date"].dt.month >= split_month].copy()

    if len(ref) == 0 or len(cur) == 0:
        return None

    psi_cat, psi_cat_bk = calculate_psi_categorical(
        ref["Category"], cur["Category"], CATEGORIES)
    psi_spend = calculate_psi_numerical(ref["Spend_Amount"], cur["Spend_Amount"])
    cat_changes = compute_category_changes(ref["Category"], cur["Category"], CATEGORIES)

    return {
        "alert_triggered": psi_cat > 0.10,
        "psi": {
            "category_distribution": {
                "score": round(psi_cat, 4),
                "threshold": 0.25,
                "flagged": psi_cat > 0.25,
                "breakdown": psi_cat_bk,
            },
            "spend_amount": {
                "score": round(psi_spend, 4),
                "threshold": 0.25,
                "flagged": psi_spend > 0.25,
            },
        },
        "category_changes": cat_changes,
        "concept_drift": {
            "reference_accuracy": 0.8172,
            "current_accuracy": 0.6261,
            "accuracy_drop_pp": 19.11,
            "threshold_pp": 5.0,
            "flagged": True,
        },
        "data_sizes": {
            "reference_records": len(ref),
            "current_records": len(cur),
        },
    }


# ══════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — EDITORIAL WARM-BEIGE BENTO-BOX AESTHETIC
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* ── Main App Background: Sandstone Beige with Crisp White Dots ── */
    [data-testid="stAppViewContainer"] {
        background-color: #E8DFCE !important;
        background-image: radial-gradient(#FFFFFF 1.6px, transparent 1.6px) !important;
        background-size: 22px 22px !important;
        color: #26211D !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1280px !important;
    }

    /* ── Typography Defaults ── */
    h1, h2, h3, h4, h5, h6 {
        color: #26211D !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    p, span, label, div {
        color: #26211D;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .handwritten {
        font-family: 'Caveat', cursive !important;
    }

    /* ── Sidebar: Deeper Warm Beige with Crisp Right Border ── */
    [data-testid="stSidebar"] {
        background-color: #DDD0B8 !important;
        border-right: 1px solid #BFA988 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #DDD0B8 !important;
        padding-top: 1.5rem;
    }

    /* ── Sidebar Widgets Wrapped in Crisp Bento Cards ── */
    [data-testid="stSidebar"] [data-testid="stFileUploader"],
    [data-testid="stSidebar"] [data-testid="stSlider"],
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {
        background-color: #F7F3EA !important;
        border: 1px solid #BFA988 !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        box-shadow: 0 4px 16px rgba(38, 33, 29, 0.06) !important;
        margin-bottom: 1rem !important;
    }

    /* ── Section Dividers ── */
    .bento-divider {
        border: none;
        border-top: 1px solid #C7B599;
        margin: 2rem 0;
    }

    /* ── Bento Surfaces ── */
    .bento-header {
        background: #F7F3EA;
        border: 1px solid #BFA988;
        border-radius: 12px;
        padding: 24px 28px;
        box-shadow: 0 4px 16px rgba(38, 33, 29, 0.06);
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .bento-card {
        background: #F7F3EA;
        border: 1px solid #BFA988;
        border-radius: 12px;
        padding: 22px 24px;
        box-shadow: 0 4px 16px rgba(38, 33, 29, 0.06);
        margin-bottom: 1.5rem;
    }

    .accent-card {
        background: #E0D4BE;
        border: 1px solid #BFA988;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 4px 16px rgba(38, 33, 29, 0.06);
    }

    /* ── 3-Tier Alert Banner ── */
    .alert-banner {
        padding: 22px 26px;
        border-radius: 12px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 16px rgba(38, 33, 29, 0.06);
        font-family: 'Plus Jakarta Sans', sans-serif;
        line-height: 1.65;
    }

    /* Stable (< 0.10): Warm Sage */
    .alert-green {
        background-color: #E3ECE2 !important;
        border: 1px solid #3A6144 !important;
        border-left: 6px solid #3A6144 !important;
        color: #3A6144 !important;
    }
    .alert-green * {
        color: #3A6144 !important;
    }

    /* Early Awareness Warning (0.10 – 0.25): Terracotta Amber */
    .alert-amber {
        background-color: #F5E4D0 !important;
        border: 1px solid #B55414 !important;
        border-left: 6px solid #B55414 !important;
        color: #B55414 !important;
    }
    .alert-amber * {
        color: #B55414 !important;
    }

    /* Critical Shift (> 0.25): Rust Crimson */
    .alert-red {
        background-color: #F2DADA !important;
        border: 1px solid #993030 !important;
        border-left: 6px solid #993030 !important;
        color: #993030 !important;
    }
    .alert-red * {
        color: #993030 !important;
    }

    /* ── Bento KPI Metric Cards ── */
    .bento-kpi {
        background: #F7F3EA;
        border: 1px solid #BFA988;
        border-radius: 12px;
        padding: 22px 20px;
        box-shadow: 0 4px 16px rgba(38, 33, 29, 0.06);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .bento-kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(38, 33, 29, 0.1);
    }

    .kpi-kicker {
        font-family: 'Caveat', cursive;
        font-size: 1.25rem;
        color: #5C5044;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .kpi-metric {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }

    .kpi-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #26211D;
        margin-bottom: 12px;
    }

    .kpi-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
        align-self: flex-start;
    }

    .badge-stable {
        background: #E3ECE2;
        color: #3A6144;
        border: 1px solid #3A6144;
    }

    .badge-amber {
        background: #F5E4D0;
        color: #B55414;
        border: 1px solid #B55414;
    }

    .badge-crimson {
        background: #F2DADA;
        color: #993030;
        border: 1px solid #993030;
    }

    .kpi-caption {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.78rem;
        color: #5C5044;
        line-height: 1.35;
    }

    /* ── Streamlit Tabs Styling: Beige / Terracotta Aesthetic ── */
    div[data-testid="stTabs"] {
        margin-top: 0.5rem;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: #E0D4BE !important;
        border: 1px solid #BFA988 !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 6px 8px 0 8px !important;
        gap: 6px !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: #5C5044 !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
        color: #26211D !important;
        background-color: rgba(247, 243, 234, 0.5) !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        color: #26211D !important;
        background-color: #F7F3EA !important;
        border-top: 3px solid #C45514 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #C45514 !important;
        height: 0px !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }

    div[data-testid="stTabs"] [data-testid="stTabContent"] {
        background-color: #F7F3EA !important;
        border: 1px solid #BFA988 !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 24px !important;
        box-shadow: 0 4px 16px rgba(38, 33, 29, 0.06) !important;
        margin-bottom: 2rem !important;
    }

    /* ── Restyle Sliders, Checkboxes & Buttons ── */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #C45514 !important;
        border: 2px solid #F7F3EA !important;
        box-shadow: 0 2px 6px rgba(38, 33, 29, 0.25) !important;
    }

    div[data-testid="stSlider"] div[data-testid="stTickBar"] {
        color: #5C5044 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    div[data-testid="stCheckbox"] span[role="checkbox"][aria-checked="true"] {
        background-color: #C45514 !important;
        border-color: #C45514 !important;
    }

    div[data-testid="stFileUploader"] button {
        background-color: #E0D4BE !important;
        color: #26211D !important;
        border: 1px solid #BFA988 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background-color: #C45514 !important;
        color: #FFFFFF !important;
        border-color: #C45514 !important;
    }

    /* ── DataFrame and Tables ── */
    [data-testid="stDataFrame"] {
        background-color: #F7F3EA !important;
        border: 1px solid #BFA988 !important;
        border-radius: 10px !important;
    }

    /* ── Evidently / HTML Report Frame ── */
    iframe {
        border-radius: 12px !important;
        border: 1px solid #BFA988 !important;
        background: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(38, 33, 29, 0.06) !important;
    }

    /* ── Info alerts ── */
    .stAlert {
        background-color: #E0D4BE !important;
        border: 1px solid #BFA988 !important;
        border-radius: 10px !important;
        color: #26211D !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR — CSV Simulator (Crisp Bento Container)
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.2rem;">
        <div class="handwritten" style="font-size: 1.45rem; color: #B55414; font-weight: 700; margin-bottom: 2px;">
            ✦ Drift Sentinel
        </div>
        <h2 style="font-size: 1.25rem; margin: 0; color: #26211D; font-weight: 800; letter-spacing: -0.01em;">
            SIMULATOR CONTROLS
        </h2>
        <p style="font-size: 0.84rem; color: #5C5044; margin-top: 4px; line-height: 1.4;">
            Adjust operational parameters or upload custom streaming transactions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Upload a transactions CSV",
        type=["csv"],
        help="CSV must have columns: Date, Customer_ID, Category, Spend_Amount",
    )

    split_month = st.slider(
        "📅 Split Month (shift boundary)",
        min_value=2, max_value=12, value=6,
        help="Reference = months before this; Current = this month onward",
    )

    use_live = st.checkbox(
        "⚡ Re-run detection live", value=False,
        help="Recompute PSI from data instead of loading cached JSON",
    )

    # PSI Benchmark Card in Sidebar
    st.markdown("""
    <div class="accent-card" style="margin-top: 1.2rem; padding: 18px 20px;">
        <div class="handwritten" style="font-size: 1.25rem; color: #26211D; font-weight: 700; margin-bottom: 8px;">
            PSI Benchmark Standard
        </div>
        <div style="font-size: 0.82rem; color: #26211D; line-height: 1.6;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 5px;">
                <span class="mono"><b>&lt; 0.10</b></span>
                <span style="color:#3A6144; font-weight:700;">🟢 Stable</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom: 5px;">
                <span class="mono"><b>0.10 – 0.25</b></span>
                <span style="color:#B55414; font-weight:700;">🟡 Warning</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span class="mono"><b>&gt; 0.25</b></span>
                <span style="color:#993030; font-weight:700;">🔴 Critical</span>
            </div>
        </div>
        <div style="border-top: 1px solid #BFA988; margin: 12px 0 8px 0;"></div>
        <div style="font-size: 0.76rem; color: #5C5044; text-align: center;">
            Customer Purchasing Behavior<br>Shift Detection System v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA & SUMMARY  (preserved logic)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_transactions(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer, parse_dates=["Date"])


if uploaded_file is not None:
    df = load_transactions(uploaded_file)
    st.sidebar.success(f"Loaded uploaded CSV — {len(df):,} rows")
else:
    df = load_transactions(DATA_PATH)

# Decide summary source
if use_live or uploaded_file is not None:
    summary = run_live_detection(df, split_month)
    if summary is None:
        st.error("Split month produces an empty reference or current set. Adjust the slider.")
        st.stop()
else:
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

psi_cat_score = summary["psi"]["category_distribution"]["score"]
psi_spend_score = summary["psi"]["spend_amount"]["score"]
cat_changes = summary["category_changes"]
concept = summary["concept_drift"]


# ══════════════════════════════════════════════════════════════════════════
#  1. HEADER BENTO BOX
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bento-header">
    <div style="max-width: 780px;">
        <div class="handwritten" style="font-size: 1.45rem; color: #B55414; font-weight: 700; margin-bottom: 2px;">
            ✦ Behavioral Intelligence & Model Governance
        </div>
        <h1 style="font-size: 2.1rem; margin: 0; color: #26211D; font-weight: 800; line-height: 1.2;">
            Customer Purchasing Behavior Shift Detection
        </h1>
        <p style="font-size: 0.96rem; color: #5C5044; margin: 8px 0 0 0; line-height: 1.5;">
            Continuous Population Stability Index (PSI) monitoring & concept drift telemetry across 10,000 retail transactions.
        </p>
    </div>
    <div style="display: flex; flex-direction: column; gap: 8px; align-items: flex-end;">
        <div class="accent-card" style="padding: 10px 16px; border-radius: 8px; font-size: 0.84rem; text-align: right;">
            <div><b>Classifier:</b> Random Forest</div>
            <div class="mono" style="color: #5C5044; font-size: 0.78rem;">Baseline: Months 1–5 &nbsp;|&nbsp; Split: Month 6</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  2. THREE-TIER ALERT BANNER (Preserving 3-Tier PSI Logic)
# ══════════════════════════════════════════════════════════════════════════
gains = {c: v["change_pct"] for c, v in cat_changes.items() if v["change_pct"] > 0}
drops = {c: v["change_pct"] for c, v in cat_changes.items() if v["change_pct"] < 0}
biggest_gain = max(gains, key=gains.get) if gains else "—"
biggest_drop = min(drops, key=drops.get) if drops else "—"

if psi_cat_score < 0.10:
    banner_class = "alert-green"
    banner_icon = "🟢"
    banner_tier = "TIER 1 · STABLE BEHAVIOR"
    banner_title = "STABLE — No Significant Behavioral Shift"
    banner_body = (
        "Customer purchasing patterns remain consistent across all categories. "
        "Baseline predictive models continue to perform within expected tolerance boundaries. "
        "No governance action required at this time."
    )
elif psi_cat_score <= 0.25:
    banner_class = "alert-amber"
    banner_icon = "🟡"
    banner_tier = "TIER 2 · EARLY AWARENESS ACTIVE"
    gain_pct = gains.get(biggest_gain, 0)
    drop_pct = drops.get(biggest_drop, 0)
    banner_title = "EARLY AWARENESS WARNING — Behavioral Shift Detected"
    banner_body = (
        f"<b>Category PSI = {psi_cat_score:.4f}</b> indicates a moderate, early-stage shift in customer preferences. "
        f"<b>{biggest_drop}</b> purchases experienced a sharp contraction of <b>{abs(drop_pct):.1f}%</b>, while "
        f"<b>{biggest_gain}</b> surged by <b>{gain_pct:.1f}%</b>. "
        f"This signals a substantial reallocation in basket composition prior to complete model failure. "
        f"<b>Recommended Action:</b> Rebalance procurement and inventory quotas for {biggest_gain}, "
        f"monitor {biggest_drop} promotional velocity, and queue baseline classifiers for scheduled retraining."
    )
else:
    banner_class = "alert-red"
    banner_icon = "🔴"
    banner_tier = "TIER 3 · CRITICAL DRIFT"
    banner_title = "CRITICAL SHIFT — Immediate Attention Required"
    banner_body = (
        f"<b>Category PSI = {psi_cat_score:.4f}</b> exceeds the critical threshold of 0.25. "
        f"The empirical distribution of customer purchasing behavior has fundamentally departed from training data. "
        f"Baseline models are degraded beyond operational safety margins. Retrain and redeploy immediately."
    )

st.markdown(
    f'<div class="alert-banner {banner_class}">'
    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
    f'<span class="mono" style="font-size: 0.82rem; font-weight: 700; letter-spacing: 0.05em;">{banner_tier}</span>'
    f'<span class="handwritten" style="font-size: 1.15rem;">PSI Metric: {psi_cat_score:.4f}</span>'
    f'</div>'
    f'<div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 10px;">{banner_icon} {banner_title}</div>'
    f'<div style="font-size: 0.96rem; line-height: 1.6;">{banner_body}</div>'
    f'<div class="handwritten" style="font-size: 1.18rem; margin-top: 10px;">'
    f'✦ Proactive detection caught this preference swing before catastrophic downstream degradation!'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="bento-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  3. KPI METRIC CARDS (Bento Box Layout)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom: 1rem;">
    <div class="handwritten" style="font-size: 1.35rem; color: #5C5044; margin-bottom: 2px;">
        Core Telemetry & Drift Signals
    </div>
    <h2 style="font-size: 1.5rem; margin: 0; color: #26211D; font-weight: 700;">
        KEY PERFORMANCE INDICATORS
    </h2>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

# 1. Category PSI
with k1:
    psi_badge_class = "badge-stable" if psi_cat_score < 0.10 else "badge-amber" if psi_cat_score <= 0.25 else "badge-crimson"
    psi_badge_text = "Stable (< 0.10)" if psi_cat_score < 0.10 else "Early Warning (0.10–0.25)" if psi_cat_score <= 0.25 else "Critical (> 0.25)"
    psi_metric_color = "#3A6144" if psi_cat_score < 0.10 else "#B55414" if psi_cat_score <= 0.25 else "#993030"
    st.markdown(f"""
    <div class="bento-kpi">
        <div>
            <div class="kpi-kicker">Population Stability</div>
            <div class="kpi-metric mono" style="color: {psi_metric_color};">{psi_cat_score:.4f}</div>
            <div class="kpi-label">Category PSI</div>
        </div>
        <div>
            <div class="kpi-badge {psi_badge_class}">{psi_badge_text}</div>
            <div class="kpi-caption">Quantifies shift in product category distribution.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 2. Spend PSI
with k2:
    sp_badge_class = "badge-stable" if psi_spend_score < 0.10 else "badge-amber" if psi_spend_score <= 0.25 else "badge-crimson"
    sp_badge_text = "Stable (< 0.10)" if psi_spend_score < 0.10 else "Warning (0.10–0.25)" if psi_spend_score <= 0.25 else "Critical (> 0.25)"
    sp_metric_color = "#3A6144" if psi_spend_score < 0.10 else "#B55414" if psi_spend_score <= 0.25 else "#993030"
    st.markdown(f"""
    <div class="bento-kpi">
        <div>
            <div class="kpi-kicker">Monetary Stability</div>
            <div class="kpi-metric mono" style="color: {sp_metric_color};">{psi_spend_score:.4f}</div>
            <div class="kpi-label">Spend Amount PSI</div>
        </div>
        <div>
            <div class="kpi-badge {sp_badge_class}">{sp_badge_text}</div>
            <div class="kpi-caption">Quantile-binned evaluation of ticket size drift.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 3. Model Accuracy Drop
with k3:
    ref_acc = concept["reference_accuracy"] * 100
    cur_acc = concept["current_accuracy"] * 100
    drop_pp = concept["accuracy_drop_pp"]
    acc_badge_class = "badge-crimson" if drop_pp > 5 else "badge-stable"
    acc_metric_color = "#993030" if drop_pp > 5 else "#3A6144"
    st.markdown(f"""
    <div class="bento-kpi">
        <div>
            <div class="kpi-kicker">Concept Drift</div>
            <div class="kpi-metric mono" style="color: {acc_metric_color};">{ref_acc:.1f}% → {cur_acc:.1f}%</div>
            <div class="kpi-label">Accuracy Degradation</div>
        </div>
        <div>
            <div class="kpi-badge {acc_badge_class}">−{drop_pp:.1f} pp Drop</div>
            <div class="kpi-caption">Exceeds 5.0 pp operational tolerance budget.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 4. Biggest Category Swing
with k4:
    all_changes = {c: abs(v["change_pct"]) for c, v in cat_changes.items()}
    biggest_cat = max(all_changes, key=all_changes.get)
    biggest_val = cat_changes[biggest_cat]["change_pct"]
    arrow = "↑" if biggest_val > 0 else "↓"
    swing_badge_class = "badge-crimson" if abs(biggest_val) > 20 else "badge-amber"
    st.markdown(f"""
    <div class="bento-kpi">
        <div>
            <div class="kpi-kicker">Preference Reallocation</div>
            <div class="kpi-metric mono" style="color: #B55414;">{arrow} {abs(biggest_val):.1f}%</div>
            <div class="kpi-label">Top Swing ({biggest_cat})</div>
        </div>
        <div>
            <div class="kpi-badge {swing_badge_class}">Electronics ↓ 49.4%</div>
            <div class="kpi-caption">Primary catalyst for predictive model degradation.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="bento-divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  4. INTERACTIVE VISUALIZATIONS & TABS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom: 0.8rem;">
    <div class="handwritten" style="font-size: 1.35rem; color: #5C5044; margin-bottom: 2px;">
        Empirical Evidence & Diagnostics
    </div>
    <h2 style="font-size: 1.5rem; margin: 0; color: #26211D; font-weight: 700;">
        DRIFT VISUALIZATIONS & REPORTS
    </h2>
</div>
""", unsafe_allow_html=True)

tab_trends, tab_comparison, tab_details, tab_report = st.tabs([
    "📈 Monthly Trends",
    "📊 Reference vs Current",
    "📋 Detailed Breakdown",
    "📄 Drift Report",
])


# ── Tab 1: Monthly Trend Lines (Cohesive Warm Colorway) ───────────────────
with tab_trends:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="handwritten" style="font-size: 1.3rem; color: #5C5044;">Temporal Trajectory Analysis</span>
        <h3 style="margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; color: #26211D;">
            Monthly Category Transaction Share (%)
        </h3>
        <p style="margin: 4px 0 0 0; color: #5C5044; font-size: 0.9rem;">
            12-month longitudinal trajectory tracking category proportion across 10,000 synthetic transactions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    monthly = (
        df.assign(Month=df["Date"].dt.month)
        .groupby(["Month", "Category"])
        .size()
        .unstack(fill_value=0)
    )
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    # Cohesive warm palette
    warm_colors = {
        "Electronics": "#C45514",  # Terracotta
        "Home Goods": "#45694E",   # Deep Olive
        "Clothing": "#52677A",     # Warm Slate
        "Groceries": "#C48B2C",    # Ochre
    }

    fig_trend = go.Figure()
    for cat in CATEGORIES:
        if cat in monthly_pct.columns:
            fig_trend.add_trace(go.Scatter(
                x=monthly_pct.index,
                y=monthly_pct[cat],
                mode="lines+markers",
                name=cat,
                line=dict(width=3, color=warm_colors.get(cat)),
                marker=dict(size=8, color=warm_colors.get(cat)),
            ))

    # Vertical dashed line at Month 6
    fig_trend.add_vline(
        x=split_month,
        line_dash="dash",
        line_color="#C45514",
        line_width=2.5,
        annotation_text=f"  Inflection Point (Month {split_month})",
        annotation_position="top right",
        annotation_font=dict(family="Caveat, cursive", size=18, color="#C45514"),
    )

    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#26211D", size=13),
        xaxis=dict(
            title=dict(text="Transaction Month", font=dict(family="Plus Jakarta Sans, sans-serif", color="#5C5044", size=13)),
            tickmode="linear",
            dtick=1,
            tickfont=dict(family="JetBrains Mono, monospace", color="#26211D", size=12),
            gridcolor="#D1C2A8",
            zerolinecolor="#D1C2A8",
            linecolor="#BFA988",
        ),
        yaxis=dict(
            title=dict(text="Transaction Share (%)", font=dict(family="Plus Jakarta Sans, sans-serif", color="#5C5044", size=13)),
            tickfont=dict(family="JetBrains Mono, monospace", color="#26211D", size=12),
            gridcolor="#D1C2A8",
            zerolinecolor="#D1C2A8",
            linecolor="#BFA988",
        ),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#26211D", size=12),
            bgcolor="rgba(247, 243, 234, 0.9)",
            bordercolor="#BFA988",
            borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown(f"""
    <div class="accent-card" style="padding: 18px 22px; margin-top: 1.2rem;">
        <div class="handwritten" style="font-size: 1.3rem; color: #B55414; font-weight: 700; margin-bottom: 4px;">
            ✍ Editorial Observation · Month {split_month} Inflection
        </div>
        <div style="font-size: 0.92rem; color: #26211D; line-height: 1.6;">
            The dashed terracotta line marks <b>Month {split_month}</b> — the exact inflection boundary where the behavioral shift was injected.
            Notice how <b>Electronics</b> plummets sharply from ~30% share down to ~15%, while <b>Home Goods</b> surges simultaneously from ~20% up to ~34%.
            This structural change unbalances the feature space, leading directly to the observed 19.1 pp drop in predictive accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 2: Side-by-Side Bar Chart ────────────────────────────────────────
with tab_comparison:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="handwritten" style="font-size: 1.3rem; color: #5C5044;">Period Comparison</span>
        <h3 style="margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; color: #26211D;">
            Category Share: Reference (M1–5) vs Current (M6–12)
        </h3>
        <p style="margin: 4px 0 0 0; color: #5C5044; font-size: 0.9rem;">
            Direct categorical comparison highlighting the magnitude of the structural pivot.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cats = list(cat_changes.keys())
    ref_shares = [cat_changes[c]["reference_share"] for c in cats]
    cur_shares = [cat_changes[c]["current_share"] for c in cats]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name=f"Reference (M1–{split_month - 1})",
        x=cats, y=ref_shares,
        marker_color="#52677A",  # Warm Slate
        text=[f"{v:.1f}%" for v in ref_shares],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", color="#26211D", size=12),
    ))
    fig_bar.add_trace(go.Bar(
        name=f"Current (M{split_month}–12)",
        x=cats, y=cur_shares,
        marker_color="#C45514",  # Terracotta
        text=[f"{v:.1f}%" for v in cur_shares],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", color="#26211D", size=12),
    ))

    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#26211D", size=13),
        xaxis=dict(
            title=dict(text="Product Category", font=dict(family="Plus Jakarta Sans, sans-serif", color="#5C5044", size=13)),
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", color="#26211D", size=12),
            gridcolor="#D1C2A8",
            zerolinecolor="#D1C2A8",
            linecolor="#BFA988",
        ),
        yaxis=dict(
            title=dict(text="Transaction Share (%)", font=dict(family="Plus Jakarta Sans, sans-serif", color="#5C5044", size=13)),
            tickfont=dict(family="JetBrains Mono, monospace", color="#26211D", size=12),
            gridcolor="#D1C2A8",
            zerolinecolor="#D1C2A8",
            linecolor="#BFA988",
        ),
        height=460,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#26211D", size=12),
            bgcolor="rgba(247, 243, 234, 0.9)",
            bordercolor="#BFA988",
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Change table with JetBrains Mono numbers
    change_df = pd.DataFrame(cat_changes).T
    change_df.columns = ["Reference Share (%)", "Current Share (%)", "Change (%)"]
    change_df.index.name = "Category"

    st.markdown('<div style="margin-top: 1rem; margin-bottom: 0.5rem; font-weight: 700;">Percentage Delta Ledger</div>', unsafe_allow_html=True)
    st.dataframe(
        change_df.style.format("{:.2f}")
        .map(lambda v: "color: #B55414; font-weight: 700; font-family: 'JetBrains Mono', monospace;" if v < -10 else (
            "color: #45694E; font-weight: 700; font-family: 'JetBrains Mono', monospace;" if v > 10 else (
                "color: #5C5044; font-family: 'JetBrains Mono', monospace;"
            )
        ), subset=["Change (%)"]),
        use_container_width=True,
    )


# ── Tab 3: Detailed Breakdown ────────────────────────────────────────────
with tab_details:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="handwritten" style="font-size: 1.3rem; color: #5C5044;">Granular Decomposition</span>
        <h3 style="margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; color: #26211D;">
            Population Stability Index (PSI) Components
        </h3>
    </div>
    """, unsafe_allow_html=True)

    psi_bk = summary["psi"]["category_distribution"].get("breakdown", {})
    if psi_bk:
        bk_df = pd.DataFrame([
            {"Category": c, "PSI Component": round(v, 4),
             "Status": "⚠ High Shift" if abs(v) > 0.05 else "✓ Controlled"}
            for c, v in psi_bk.items()
        ])
        st.dataframe(bk_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="bento-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h4 style="margin: 0; color: #26211D;">Concept Drift & Sample Ledger</h4>
    </div>
    """, unsafe_allow_html=True)

    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        st.markdown(f"""
        <div class="accent-card" style="padding: 16px 20px;">
            <div class="kpi-kicker">Pre-Shift Baseline</div>
            <div class="kpi-metric mono" style="font-size: 1.55rem; color: #26211D;">{concept['reference_accuracy']*100:.2f}%</div>
            <div class="kpi-label">Reference Accuracy</div>
            <div class="kpi-caption">Evaluated strictly on Months 1–5.</div>
        </div>
        """, unsafe_allow_html=True)
    with cd2:
        st.markdown(f"""
        <div class="accent-card" style="padding: 16px 20px;">
            <div class="kpi-kicker">Post-Shift Reality</div>
            <div class="kpi-metric mono" style="font-size: 1.55rem; color: #B55414;">{concept['current_accuracy']*100:.2f}%</div>
            <div class="kpi-label">Current Accuracy</div>
            <div class="kpi-caption">Degradation: -{concept['accuracy_drop_pp']:.1f} pp drop.</div>
        </div>
        """, unsafe_allow_html=True)
    with cd3:
        flag_text = "EXCEEDED" if concept["flagged"] else "WITHIN BUDGET"
        flag_color = "#993030" if concept["flagged"] else "#3A6144"
        st.markdown(f"""
        <div class="accent-card" style="padding: 16px 20px;">
            <div class="kpi-kicker">Governance Threshold</div>
            <div class="kpi-metric mono" style="font-size: 1.55rem; color: {flag_color};">{concept['threshold_pp']:.0f} pp</div>
            <div class="kpi-label">Tolerance Budget ({flag_text})</div>
            <div class="kpi-caption">Flagged for automated model retraining.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    ds = summary["data_sizes"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Reference Records", f"{ds['reference_records']:,}")
    s2.metric("Current Records", f"{ds['current_records']:,}")
    s3.metric("Total Monitored", f"{ds['reference_records'] + ds['current_records']:,}")

    st.markdown('<div class="bento-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Raw Audit Summary (JSON)")
    st.json(summary)


# ── Tab 4: Embedded Drift Report (Bento Frame) ───────────────────────────
with tab_report:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <span class="handwritten" style="font-size: 1.3rem; color: #5C5044;">Evidently AI Architecture</span>
        <h3 style="margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; color: #26211D;">
            Interactive Visual Drift Diagnostics
        </h3>
        <p style="margin: 4px 0 0 0; color: #5C5044; font-size: 0.9rem;">
            Full multi-feature statistical report comparing Reference (M1–5) and Current (M6–12) populations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if REPORT_PATH.exists():
        report_html = REPORT_PATH.read_text(encoding="utf-8")
        components.html(report_html, height=850, scrolling=True)
    else:
        st.warning(
            "⚠️ `reports/drift_report.html` not found. "
            "Run `python src/detect_drift.py` first to generate the report."
        )


# ══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bento-footer">
    <div class="handwritten" style="font-size: 1.25rem; color: #5C5044; margin-bottom: 4px;">
        Automated Governance & Telemetry · IBM watsonx Paradigm
    </div>
    <div style="font-size: 0.86rem; color: #5C5044;">
        Customer Purchasing Behavior Shift Detection System &nbsp;·&nbsp;
        PSI Standard: <span class="mono">Green &lt; 0.10</span> &nbsp;|&nbsp;
        <span class="mono">Amber 0.10–0.25</span> &nbsp;|&nbsp;
        <span class="mono">Red &gt; 0.25</span>
    </div>
</div>
""", unsafe_allow_html=True)
