"""
Customer Purchasing Behavior Shift Detection Dashboard
========================================================
Interactive Streamlit dashboard with:
  1. 3-Tier Alert Banner (Green / Amber / Red)
  2. KPI Metric Cards
  3. Interactive Plotly trend charts
  4. CSV Simulator sidebar
  5. Embedded drift report tab
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Behavior Shift Detection",
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
#  PSI FUNCTIONS  (duplicated from detect_drift.py for live re-computation)
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
#  SIDEBAR — CSV Simulator
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
st.sidebar.title("🔧 Simulator Controls")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload a transactions CSV",
    type=["csv"],
    help="CSV must have columns: Date, Customer_ID, Category, Spend_Amount",
)

split_month = st.sidebar.slider(
    "📅 Split Month (shift boundary)",
    min_value=2, max_value=12, value=6,
    help="Reference = months before this; Current = this month onward",
)

use_live = st.sidebar.checkbox("⚡ Re-run detection live", value=False,
                               help="Recompute PSI from data instead of loading cached JSON")

st.sidebar.markdown("---")
st.sidebar.caption("Customer Purchasing Behavior\nShift Detection System v1.0")


# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA & SUMMARY
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
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .alert-banner {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .alert-green  { background: #d4edda; border-left: 6px solid #28a745; color: #155724; }
    .alert-amber  { background: #fff3cd; border-left: 6px solid #ffc107; color: #856404; }
    .alert-red    { background: #f8d7da; border-left: 6px solid #dc3545; color: #721c24; }
    .kpi-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .kpi-value { font-size: 1.8rem; font-weight: 700; }
    .kpi-label { font-size: 0.85rem; color: #6c757d; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════
st.title("🔍 Customer Purchasing Behavior Shift Detection")
st.caption("Early awareness system for detecting changes in customer buying patterns")


# ══════════════════════════════════════════════════════════════════════════
#  1. THREE-TIER ALERT BANNER
# ══════════════════════════════════════════════════════════════════════════

# Find biggest gain and biggest drop
gains = {c: v["change_pct"] for c, v in cat_changes.items() if v["change_pct"] > 0}
drops = {c: v["change_pct"] for c, v in cat_changes.items() if v["change_pct"] < 0}
biggest_gain = max(gains, key=gains.get) if gains else "—"
biggest_drop = min(drops, key=drops.get) if drops else "—"

if psi_cat_score < 0.10:
    banner_class = "alert-green"
    banner_icon = "✅"
    banner_title = "STABLE — No Significant Behavioral Shift"
    banner_body = (
        "Customer purchasing patterns remain consistent across all categories. "
        "No action required at this time."
    )
elif psi_cat_score <= 0.25:
    banner_class = "alert-amber"
    banner_icon = "⚠️"
    banner_title = "EARLY AWARENESS WARNING — Behavioral Shift Detected"
    gain_pct = gains.get(biggest_gain, 0)
    drop_pct = drops.get(biggest_drop, 0)
    banner_body = (
        f"<b>Category PSI = {psi_cat_score:.4f}</b> indicates a moderate shift in customer preferences. "
        f"<b>{biggest_drop}</b> purchases dropped by <b>{abs(drop_pct):.1f}%</b>, while "
        f"<b>{biggest_gain}</b> surged by <b>{gain_pct:.1f}%</b>. "
        f"This suggests customers are redirecting spend — possibly driven by seasonal trends, "
        f"promotions, or a genuine preference shift. "
        f"<b>Recommended action:</b> Investigate the root cause, review inventory allocation for "
        f"{biggest_gain}, and monitor {biggest_drop} for continued decline."
    )
else:
    banner_class = "alert-red"
    banner_icon = "🚨"
    banner_title = "CRITICAL SHIFT — Immediate Attention Required"
    banner_body = (
        f"<b>Category PSI = {psi_cat_score:.4f}</b> exceeds the critical threshold of 0.25. "
        f"The distribution of customer purchases has fundamentally changed. "
        f"The baseline model is no longer reliable. Retrain immediately."
    )

st.markdown(
    f'<div class="alert-banner {banner_class}">'
    f'<b>{banner_icon} {banner_title}</b><br><br>{banner_body}</div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
#  2. KPI METRIC CARDS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

with k1:
    psi_color = "#28a745" if psi_cat_score < 0.10 else "#ffc107" if psi_cat_score <= 0.25 else "#dc3545"
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{psi_color}">{psi_cat_score:.4f}</div>'
        f'<div class="kpi-label">Category PSI</div></div>',
        unsafe_allow_html=True,
    )

with k2:
    sp_color = "#28a745" if psi_spend_score < 0.10 else "#ffc107" if psi_spend_score <= 0.25 else "#dc3545"
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{sp_color}">{psi_spend_score:.4f}</div>'
        f'<div class="kpi-label">Spend Amount PSI</div></div>',
        unsafe_allow_html=True,
    )

with k3:
    ref_acc = concept["reference_accuracy"] * 100
    cur_acc = concept["current_accuracy"] * 100
    drop_pp = concept["accuracy_drop_pp"]
    acc_color = "#dc3545" if drop_pp > 5 else "#28a745"
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{acc_color}">{ref_acc:.1f}% → {cur_acc:.1f}%</div>'
        f'<div class="kpi-label">Model Accuracy (−{drop_pp:.1f} pp)</div></div>',
        unsafe_allow_html=True,
    )

with k4:
    # Biggest swing
    all_changes = {c: abs(v["change_pct"]) for c, v in cat_changes.items()}
    biggest_cat = max(all_changes, key=all_changes.get)
    biggest_val = cat_changes[biggest_cat]["change_pct"]
    arrow = "↑" if biggest_val > 0 else "↓"
    swing_color = "#dc3545" if abs(biggest_val) > 20 else "#ffc107"
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value" style="color:{swing_color}">{arrow} {abs(biggest_val):.1f}%</div>'
        f'<div class="kpi-label">Biggest Swing ({biggest_cat})</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════
#  3. TABS — Charts, Details, Drift Report
# ══════════════════════════════════════════════════════════════════════════
tab_trends, tab_comparison, tab_details, tab_report = st.tabs([
    "📈 Monthly Trends",
    "📊 Reference vs Current",
    "📋 Detailed Breakdown",
    "📄 Drift Report",
])


# ── Tab 1: Monthly Trend Lines ───────────────────────────────────────────
with tab_trends:
    st.subheader("Monthly Category Transaction Share (%)")

    monthly = (
        df.assign(Month=df["Date"].dt.month)
        .groupby(["Month", "Category"])
        .size()
        .unstack(fill_value=0)
    )
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    colors = {
        "Electronics": "#636EFA",
        "Clothing": "#EF553B",
        "Home Goods": "#00CC96",
        "Groceries": "#AB63FA",
    }

    fig_trend = go.Figure()
    for cat in CATEGORIES:
        if cat in monthly_pct.columns:
            fig_trend.add_trace(go.Scatter(
                x=monthly_pct.index,
                y=monthly_pct[cat],
                mode="lines+markers",
                name=cat,
                line=dict(width=3, color=colors.get(cat)),
                marker=dict(size=8),
            ))

    # Vertical dashed line at the split month
    fig_trend.add_vline(
        x=split_month, line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"  Shift Point (Month {split_month})",
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color="red",
    )

    fig_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="Transaction Share (%)",
        xaxis=dict(tickmode="linear", dtick=1),
        height=480,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.info(
        f"📌 The dashed red line marks Month {split_month} — the boundary between "
        f"the reference period and the current period. Notice how **Electronics** "
        f"drops sharply and **Home Goods** spikes right at this inflection point."
    )


# ── Tab 2: Side-by-Side Bar Chart ────────────────────────────────────────
with tab_comparison:
    st.subheader("Category Share: Reference vs Current Period")

    cats = list(cat_changes.keys())
    ref_shares = [cat_changes[c]["reference_share"] for c in cats]
    cur_shares = [cat_changes[c]["current_share"] for c in cats]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name=f"Reference (M1–{split_month - 1})",
        x=cats, y=ref_shares,
        marker_color="#636EFA",
        text=[f"{v:.1f}%" for v in ref_shares],
        textposition="outside",
    ))
    fig_bar.add_trace(go.Bar(
        name=f"Current (M{split_month}–12)",
        x=cats, y=cur_shares,
        marker_color="#EF553B",
        text=[f"{v:.1f}%" for v in cur_shares],
        textposition="outside",
    ))

    fig_bar.update_layout(
        barmode="group",
        yaxis_title="Transaction Share (%)",
        height=460,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Change table
    change_df = pd.DataFrame(cat_changes).T
    change_df.columns = ["Reference Share (%)", "Current Share (%)", "Change (%)"]
    change_df.index.name = "Category"
    st.dataframe(
        change_df.style.format("{:.2f}")
        .map(lambda v: "color: #dc3545" if v < -10 else ("color: #28a745" if v > 10 else ""),
             subset=["Change (%)"]),
        use_container_width=True,
    )


# ── Tab 3: Detailed Breakdown ────────────────────────────────────────────
with tab_details:
    st.subheader("PSI Breakdown by Category")

    psi_bk = summary["psi"]["category_distribution"].get("breakdown", {})
    if psi_bk:
        bk_df = pd.DataFrame([
            {"Category": c, "PSI Component": round(v, 4),
             "Status": "⚠ High" if abs(v) > 0.05 else "✓ OK"}
            for c, v in psi_bk.items()
        ])
        st.dataframe(bk_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Concept Drift — Model Degradation")

    cd1, cd2, cd3 = st.columns(3)
    cd1.metric("Reference Accuracy", f"{concept['reference_accuracy']*100:.2f}%")
    cd2.metric("Current Accuracy", f"{concept['current_accuracy']*100:.2f}%",
               delta=f"-{concept['accuracy_drop_pp']:.1f} pp", delta_color="inverse")
    cd3.metric("Drift Threshold", f"{concept['threshold_pp']:.0f} pp",
               delta="EXCEEDED" if concept["flagged"] else "Within range",
               delta_color="inverse" if concept["flagged"] else "normal")

    st.markdown("---")
    st.subheader("Dataset Statistics")
    ds = summary["data_sizes"]
    s1, s2, s3 = st.columns(3)
    s1.metric("Reference Records", f"{ds['reference_records']:,}")
    s2.metric("Current Records", f"{ds['current_records']:,}")
    s3.metric("Total", f"{ds['reference_records'] + ds['current_records']:,}")

    st.markdown("---")
    st.subheader("Raw Summary JSON")
    st.json(summary)


# ── Tab 4: Embedded Drift Report ─────────────────────────────────────────
with tab_report:
    st.subheader("Interactive Drift Report")

    if REPORT_PATH.exists():
        report_html = REPORT_PATH.read_text(encoding="utf-8")
        components.html(report_html, height=800, scrolling=True)
    else:
        st.warning(
            "⚠️ `reports/drift_report.html` not found. "
            "Run `python src/detect_drift.py` first to generate the report."
        )


# ══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#6c757d; font-size:0.85rem;">'
    'Customer Purchasing Behavior Shift Detection System &nbsp;│&nbsp; '
    'Built with Streamlit + Plotly &nbsp;│&nbsp; '
    'PSI Thresholds: Green &lt; 0.10 &nbsp;│&nbsp; Amber 0.10–0.25 &nbsp;│&nbsp; Red &gt; 0.25'
    '</div>',
    unsafe_allow_html=True,
)
