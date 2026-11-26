"""
Behavioral Shift Detection System
===================================
Detects data drift (PSI) and concept drift (model degradation)
between pre-shift (months 1-5) and post-shift (months 6-12) data.

Outputs:
  - Terminal:  [EARLY AWARENESS ALERT] summary
  - File:      reports/drift_report.html   (Evidently visual report)
  - File:      reports/drift_summary.json  (machine-readable metrics)
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
DATA_PATH = Path("data/transactions.csv")
MODEL_PATH = Path("src/baseline_model.pkl")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

PSI_THRESHOLD = 0.25          # PSI > 0.25 ⇒ significant shift
ACCURACY_DROP_THRESHOLD = 0.05  # > 5 pp drop ⇒ concept drift

CATEGORIES = ["Electronics", "Clothing", "Home Goods", "Groceries"]


# ══════════════════════════════════════════════════════════════════════════
#  PSI CALCULATION
# ══════════════════════════════════════════════════════════════════════════
def calculate_psi_categorical(reference: pd.Series, current: pd.Series,
                              categories: list[str]) -> tuple[float, dict]:
    """
    Population Stability Index for a categorical feature.
    Returns (psi_score, {category: psi_component}).
    """
    ref_counts = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_counts = current.value_counts(normalize=True).reindex(categories, fill_value=0)

    # Avoid log(0) with small epsilon
    eps = 1e-6
    ref_p = ref_counts.values + eps
    cur_p = cur_counts.values + eps

    psi_components = (cur_p - ref_p) * np.log(cur_p / ref_p)
    psi_total = float(np.sum(psi_components))

    breakdown = {cat: float(comp) for cat, comp in zip(categories, psi_components)}
    return psi_total, breakdown


def calculate_psi_numerical(reference: pd.Series, current: pd.Series,
                            n_bins: int = 10) -> float:
    """
    Population Stability Index for a numerical feature using quantile bins.
    """
    # Build bins from reference distribution
    breakpoints = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(reference, breakpoints)
    bins[0] = -np.inf
    bins[-1] = np.inf

    ref_hist = np.histogram(reference, bins=bins)[0]
    cur_hist = np.histogram(current, bins=bins)[0]

    # Normalise to proportions
    eps = 1e-6
    ref_p = ref_hist / ref_hist.sum() + eps
    cur_p = cur_hist / cur_hist.sum() + eps

    psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    return psi


def compute_category_percentage_changes(reference: pd.Series,
                                        current: pd.Series,
                                        categories: list[str]) -> dict:
    """Exact percentage change in transaction share per category."""
    ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    changes = {}
    for cat in categories:
        r, c = ref_share[cat], cur_share[cat]
        pct = ((c - r) / r * 100) if r > 0 else float("inf")
        changes[cat] = {
            "reference_share": round(float(r) * 100, 2),
            "current_share": round(float(c) * 100, 2),
            "change_pct": round(float(pct), 2),
        }
    return changes


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (mirrors train_baseline.py exactly)
# ══════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the identical feature engineering used during training."""
    out = df.copy()
    out["DayOfWeek"] = out["Date"].dt.day_of_week
    out["Month"] = out["Date"].dt.month
    out["DayOfMonth"] = out["Date"].dt.day
    out["IsWeekend"] = (out["DayOfWeek"] >= 5).astype(int)
    # Frequency-encode Customer_ID within each subset
    cust_freq = out["Customer_ID"].value_counts()
    out["Customer_Frequency"] = out["Customer_ID"].map(cust_freq)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  VISUAL DRIFT REPORT  (standalone HTML with plotly)
# ══════════════════════════════════════════════════════════════════════════
def generate_drift_report(reference_df: pd.DataFrame,
                          current_df: pd.DataFrame,
                          psi_category: float,
                          psi_spend: float,
                          cat_changes: dict,
                          ref_accuracy: float,
                          cur_accuracy: float,
                          output_path: Path) -> bool:
    """Generate a self-contained HTML drift report with plotly charts."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio

        cats = list(cat_changes.keys())
        ref_shares = [cat_changes[c]["reference_share"] for c in cats]
        cur_shares = [cat_changes[c]["current_share"] for c in cats]
        pct_changes = [cat_changes[c]["change_pct"] for c in cats]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Category Distribution: Reference vs Current",
                "Category Share Change (%)",
                "Spend Amount Distribution",
                "Model Accuracy Comparison",
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]],
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        # 1 ─ Category distribution side-by-side
        fig.add_trace(go.Bar(name="Reference (M1-5)", x=cats, y=ref_shares,
                             marker_color="#636EFA"), row=1, col=1)
        fig.add_trace(go.Bar(name="Current (M6-12)", x=cats, y=cur_shares,
                             marker_color="#EF553B"), row=1, col=1)

        # 2 ─ Percentage change per category
        colors = ["#00CC96" if v > 0 else "#EF553B" for v in pct_changes]
        fig.add_trace(go.Bar(name="% Change", x=cats, y=pct_changes,
                             marker_color=colors, showlegend=False),
                      row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

        # 3 ─ Spend distribution overlay
        fig.add_trace(go.Histogram(x=reference_df["Spend_Amount"], name="Ref Spend",
                                   opacity=0.6, marker_color="#636EFA",
                                   nbinsx=50, showlegend=False), row=2, col=1)
        fig.add_trace(go.Histogram(x=current_df["Spend_Amount"], name="Cur Spend",
                                   opacity=0.6, marker_color="#EF553B",
                                   nbinsx=50, showlegend=False), row=2, col=1)
        fig.update_layout(barmode="overlay")

        # 4 ─ Model accuracy comparison
        fig.add_trace(go.Bar(
            x=["Reference\n(M1-5)", "Current\n(M6-12)"],
            y=[ref_accuracy * 100, cur_accuracy * 100],
            marker_color=["#636EFA", "#EF553B"],
            text=[f"{ref_accuracy*100:.1f}%", f"{cur_accuracy*100:.1f}%"],
            textposition="outside",
            showlegend=False,
        ), row=2, col=2)

        fig.update_layout(
            title_text=(
                f"<b>Data Drift Report</b> &nbsp;│&nbsp; "
                f"Category PSI: {psi_category:.4f} &nbsp;│&nbsp; "
                f"Spend PSI: {psi_spend:.4f} &nbsp;│&nbsp; "
                f"Accuracy Drop: {(ref_accuracy - cur_accuracy)*100:.1f} pp"
            ),
            title_x=0.5,
            height=750,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="Share (%)", row=1, col=1)
        fig.update_yaxes(title_text="Change (%)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=2, col=2)

        pio.write_html(fig, str(output_path), include_plotlyjs=True,
                       full_html=True, auto_open=False)
        print(f"  ✓ Drift report saved → {output_path}")
        return True

    except Exception as e:
        print(f"  ✗ Drift report generation failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    # ── Load data ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  BEHAVIORAL SHIFT DETECTION SYSTEM")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"\nTotal transactions loaded: {len(df):,}")

    # ── Split into reference / current ────────────────────────────────────
    reference_data = df[df["Date"].dt.month <= 5].copy()
    current_data = df[df["Date"].dt.month >= 6].copy()
    print(f"Reference data (months 1-5): {len(reference_data):,}")
    print(f"Current   data (months 6-12): {len(current_data):,}")

    # ══════════════════════════════════════════════════════════════════════
    #  1. STATISTICAL DATA DRIFT  (PSI)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  1. STATISTICAL DATA DRIFT  (Population Stability Index)")
    print(f"{'─' * 70}")

    # PSI – Category distribution
    psi_category, psi_cat_breakdown = calculate_psi_categorical(
        reference_data["Category"], current_data["Category"], CATEGORIES
    )
    cat_flagged = psi_category > PSI_THRESHOLD

    print(f"\n  Category PSI:     {psi_category:.4f}  "
          f"{'🚨 SIGNIFICANT SHIFT' if cat_flagged else '✓ Stable'}")
    for cat, comp in psi_cat_breakdown.items():
        print(f"    {cat:<15s}  PSI component: {comp:+.4f}")

    # PSI – Spend_Amount
    psi_spend = calculate_psi_numerical(
        reference_data["Spend_Amount"], current_data["Spend_Amount"]
    )
    spend_flagged = psi_spend > PSI_THRESHOLD

    print(f"\n  Spend_Amount PSI: {psi_spend:.4f}  "
          f"{'🚨 SIGNIFICANT SHIFT' if spend_flagged else '✓ Stable'}")

    # Category percentage changes
    cat_changes = compute_category_percentage_changes(
        reference_data["Category"], current_data["Category"], CATEGORIES
    )
    print("\n  Category Transaction Share Changes:")
    for cat, info in cat_changes.items():
        arrow = "↑" if info["change_pct"] > 0 else "↓"
        print(f"    {cat:<15s}  {info['reference_share']:5.1f}% → "
              f"{info['current_share']:5.1f}%  "
              f"({arrow} {abs(info['change_pct']):.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    #  2. CONCEPT DRIFT  (Model Degradation)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  2. CONCEPT DRIFT  (Model Performance Degradation)")
    print(f"{'─' * 70}")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    le = artifact["label_encoder"]
    feature_cols = artifact["feature_columns"]

    # Engineer features for both periods
    ref_eng = engineer_features(reference_data)
    cur_eng = engineer_features(current_data)

    # Reference accuracy
    X_ref = ref_eng[feature_cols]
    y_ref = le.transform(ref_eng["Category"])
    ref_accuracy = accuracy_score(y_ref, model.predict(X_ref))

    # Current accuracy
    X_cur = cur_eng[feature_cols]
    y_cur = le.transform(cur_eng["Category"])
    cur_accuracy = accuracy_score(y_cur, model.predict(X_cur))

    accuracy_drop = ref_accuracy - cur_accuracy
    concept_drift_flagged = accuracy_drop > ACCURACY_DROP_THRESHOLD

    print(f"\n  Reference accuracy (months 1-5): {ref_accuracy:.4f}  ({ref_accuracy*100:.2f}%)")
    print(f"  Current   accuracy (months 6-12): {cur_accuracy:.4f}  ({cur_accuracy*100:.2f}%)")
    print(f"  Accuracy drop:                    {accuracy_drop:.4f}  ({accuracy_drop*100:.2f} pp)")
    print(f"  Status: {'🚨 CONCEPT DRIFT DETECTED' if concept_drift_flagged else '✓ Within tolerance'}")

    # Detailed classification report on current data
    print(f"\n  Classification Report (current data, months 6-12):")
    cur_pred = model.predict(X_cur)
    print(classification_report(y_cur, cur_pred, target_names=le.classes_, zero_division=0))

    # ══════════════════════════════════════════════════════════════════════
    #  3. EVIDENTLY VISUAL REPORT
    # ══════════════════════════════════════════════════════════════════════
    print(f"{'─' * 70}")
    print("  3. VISUAL DRIFT REPORT")
    print(f"{'─' * 70}\n")

    report_path = REPORTS_DIR / "drift_report.html"
    generate_drift_report(
        ref_eng, cur_eng,
        psi_category, psi_spend, cat_changes,
        ref_accuracy, cur_accuracy,
        report_path,
    )

    # ══════════════════════════════════════════════════════════════════════
    #  4. ALERTING – EARLY AWARENESS ALERT
    # ══════════════════════════════════════════════════════════════════════
    any_alert = cat_flagged or spend_flagged or concept_drift_flagged

    print(f"\n{'=' * 70}")
    if any_alert:
        print("  ⚠️  [EARLY AWARENESS ALERT]  BEHAVIORAL SHIFT DETECTED")
    else:
        print("  ✅  [STATUS] No significant behavioral shift detected")
    print(f"{'=' * 70}")

    print(f"\n  PSI Scores:")
    print(f"    Category distribution:  {psi_category:.4f}  "
          f"(threshold: {PSI_THRESHOLD})  "
          f"{'⚠ FLAGGED' if cat_flagged else '✓ OK'}")
    print(f"    Spend_Amount:           {psi_spend:.4f}  "
          f"(threshold: {PSI_THRESHOLD})  "
          f"{'⚠ FLAGGED' if spend_flagged else '✓ OK'}")

    print(f"\n  Category Swings:")
    for cat, info in cat_changes.items():
        arrow = "↑" if info["change_pct"] > 0 else "↓"
        flag = " ⚠" if abs(info["change_pct"]) > 20 else ""
        print(f"    {cat:<15s}  {arrow} {abs(info['change_pct']):6.1f}%{flag}")

    print(f"\n  Model Accuracy:")
    print(f"    Reference: {ref_accuracy*100:.2f}%  →  Current: {cur_accuracy*100:.2f}%  "
          f"(drop: {accuracy_drop*100:.2f} pp)  "
          f"{'⚠ CONCEPT DRIFT' if concept_drift_flagged else '✓ OK'}")

    # ══════════════════════════════════════════════════════════════════════
    #  5. SAVE SUMMARY METRICS → JSON
    # ══════════════════════════════════════════════════════════════════════
    summary = {
        "alert_triggered": any_alert,
        "psi": {
            "category_distribution": {
                "score": round(psi_category, 4),
                "threshold": PSI_THRESHOLD,
                "flagged": cat_flagged,
                "breakdown": {k: round(v, 4) for k, v in psi_cat_breakdown.items()},
            },
            "spend_amount": {
                "score": round(psi_spend, 4),
                "threshold": PSI_THRESHOLD,
                "flagged": spend_flagged,
            },
        },
        "category_changes": cat_changes,
        "concept_drift": {
            "reference_accuracy": round(ref_accuracy, 4),
            "current_accuracy": round(cur_accuracy, 4),
            "accuracy_drop_pp": round(accuracy_drop * 100, 2),
            "threshold_pp": ACCURACY_DROP_THRESHOLD * 100,
            "flagged": concept_drift_flagged,
        },
        "data_sizes": {
            "reference_records": len(reference_data),
            "current_records": len(current_data),
        },
        "evidently_report": str(report_path),
    }

    summary_path = REPORTS_DIR / "drift_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")
    print(f"  Evidently report → {report_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
