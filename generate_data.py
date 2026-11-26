"""
Synthetic Retail Transaction Generator
=======================================
Generates 10,000 retail transactions over 12 months with an artificial
behavioral shift injected at month 6:
  - Electronics purchase probability drops by 50%
  - Home Goods purchase probability spikes by 70%
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
NUM_TRANSACTIONS = 10_000
NUM_CUSTOMERS = 500
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
SEED = 42

# Category probability distributions (must sum to 1.0)
# Baseline (months 1-5):
BASELINE_PROBS = {
    "Electronics": 0.30,
    "Clothing": 0.25,
    "Home Goods": 0.20,
    "Groceries": 0.25,
}

# Spend-amount parameters per category (mean, std)
SPEND_PARAMS = {
    "Electronics": (250, 120),
    "Clothing": (80, 35),
    "Home Goods": (150, 60),
    "Groceries": (45, 20),
}

# ── RNG ────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)


def build_shifted_probs() -> dict:
    """Apply the month-6 behavioral shift and re-normalise."""
    shifted = dict(BASELINE_PROBS)
    shifted["Electronics"] *= 0.50  # drop 50%
    shifted["Home Goods"] *= 1.70  # spike 70%
    # Re-normalise so probabilities sum to 1
    total = sum(shifted.values())
    return {k: v / total for k, v in shifted.items()}


SHIFTED_PROBS = build_shifted_probs()

# ── Generate dates ─────────────────────────────────────────────────────────
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
dates = rng.choice(date_range, size=NUM_TRANSACTIONS)
dates = pd.to_datetime(dates)

# ── Assign customers ──────────────────────────────────────────────────────
customer_ids = rng.integers(1, NUM_CUSTOMERS + 1, size=NUM_TRANSACTIONS)

# ── Assign categories with the month-6 shift ──────────────────────────────
categories_list = list(BASELINE_PROBS.keys())
months = pd.Series(dates).dt.month.values

categories = np.empty(NUM_TRANSACTIONS, dtype=object)

pre_mask = months < 6
post_mask = ~pre_mask  # month >= 6

baseline_p = [BASELINE_PROBS[c] for c in categories_list]
shifted_p = [SHIFTED_PROBS[c] for c in categories_list]

categories[pre_mask] = rng.choice(categories_list, size=pre_mask.sum(), p=baseline_p)
categories[post_mask] = rng.choice(categories_list, size=post_mask.sum(), p=shifted_p)

# ── Generate spend amounts ────────────────────────────────────────────────
spend_amounts = np.empty(NUM_TRANSACTIONS, dtype=float)
for cat, (mu, sigma) in SPEND_PARAMS.items():
    mask = categories == cat
    n = mask.sum()
    spend_amounts[mask] = rng.normal(mu, sigma, size=n)

# Floor at $1 – no negative or zero transactions
spend_amounts = np.maximum(spend_amounts, 1.0)
spend_amounts = np.round(spend_amounts, 2)

# ── Assemble DataFrame ───────────────────────────────────────────────────
df = pd.DataFrame(
    {
        "Date": dates,
        "Customer_ID": customer_ids,
        "Category": categories,
        "Spend_Amount": spend_amounts,
    }
)

df = df.sort_values("Date").reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "transactions.csv"
df.to_csv(out_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"Saved {len(df):,} transactions → {out_path}")
print(f"\nBaseline probs (months 1-5): {BASELINE_PROBS}")
print(
    f"Shifted  probs (months 6-12): { {k: round(v, 4) for k, v in SHIFTED_PROBS.items()} }"
)
print(f"\n── Category distribution by period ──")

df["Period"] = np.where(df["Date"].dt.month < 6, "Pre-shift (1-5)", "Post-shift (6-12)")
print(
    df.groupby("Period")["Category"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .round(4)
    .to_string()
)

print(f"\n── Sample rows ──")
print(df.drop(columns="Period").head(10).to_string(index=False))
