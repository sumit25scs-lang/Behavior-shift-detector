"""
Baseline Category Classifier
=============================
Trains a RandomForestClassifier to predict product Category from
transactional features, using ONLY the first 5 months of data
(before the injected behavioral shift at month 6).

Features:
  - Spend_Amount           (continuous)
  - Customer_Frequency     (frequency-encoded Customer_ID)
  - DayOfWeek              (0=Mon … 6=Sun)
  - Month                  (1-5)
  - DayOfMonth             (1-31)
  - IsWeekend              (binary)

Output:
  - Prints accuracy + classification report
  - Saves trained model → src/baseline_model.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Load data ─────────────────────────────────────────────────────────────
DATA_PATH = Path("data/transactions.csv")
MODEL_DIR = Path("src")
MODEL_PATH = MODEL_DIR / "baseline_model.pkl"

print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# ── Filter to first 5 months ONLY (pre-shift) ────────────────────────────
df = df[df["Date"].dt.month <= 5].copy()
print(f"Pre-shift records (months 1-5): {len(df):,}")

# ── Feature Engineering ───────────────────────────────────────────────────

# Date-derived features
df["DayOfWeek"] = df["Date"].dt.day_of_week      # 0=Mon, 6=Sun
df["Month"] = df["Date"].dt.month
df["DayOfMonth"] = df["Date"].dt.day
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

# Frequency-encode Customer_ID
customer_freq = df["Customer_ID"].value_counts()
df["Customer_Frequency"] = df["Customer_ID"].map(customer_freq)

# ── Prepare X, y ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Spend_Amount",
    "Customer_Frequency",
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "IsWeekend",
]

X = df[FEATURE_COLS]
y = df["Category"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nFeatures: {FEATURE_COLS}")
print(f"Target classes: {list(le.classes_)}")
print(f"Class distribution:\n{pd.Series(y).value_counts().to_string()}\n")

# ── Train / Test split (80/20, stratified) ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)
print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ── Train RandomForestClassifier ──────────────────────────────────────────
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"  BASELINE MODEL ACCURACY: {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"{'='*60}\n")

print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
))

# ── Feature importances ──────────────────────────────────────────────────
print("Feature Importances:")
for name, imp in sorted(
    zip(FEATURE_COLS, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"  {name:<25s} {imp:.4f}")

# ── Save model + label encoder ───────────────────────────────────────────
MODEL_DIR.mkdir(exist_ok=True)

artifact = {
    "model": model,
    "label_encoder": le,
    "feature_columns": FEATURE_COLS,
}
joblib.dump(artifact, MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")

