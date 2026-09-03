import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "ott_churn_lifestyle_fe.csv"
TARGET = "churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20

LEAKAGE_COLUMNS = [
    "customer_id",
    "churn",
    "simulated_churn_probability",
    "simulated_churn_reason",
]

NUMERIC_FEATURES = [
    "age",
    "daily_free_hours",
    "avg_ott_hours_per_day",
    "number_of_platforms",
    "monthly_fee",
    "tenure_months",
    "days_since_last_login",
    "num_support_tickets_last_3m",
    "fee_per_platform",
    "engagement_ratio",
]

CATEGORICAL_FEATURES = [
    "user_type",
    "city_tier",
    "primary_device",
    "net_quality",
    "plan",
    "auto_renewable_enabled",
    "lifestyle",
]

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=LEAKAGE_COLUMNS)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]

thresholds_to_test = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
rows = []

for threshold in thresholds_to_test:
    y_pred = (y_prob >= threshold).astype(int)

    rows.append(
        {
            "threshold": threshold,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision_churn": round(
                precision_score(y_test, y_pred, zero_division=0), 4
            ),
            "recall_churn": round(
                recall_score(y_test, y_pred, zero_division=0), 4
            ),
            "f1_churn": round(
                f1_score(y_test, y_pred, zero_division=0), 4
            ),
            "macro_f1": round(
                f1_score(y_test, y_pred, average="macro", zero_division=0), 4
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
    )

results_df = pd.DataFrame(rows)

best_row = results_df.loc[results_df["f1_churn"].idxmax()]

Path("results").mkdir(exist_ok=True)

results_df.to_csv("results/threshold_comparison.csv", index=False)

with open("results/threshold_analysis.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "model": "RandomForestClassifier(class_weight='balanced')",
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "average_precision": round(float(average_precision_score(y_test, y_prob)), 4),
            "best_threshold_by_churn_f1": best_row.to_dict(),
        },
        file,
        indent=2,
    )

precision, recall, curve_thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Churn Prediction: Precision-Recall Curve")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/precision_recall_curve.png", dpi=150)
plt.close()

print("\n--- Threshold Comparison ---")
print(results_df.to_string(index=False))

print("\n--- Selected threshold by highest churn F1 ---")
print(best_row.to_string())

print(f"\nROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Average Precision: {average_precision_score(y_test, y_prob):.4f}")

print("\nSaved:")
print("- results/threshold_comparison.csv")
print("- results/threshold_analysis.json")
print("- results/precision_recall_curve.png")