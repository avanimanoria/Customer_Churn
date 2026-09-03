import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
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

models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "logistic_regression_balanced": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "random_forest_balanced": RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    ),
}

results = {}

for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_churn": round(
            float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "recall_churn": round(
            float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "f1_churn": round(
            float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)), 4
        ),
        "macro_f1": round(
            float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4
        ),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.sort_values(
    by=["roc_auc", "f1_churn"],
    ascending=False,
)

Path("results").mkdir(exist_ok=True)

comparison_df.to_csv("results/model_comparison.csv")

with open("results/model_comparison.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\n--- Model Comparison ---")
print(comparison_df.to_string())

print("\nSaved:")
print("- results/model_comparison.csv")
print("- results/model_comparison.json")