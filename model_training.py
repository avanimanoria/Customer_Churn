import json
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 1. Load engineered dataset
df = pd.read_csv("ott_churn_lifestyle_fe.csv")

# 2. Define target and features
target = "churn"
LEAKAGE_COLUMNS = [
    "customer_id",
    "churn",
    "simulated_churn_probability",
    "simulated_churn_reason",
]

X = df.drop(columns=LEAKAGE_COLUMNS)
y = df[target] 

# 3. Identify numeric and categorical columns
numeric_features = [
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


categorical_features = [
    "user_type",
    "city_tier",
    "primary_device",
    "net_quality",
    "plan",
    "auto_renewable_enabled",
    "lifestyle",
]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 5. Preprocessing: scale numeric, one‑hot encode categoricals
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 6. Model pipeline: preprocessing + logistic regression
clf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

# 7. Train the model
clf.fit(X_train, y_train)

# 8. Evaluate on test set
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

metrics = {
    "model": "LogisticRegression",
    "random_state": 42,
    "test_size": 0.20,
    "positive_class": "1 = churn",
    "accuracy": round(float(accuracy), 4),
    "precision_churn": round(float(precision), 4),
    "recall_churn": round(float(recall), 4),
    "f1_score_churn": round(float(f1), 4),
    "roc_auc": round(float(roc_auc), 4),
    "confusion_matrix": cm.tolist(),
}

print("\n--- Model Evaluation ---")
print(f"Model:     {metrics['model']}")
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision for churn (class 1): {metrics['precision_churn']:.4f}")
print(f"Recall for churn (class 1):    {metrics['recall_churn']:.4f}")
print(f"F1-score for churn (class 1):  {metrics['f1_score_churn']:.4f}")
print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

Path("results").mkdir(exist_ok=True)

with open("results/metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nSaved metrics to results/metrics.json")


# 9. Save the trained pipeline (preprocessing + model)
joblib.dump(clf, "churn_model.pkl")
print("Model saved to churn_model.pkl")
