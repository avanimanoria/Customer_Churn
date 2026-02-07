import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1. Load engineered dataset
df = pd.read_csv("ott_churn_lifestyle_fe.csv")

# 2. Define target and features
target = "churn"
X = df.drop(columns=[target, "churn_reason"])  # keep churn_reason only for analysis
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

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification report:\n", classification_report(y_test, y_pred))


# 9. Save the trained pipeline (preprocessing + model)
joblib.dump(clf, "churn_model.pkl")
print("Model saved to churn_model.pkl")
