import pandas as pd
import joblib

# 1. Load the engineered dataset
df = pd.read_csv("ott_churn_lifestyle_fe.csv")

# 2. Load the trained model pipeline
model = joblib.load("churn_model.pkl")

# 3. Prepare features (X) the same way as during training
#    Drop target and any columns you don't want as inputs
X = df.drop(columns=["churn", "churn_reason"])

# 4. Predict churn probabilities and labels for all customers
churn_prob = model.predict_proba(X)[:, 1]   # probability of churn = 1
churn_pred = model.predict(X)               # predicted class: 0 or 1

# 5. Build an output DataFrame with key columns
scored_df = pd.DataFrame({
    "customer_id": df["customer_id"],
    "churn_prob": churn_prob,
    "churn_pred": churn_pred,
    "churn_reason": df["churn_reason"],
    "user_type": df["user_type"],
    "plan": df["plan"],
    "net_quality": df["net_quality"],
    "monthly_fee": df["monthly_fee"],
    "lifestyle": df["lifestyle"],
})

# 6. Save predictions for use by UiPath or analysis
scored_df.to_csv("churn_scored.csv", index=False)
print("Scored customers saved to churn_scored.csv")
