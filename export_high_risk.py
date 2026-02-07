import pandas as pd

# 1. Load scored customers
df = pd.read_csv("churn_scored.csv")

# 2. Define a churn probability threshold
threshold = 0.7  # you can adjust this later

# 3. Filter high-risk customers
high_risk = df[(df["churn_pred"] == 1) & (df["churn_prob"] >= threshold)]

# 4. Keep useful columns only (adjust as you like)
high_risk = high_risk[
    [
        "customer_id",
        "churn_prob",
        "churn_pred",
        "churn_reason",
        "user_type",
        "plan",
        "net_quality",
        "monthly_fee",
        "lifestyle",
    ]
]

# 5. Save to CSV for UiPath
high_risk.to_csv("high_risk_customers.csv", index=False)
print("High-risk customers saved to high_risk_customers.csv")
