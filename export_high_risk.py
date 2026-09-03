from pathlib import Path

import pandas as pd


INPUT_PATH = "churn_scored.csv"
OUTPUT_PATH = "high_risk_customers.csv"

# Broad, low-cost retention campaign threshold.
# Selected from validation/test threshold analysis.
CHURN_RISK_THRESHOLD = 0.30


df = pd.read_csv(INPUT_PATH)

high_risk = df[df["churn_prob"] >= CHURN_RISK_THRESHOLD].copy()

high_risk["risk_band"] = pd.cut(
    high_risk["churn_prob"],
    bins=[0.0, 0.50, 0.70, 1.0],
    labels=["medium", "high", "critical"],
    include_lowest=True,
)

high_risk = high_risk[
    [
        "customer_id",
        "churn_prob",
        "risk_band",
        "user_type",
        "plan",
        "net_quality",
        "monthly_fee",
        "lifestyle",
    ]
].sort_values("churn_prob", ascending=False)

Path("results").mkdir(exist_ok=True)

high_risk.to_csv(OUTPUT_PATH, index=False)

summary = (
    high_risk["risk_band"]
    .value_counts(dropna=False)
    .rename_axis("risk_band")
    .reset_index(name="customers")
)

summary.to_csv("results/high_risk_export_summary.csv", index=False)

print(f"High-risk threshold: {CHURN_RISK_THRESHOLD:.2f}")
print(f"Customers selected: {len(high_risk)}")
print("\nCustomers by risk band:")
print(high_risk["risk_band"].value_counts(dropna=False))
print(f"\nSaved prioritized customer list to: {OUTPUT_PATH}")
print("Saved export summary to: results/high_risk_export_summary.csv")