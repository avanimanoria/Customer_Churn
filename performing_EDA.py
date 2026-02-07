import numpy as np
import pandas as pd

#dataframe
df = pd.read_csv("ott_churn_lifestyle_dataset.csv")
print(df.head())#print few rows
print(df.info())#print summary of df
value_counts = df['churn'].value_counts()
print("Churn Value Counts:\n", value_counts)
#The dataset contains 10,000 observations, 
# with roughly 79% labeled as churners and 21% as non‑churners, 
# indicating a high churn environment.

# Churn rate by user_type
print("\nChurn rate by user_type:")
print(df.groupby('user_type')['churn'].mean())

# Churn rate by plan
print("\nChurn rate by plan:")
print(df.groupby('plan')['churn'].mean())

# Churn rate by net_quality
print("\nChurn rate by net_quality:")
print(df.groupby('net_quality')['churn'].mean())


# Correlation between key numeric features and churn
corr_cols = ['monthly_fee', 'tenure_months', 'days_since_last_login', 'churn']
print("\nCorrelation between key numeric features and churn:")
print(df[corr_cols].corr())
