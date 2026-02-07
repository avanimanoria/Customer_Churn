import numpy as np
import pandas as pd

#Load the existing dataset
df = pd.read_csv("ott_churn_lifestyle_dataset.csv")

#Creating fee_per_platform
#how much the user pays per OTT platform they subscribe to
df["fee_per_platform"] = df["monthly_fee"] / df["number_of_platforms"]

#Creating engagement_ratio
#fraction of free time spent watching OTT
#using np.where to avoid dividing by zero when daily_free_hours is 0
df["engagement_ratio"] = np.where(
    df["daily_free_hours"] > 0,
    df["avg_ott_hours_per_day"] / df["daily_free_hours"],
    0.0,
)

# Clipping engagement_ratio to [0, 1] just to keep it interpretable
df["engagement_ratio"] = df["engagement_ratio"].clip(0, 1)

# Saving the enhanced dataset
df.to_csv("ott_churn_lifestyle_fe.csv", index=False)
