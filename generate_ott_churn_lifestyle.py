import numpy as np
import pandas as pd
import random
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
N = 10000 #total number of rows i want in my dataset

user = ["school student", "college student", "working professional", "retired individual"]
devices = ["phone", "tablet", "laptop", "smart TV", "other device"]
internet_level = ["low", "medium", "high"]
plan_types = ["basic", "standard", "premium"]
city = ["tier 1", "tier 2", "tier 3"]
simulated_churn_reasons = [
    "too_expensive",
    "low_engagement",
    "technical_issues",
    "competitor_switch",
    "content_dissatisfaction",
]
lifestyle = ["binge watcher", "casual viewer", "family viewer", "sports enthusiast", "news junkie"] 

data = [] #empty dictionary to hold data

#for each customer
for i in range(N):
    customer_id = f"CUST_{i+1:05d}" #unique customer id
    user_type = random.choices(
    user,
    weights=[0.10, 0.40, 0.35, 0.15],)[0]

    if user_type == "school student":
        age = np.random.randint(10, 18)
        daily_free_hours = np.random.normal(4, 1.5) #normal distribution for free hours
        #number of free hours should be between 0 and 8 so clipping them because free hourse cnan't be negative or more than 8 for a school student
        daily_free_hours = np.clip(daily_free_hours, 0, 8) 
    elif user_type == "college student":
        age = np.random.randint(18, 25)
        daily_free_hours = np.random.normal(3.5, 1.5)
        daily_free_hours = np.clip(daily_free_hours, 0, 10)
    elif user_type == "working professional":
        age = np.random.randint(25, 50)
        daily_free_hours = np.random.normal(2, 1)
        daily_free_hours = np.clip(daily_free_hours, 0, 6)
    else:
        age = np.random.randint(50, 90)
        daily_free_hours = np.random.normal(3, 1)
        daily_free_hours = np.clip(daily_free_hours, 0, 8)

    #to load random strings from the lists above
    city_tier = random.choice(city)
    avg_ott_hours_per_day = daily_free_hours * np.random.uniform(0.3, 0.9)
    avg_ott_hours_per_day = np.clip(avg_ott_hours_per_day, 0, daily_free_hours)
    number_of_platforms = np.random.randint(1, 4)
    primary_device = np.random.choice(devices)
    net_quality = np.random.choice(internet_level, p=[0.2, 0.3, 0.5]) #assuming more people have high quality internet(probability)
    plan = np.random.choice(plan_types, p=[0.3, 0.4, 0.3]) #assuming standard plan is more popular

    #monthly fee based on plan and city tier
    if plan=="basic":
        base_fee = 199
    elif plan=="standard":
        base_fee = 399
    else:
        base_fee = 699


    if city_tier == "tier 1":
        city_factor = 1.1
    elif city_tier == "tier 2":
        city_factor = 1.0
    else:
        city_factor = 0.9

    #calculate monthly fee with some randomness
    monthly_fee = base_fee * city_factor
    monthly_fee += np.random.normal(0, 20)
    monthly_fee = max(99, monthly_fee)
    monthly_fee = round(monthly_fee, 2)

    tenure_months = np.random.randint(1, 36)
    auto_renewable_enabled = np.random.choice([True, False])
    days_since_last_login = np.random.randint(0, 60)
    num_support_tickets_last_3m = np.random.poisson(0.5)#how many times user contacted support in last 3 months

    # Synthetic churn-risk model.
# Higher values increase churn probability; the intercept keeps churn
# near a realistic minority rate for this simulated use case.
    logit = -2.0

    # Short-tenure customers are more likely to leave.
    if tenure_months < 3:
        logit += 0.90
    elif tenure_months < 6:
        logit += 0.45

    # More platforms can increase subscription cost/choice pressure.
    if number_of_platforms >= 3:
        logit += 0.50
    elif number_of_platforms == 2:
        logit += 0.20

    # Low engagement and limited free time can increase cancellation risk.
    if daily_free_hours < 1:
        logit += 0.70
    elif daily_free_hours < 3:
        logit += 0.25

    # Poor connectivity hurts the streaming experience.
    if net_quality == "low":
        logit += 0.75
    elif net_quality == "medium":
        logit += 0.25

    # Inactivity is a strong churn signal.
    if days_since_last_login > 30:
        logit += 0.90
    elif days_since_last_login > 15:
        logit += 0.35

    # Auto-renewal reduces accidental/non-intentional churn.
    if not auto_renewable_enabled:
        logit += 0.60
    else:
        logit -= 0.20

    # Multiple support tickets can indicate unresolved issues.
    if num_support_tickets_last_3m > 3:
        logit += 0.60
    elif num_support_tickets_last_3m > 1:
        logit += 0.25

    # Add noise so labels are not deterministic.
    logit += np.random.normal(0, 0.50)

    churn_probability_simulated = 1 / (1 + np.exp(-logit))
    churn = np.random.binomial(1, churn_probability_simulated)

    if churn == 1:
        if monthly_fee > 600 or number_of_platforms >= 3:
            simulated_churn_reason = "too expensive"
        elif daily_free_hours < 2 or avg_ott_hours_per_day < 1:
            simulated_churn_reason = "no time to watch"
        elif net_quality in ["low", "medium"]:
            simulated_churn_reason = "technical issues"
        elif tenure_months >= 12:
            simulated_churn_reason = "switched to competitor"
        else:
            simulated_churn_reason = "content not interesting"
    else:
        simulated_churn_reason = "none"

    row = {
        
        "customer_id": customer_id,
        "user_type": user_type,
        "age": age,
        "city_tier": city_tier,
        "daily_free_hours": daily_free_hours,
        "avg_ott_hours_per_day": avg_ott_hours_per_day,
        "number_of_platforms": number_of_platforms,
        "primary_device": primary_device,
        "net_quality": net_quality,
        "plan": plan,
        "monthly_fee": monthly_fee,
        "tenure_months": tenure_months,
        "auto_renewable_enabled": auto_renewable_enabled,
        "days_since_last_login": days_since_last_login,
        "num_support_tickets_last_3m": num_support_tickets_last_3m,
        "simulated_churn_probability": churn_probability_simulated,
        "churn": churn,
        "simulated_churn_reason": simulated_churn_reason,
        "lifestyle": random.choice(lifestyle)}

    data.append(row)

df = pd.DataFrame(data)
df.to_csv("ott_churn_lifestyle_dataset.csv", index=False)
