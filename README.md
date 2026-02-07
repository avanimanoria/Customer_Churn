
Customer Churn — OTT Lifestyle
================================

Imagine your OTT platform as a buzzing Friday‑night city: millions log in, scroll endlessly, then—without a sound—some of them simply vanish. This project is about finding those silent exits *before* they happen, and turning them into second chances. 

Overview
--------

- **Project:**
 Predict and score customers at risk of churn for an OTT lifestyle dataset.
- **Purpose:** 
Process raw OTT customer data, build a churn model, score customers, and export high‑risk lists for downstream action.

Instead of guessing why people leave, this repository builds a data‑driven early‑warning system that flags **who** is likely to churn and **why**, so product, marketing, and support teams can intervene in time. The dataset reflects everyday streaming life: college students on basic plans, working professionals on premium, retired users on news and family content, each with a churn probability, predicted churn flag, and a human‑readable churn reason such as “too expensive”, “no time to watch”, “technical issues”, or “switched to competitor”. 

Repository Structure
--------------------

### Files

- **generate_ott_churn_lifestyle.py**  
  Ingests or generates the raw OTT churn dataset, simulating user types, plans, net quality, monthly fees, and viewing lifestyles.

- **ott_churn_lifestyle_dataset.csv**  
  Raw OTT lifestyle dataset and source for feature engineering.

- **feature_engineering.py**  
  Transforms raw data into model‑ready features and saves the engineered dataset.

- **ott_churn_lifestyle_fe.csv**  
  Feature‑engineered dataset produced by `feature_engineering.py`. 

- **performing_EDA.py**  
  Exploratory data analysis scripts and visualizations to validate assumptions and feature choices. 

- **model_training.py**  
  Trains the churn model, handles encoders/scalers, and persists all model artifacts. 

- **score_customers.py**  
  Loads model artifacts and scores customers; writes `churn_scored.csv` with `customerid`, `churnprob`, `churnpred`, `churnreason`, `usertype`, `plan`, `netquality`, `monthlyfee`, and `lifestyle`. 

- **churn_scored.csv**  
  Example output containing full customer‑level churn probabilities, predictions, and reasons. 

- **export_high_risk.py**  
  Filters scored customers by a probability threshold (e.g., ≥ 0.7) and produces `high_risk_customers.csv` and `high_risk_customers_uipath.csv` for downstream consumers.

- **high_risk_customers.csv** and **high_risk_customers_uipath.csv**  
  Actionable lists of high‑risk customers including churn reason, user type, plan, network quality, and lifestyle segment. 

Quick Start
-----------

1. Create and activate a Python environment (Python 3.8+ recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

3. Run the end‑to‑end pipeline:

```powershell
python generate_ott_churn_lifestyle.py
python feature_engineering.py
python model_training.py
python score_customers.py
python export_high_risk.py
```

Notes: Some scripts may accept arguments (paths, thresholds, seeds). Check each script’s top‑level docstring for configurable options. 

Outputs & Where to Find Them
----------------------------

- **Feature‑engineered data**  
  `ott_churn_lifestyle_fe.csv` — cleaned and transformed dataset for modelling and EDA. 

- **Trained model & artifacts**  
  Saved by `model_training.py` (model, encoders, scalers); see that script for exact paths. 

- **Scored customers**  
  `churn_scored.csv` — customer‑level churn probabilities, predictions, and churn reasons. 

- **High‑risk exports**  
  `high_risk_customers.csv` and `high_risk_customers_uipath.csv` — filtered lists of high‑risk customers ready for CRM, marketing, or RPA workflows. 

Scripts Summary
---------------

- **Data prep:**  
  `generate_ott_churn_lifestyle.py` and `feature_engineering.py` handle data creation/ingestion and preprocessing.

- **EDA:**  
  `performing_EDA.py` explores patterns such as which lifestyles or plans tend to be tagged “too expensive” or “no time to watch” at higher churn probabilities. 

- **Modeling:**  
  `model_training.py` trains and persists the churn model and preprocessing pipeline. 

- **Scoring & export:**  
  `score_customers.py` scores customers using the trained model; `export_high_risk.py` extracts high‑risk customers for direct outreach or automation. 

Implementation‑Ready Business Conclusion
----------------------------------------

This project turns churn prediction from a one‑off notebook into a **repeatable retention engine**: on a daily or weekly schedule, you can regenerate high‑risk customer lists and immediately trigger precise, segment‑wise actions instead of blanket discounts. Price‑sensitive students flagged as “too expensive” can receive targeted plan offers, users with “technical issues” can be auto‑assigned to proactive support, and “no time to watch” customers can get concise, personalized content suggestions rather than generic promos.

By wiring `high_risk_customers_uipath.csv` into your CRM, marketing automation, or UiPath bots, the OTT platform can systematically reduce churn, protect lifetime value, and continuously experiment with interventions—closing the Friday‑night leak one high‑risk customer at a time. 
=======
