# Customer Churn Prediction for OTT and Lifestyle Services

An end-to-end machine-learning project that predicts churn risk for customers in a simulated OTT and lifestyle subscription business.

The project implements a reproducible batch ML workflow:

```text
Synthetic data generation
        ↓
Exploratory data analysis
        ↓
Feature engineering and validation
        ↓
Model training and comparison
        ↓
Threshold analysis
        ↓
Customer scoring
        ↓
Risk-banded retention outreach export
```

> **Important:** This project uses a self-generated synthetic dataset. It does not use real customer or company data. The results demonstrate ML pipeline design, evaluation, and decision-making—not validated real-world business performance.

---

## Problem Statement

Subscription businesses can lose revenue when customers cancel or stop renewing their plans. The goal of this project is to identify customers with a higher probability of churn so a retention team can prioritize appropriate outreach.

The project uses customer profile, subscription, engagement, usage, connectivity, tenure, and support-related features to estimate churn risk.

Rather than treating every predicted customer the same, the output assigns risk bands that can support different retention actions:

- **Medium risk:** Low-cost automated engagement, such as a reminder or personalized content recommendation
- **High risk:** Targeted in-app offer or retention campaign
- **Critical risk:** Higher-cost intervention, such as manual review or a premium incentive

---

## Key Features

- Generates a reproducible synthetic OTT/lifestyle customer dataset with 10,000 records
- Performs exploratory data analysis on churn distribution and feature relationships
- Creates engineered features such as `fee_per_platform` and `engagement_ratio`
- Validates data quality before model training
- Prevents target leakage by excluding customer identifiers and simulation-only target-derived columns
- Applies preprocessing using `StandardScaler` for numeric features and `OneHotEncoder` for categorical features
- Compares Logistic Regression and Random Forest classification models
- Evaluates class-weighted and unweighted model variants
- Uses ROC-AUC, precision, recall, F1-score, macro F1, confusion matrices, and average precision
- Performs threshold analysis to align churn predictions with retention campaign cost
- Scores all customers in batch and exports a risk-banded outreach list

---

## Dataset

The project uses a self-generated synthetic dataset containing **10,000 customer records** for an OTT and lifestyle subscription scenario.

### Example feature groups

| Category | Example features |
|---|---|
| Customer profile | `age`, `user_type`, `city_tier`, `lifestyle` |
| Engagement | `daily_free_hours`, `avg_ott_hours_per_day`, `days_since_last_login` |
| Subscription | `plan`, `monthly_fee`, `number_of_platforms`, `tenure_months` |
| Service experience | `net_quality`, `num_support_tickets_last_3m`, `auto_renewable_enabled` |
| Engineered features | `fee_per_platform`, `engagement_ratio` |
| Target | `churn`, where `1` indicates churn and `0` indicates non-churn |

### Data distribution

After updating the synthetic generator, the dataset contains:

| Class | Customer count | Share |
|---|---:|---:|
| Non-churn (`0`) | 6,150 | 61.5% |
| Churn (`1`) | 3,850 | 38.5% |

### Leakage prevention

The synthetic generator creates fields that help generate the target label. These fields are **not used as model inputs**:

- `customer_id`
- `churn`
- `simulated_churn_probability`
- `simulated_churn_reason`

This is important because target-derived fields would create data leakage and make the model evaluation misleading.

---

## Tech Stack

| Area | Tools |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn |
| Preprocessing | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder` |
| Models | Logistic Regression, Random Forest Classifier |
| Model persistence | Joblib |
| Visualizations | Matplotlib, Seaborn |
| Output artifacts | CSV, JSON, PNG |

---

## Repository Structure

```text
Customer_Churn/
│
├── generate_ott_churn_lifestyle.py
│   Generates the synthetic OTT/lifestyle customer dataset.
│
├── ott_churn_lifestyle_dataset.csv
│   Generated raw synthetic dataset.
│
├── performing_EDA.py
│   Prints dataset information, churn distribution, group-level churn rates,
│   and selected correlations.
│
├── feature_engineering.py
│   Creates engineered features and runs data-quality validation checks.
│
├── ott_churn_lifestyle_fe.csv
│   Feature-engineered dataset used for training and scoring.
│
├── model_training.py
│   Trains the Logistic Regression baseline, evaluates it, saves metrics,
│   and serializes the preprocessing-plus-model pipeline.
│
├── compare_models.py
│   Compares Logistic Regression and Random Forest models, including
│   class-weighted variants.
│
├── threshold_analysis.py
│   Evaluates multiple classification thresholds and saves a
│   precision-recall curve.
│
├── score_customers.py
│   Loads the trained pipeline and creates customer-level churn scores.
│
├── export_high_risk.py
│   Creates a risk-banded retention outreach list using a probability threshold.
│
├── churn_model.pkl
│   Serialized trained scikit-learn pipeline.
│
├── churn_scored.csv
│   Customer-level churn probability and default-threshold prediction output.
│
├── retention_outreach_candidates.csv
│   Prioritized export for low-cost retention outreach.
│
└── results/
    ├── metrics.json
    ├── model_comparison.csv
    ├── model_comparison.json
    ├── threshold_comparison.csv
    ├── threshold_analysis.json
    ├── precision_recall_curve.png
    └── high_risk_export_summary.csv
```

> Some generated CSV files are project artifacts. In a production repository, large generated data and model artifacts should usually be stored in object storage or generated locally rather than committed to Git.

---

## Installation

### Prerequisites

- Python 3.9 or later
- pip
- Git

### 1. Clone the repository

```powershell
git clone https://github.com/avanimanoria/Customer_Churn.git
cd Customer_Churn
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

---

## Run the Pipeline

Run the scripts in this order from the repository root.

### 1. Generate synthetic customer data

```powershell
python generate_ott_churn_lifestyle.py
```

This creates:

```text
ott_churn_lifestyle_dataset.csv
```

### 2. Create engineered features and validate data

```powershell
python feature_engineering.py
```

This creates:

```text
ott_churn_lifestyle_fe.csv
```

The script validates:

- No zero values in `number_of_platforms`
- `engagement_ratio` remains between `0` and `1`
- `customer_id` is unique
- `churn` contains only `0` and `1`
- Churn distribution is printed for review

### 3. Run exploratory data analysis

```powershell
python performing_EDA.py
```

The EDA script prints:

- Dataset schema and summary
- Churn class distribution
- Churn rate by user type
- Churn rate by subscription plan
- Churn rate by network quality
- Correlation of selected numeric features with churn

### 4. Train the baseline model

```powershell
python model_training.py
```

This trains a Logistic Regression pipeline using:

- `StandardScaler` for numerical features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features
- Stratified 80/20 train-test split
- `random_state=42`

It saves:

```text
churn_model.pkl
results/metrics.json
```

### 5. Compare models

```powershell
python compare_models.py
```

This compares:

- Logistic Regression
- Class-weighted Logistic Regression
- Random Forest
- Class-weighted Random Forest

The script saves:

```text
results/model_comparison.csv
results/model_comparison.json
```

### 6. Run threshold analysis

```powershell
python threshold_analysis.py
```

This evaluates thresholds from `0.30` to `0.70`, then saves:

```text
results/threshold_comparison.csv
results/threshold_analysis.json
results/precision_recall_curve.png
```

### 7. Score all customers

```powershell
python score_customers.py
```

This creates:

```text
churn_scored.csv
```

### 8. Create the retention outreach export

```powershell
python export_high_risk.py
```

This creates:

```text
retention_outreach_candidates.csv
results/high_risk_export_summary.csv
```

---

## Model Evaluation

The project uses a stratified 80/20 train-test split with `random_state=42`.

The model comparison was evaluated using class-specific churn precision, churn recall, churn F1-score, macro F1-score, ROC-AUC, and the confusion matrix. Accuracy alone was not used for model selection because it can hide unequal performance across churn and non-churn classes.

### Model comparison

| Model | Accuracy | Churn Precision | Churn Recall | Churn F1 | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6475 | 0.5724 | 0.3338 | 0.4217 | 0.5841 | **0.6527** |
| Logistic Regression, class-weighted | 0.6040 | 0.4883 | **0.5948** | **0.5363** | 0.5954 | 0.6526 |
| Random Forest | 0.6510 | **0.5887** | 0.3104 | 0.4065 | 0.5796 | 0.6504 |
| Random Forest, class-weighted | 0.6200 | 0.5062 | 0.5312 | 0.5184 | **0.6023** | 0.6448 |

### Selected operational model

The project uses a **class-weighted Random Forest** for threshold analysis and operational scoring.

It was chosen because it produced the best **macro F1-score (0.6023)** among the tested default-threshold models, giving the most balanced performance across churn and non-churn classes.

> The best ROC-AUC came from Logistic Regression at 0.6527. This is a moderate baseline result, not a production-grade predictive result. The project intentionally documents this trade-off instead of choosing a model based on accuracy alone.

### Threshold analysis

The final model produces a churn probability. Different thresholds support different retention strategies.

| Threshold | Churn Precision | Churn Recall | Churn F1 | Intended Use |
|---|---:|---:|---:|---|
| 0.30 | 0.4190 | **0.9234** | **0.5764** | Broad, low-cost automated outreach |
| 0.50 | 0.5062 | 0.5312 | 0.5184 | Balanced default classification |
| 0.70 | **0.7292** | 0.0455 | 0.0856 | High-confidence, expensive/manual intervention |

At the `0.30` threshold, the held-out test set results were:

```text
True negatives: 244
False positives: 986
False negatives: 59
True positives: 711
```

This threshold catches **711 of 770 actual churners** but also creates many false-positive alerts. Therefore, it should only be used for low-cost interventions such as automated email, notifications, or in-app messaging.

---

## Retention Outreach Output

The export script uses:

```python
CHURN_RISK_THRESHOLD = 0.30
```

This threshold is intentionally broad. It selects customers for **retention outreach**, not a list of only highly certain churners.

### Current exported cohort

| Risk band | Probability range | Customers selected | Recommended action |
|---|---:|---:|---|
| Medium | 0.30–0.50 | 4,348 | Automated, low-cost engagement outreach |
| High | 0.50–0.70 | 2,115 | Targeted in-app retention campaign |
| Critical | 0.70+ | 244 | Higher-cost offer or manual review |
| Total | 0.30+ | 6,707 | Broad retention outreach cohort |

The output is sorted by churn probability and includes:

```text
customer_id
churn_prob
risk_band
user_type
plan
net_quality
monthly_fee
lifestyle
```

The export intentionally does **not** include `simulated_churn_reason`, because a real operational system would not know the true reason for a future churn event at prediction time.

---

## Business Interpretation

This project demonstrates how model scores can be converted into operational segments instead of being treated as a single yes/no answer.

For example:

- Customers in the **medium-risk** band can receive low-cost engagement nudges.
- Customers in the **high-risk** band can receive targeted content, plan, or renewal interventions.
- Customers in the **critical-risk** band can be reviewed for more expensive retention action.

A real business would choose the final threshold using intervention cost, customer lifetime value, expected retention uplift, and available campaign capacity.

---

## Limitations

- The dataset is synthetic, so the results do not prove real-world churn performance.
- The synthetic target is stochastic, meaning some churn outcomes contain intentional randomness.
- The best ROC-AUC is moderate, so the models are baseline experiments rather than production-ready predictive systems.
- The current pipeline is batch-oriented and does not provide real-time inference.
- The current project does not include model monitoring, drift detection, scheduled retraining, or feature-store infrastructure.
- The threshold was examined on the held-out test set for this portfolio analysis. In a production workflow, threshold selection should use a separate validation set, followed by a final untouched test set.
- Real deployment would require privacy review, data governance, fairness checks, logging, security controls, and business validation.

---

## Future Improvements

- Add a `requirements.txt` file with pinned package versions
- Add unit tests for feature engineering and schema validation
- Add a data dictionary for all source and engineered features
- Use cross-validation and hyperparameter tuning
- Add feature importance and SHAP-based model explainability
- Add calibration analysis for predicted probabilities
- Use a dedicated validation set for model and threshold selection
- Add MLflow or Weights & Biases for experiment tracking
- Create a FastAPI inference service
- Build a Streamlit dashboard for churn-risk exploration
- Add scheduled batch jobs and model-monitoring workflows
- Replace synthetic data with appropriately governed real historical data in a production setting

---

## Resume Description

**Customer Churn Prediction | Python, Pandas, scikit-learn**

Built an end-to-end churn-risk scoring pipeline on a self-generated 10,000-record OTT/lifestyle dataset; implemented feature engineering, leakage prevention, model comparison, threshold analysis, batch scoring, and risk-banded retention outreach exports.

---

## Author

**Avani Manoria**

- GitHub: [@avanimanoria](https://github.com/avanimanoria)
- Project repository: [Customer_Churn](https://github.com/avanimanoria/Customer_Churn)