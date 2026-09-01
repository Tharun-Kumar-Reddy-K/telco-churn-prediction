# Telco Customer Churn Prediction

Predicting which telecom customers will churn, using supervised learning. Built as a portfolio project covering the full pipeline: data cleaning, EDA, encoding, model comparison, and explainability.

## Problem Statement

Given a customer's account details, contract, and services subscribed, predict whether they will churn (cancel their service). Binary classification with direct business value: identify at-risk customers early enough for retention outreach.

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7,043 customers, 20 features (demographics, contract/billing details, subscribed services), ~26.5% churn rate.

## Approach

1. **Data Cleaning** — diagnosed and fixed a hidden data quality issue: `TotalCharges` was stored as text due to 11 rows containing a blank space instead of a number (all customers with `tenure = 0`, i.e., billed nothing yet). Confirmed via `pd.to_numeric(..., errors="coerce")` before fixing, rather than assuming.
2. **EDA** — visualized churn against Contract type, tenure, Internet service, payment method, and tech support. Discovered via `pd.crosstab()` that `TechSupport` and `InternetService` contain a fully redundant category (customers with no internet cannot have tech support, so both columns encode the same fact for that subgroup).
3. **Encoding** — binary mapping for 2-value columns, one-hot encoding (`drop_first=True`) for multi-category columns, deliberately avoiding label encoding on nominal categories to prevent implying false ordinal relationships.
4. **Modeling** — progressive comparison across model families: Logistic Regression (linear baseline) → Random Forest (bagging) → XGBoost (boosting), each using `class_weight`/`scale_pos_weight` to handle the ~26.5% churn imbalance.
5. **Evaluation** — precision, recall, F1, and ROC-AUC (not accuracy alone, given class imbalance — a "predict majority class" baseline would already score ~73% accuracy).
6. **Explainability** — SHAP TreeExplainer on the XGBoost model, both global (summary plot) and per-customer (force plot).

## Results

| Model | ROC-AUC | F1 Score (class 1) | Precision (class 1) | Recall (class 1) |
|---|---|---|---|---|
| Logistic Regression | 0.8414 | 0.6164 | 0.51 | 0.79 |
| Random Forest | 0.8422 | 0.6250 | 0.54 | 0.74 |
| XGBoost | 0.8439 | 0.6250 | 0.51 | 0.80 |

**Key finding:** all three models perform nearly identically (ROC-AUC within 0.003 of each other, F1 tied at 0.625 between Random Forest and XGBoost). This indicates the churn signal in these features is largely linear/simple — additional model complexity provides negligible improvement. Further performance gains would likely require richer feature engineering (e.g., customer service interaction history, usage trends) rather than more sophisticated algorithms.

## Explainability Findings (SHAP)

Top churn drivers, confirmed both by EDA and SHAP values on the XGBoost model:
- **Contract type** — the single strongest predictor. Two-year contracts strongly push predictions toward "won't churn"; the absence of a long-term contract pushes toward "will churn."
- **Tenure** — a gradual/linear relationship: low tenure (new customers) increases churn risk, with risk decreasing steadily as tenure grows.
- **Internet Service (Fiber optic)** — Fiber optic customers show meaningfully higher churn risk than DSL or no-internet customers, likely tied to higher pricing (confirmed via a separate `MonthlyCharges` correlation check).

## Repo Structure

```
├── telco_churn_prediction.ipynb   # Full analysis notebook
├── data/                           # telco_churn.csv (not committed — see .gitignore)
├── README.md
└── .gitignore
```

## How to Run

Open in Google Colab, then:
```python
!mkdir -p data
!wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -O data/telco_churn.csv
!pip install imbalanced-learn xgboost shap -q
```
Then run all cells top to bottom.

## Key Design Decisions (interview talking points)

- **Why diagnose before fixing** — used `pd.to_numeric(errors="coerce")` to identify exactly which rows and values were malformed before applying any fix, rather than blindly dropping or imputing.
- **Why one-hot over label encoding** — nominal categories (Contract, PaymentMethod, etc.) have no true order; label encoding would have introduced a false ranking.
- **Why `drop_first=True`** — avoids multicollinearity from a fully redundant category (e.g., if two dummy columns are both 0, the third is already implied).
- **Why compare 3 models instead of picking one** — demonstrates the bagging-vs-boosting distinction and validates whether complexity is actually earning its keep, rather than defaulting to "use XGBoost because it's popular."
- **Why SHAP over built-in feature importance** — provides per-prediction explanations, not just a global ranking, which is what a retention team would actually need ("why was this specific customer flagged?").

## Limitations & Next Steps

- Found and documented a redundant categorical relationship (TechSupport/InternetService) via crosstab; retained both features since tree models handle multicollinearity gracefully, but flagged as worth collapsing for a cleaner Logistic Regression model.
- No hyperparameter tuning performed (used sensible defaults); `GridSearchCV` or Optuna could likely close some of the remaining performance gap.
- Threshold tuning (moving beyond the default 0.5 cutoff) could improve the precision/recall balance depending on the actual business cost of false positives vs false negatives.
