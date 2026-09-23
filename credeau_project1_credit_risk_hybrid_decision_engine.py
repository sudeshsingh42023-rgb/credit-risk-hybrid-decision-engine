"""
Project 1: Risk-Based Underwriting & Hybrid Decision Engine
(Bureau + Banking + Rule-Engine ML Scorecard, for Credeau Data Analyst/Data Scientist role)

Simulates a realistic loan-applicant pool with bureau, financial and transactional
features, builds an ML credit-risk scorecard, blends it with a policy rule layer
into a 3-way decision (Approve / Refer / Reject), and backtests the hybrid
strategy against a rules-only baseline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

rng = np.random.default_rng(42)
N = 25000

# ---------------------------------------------------------------
# 1. Simulate bureau + financial + transactional applicant data
# ---------------------------------------------------------------
age = rng.integers(21, 60, N)
income = np.round(rng.lognormal(mean=10.8, sigma=0.45, size=N), -2)  # monthly income, INR
bureau_score = np.clip(rng.normal(700, 90, N), 300, 900).astype(int)
existing_loans = rng.poisson(1.1, N)
dti = np.clip(rng.normal(35, 15, N) + existing_loans * 4, 2, 95)  # debt-to-income %
avg_bank_balance = np.round(np.clip(rng.lognormal(9.2, 0.6, N), 500, None), -2)
bounced_txns_6m = rng.poisson(np.clip(0.6 + (700 - bureau_score) / 300, 0, None))
vintage_months = rng.integers(0, 180, N)  # credit history length
loan_amount = np.round(rng.lognormal(11.2, 0.5, N), -3)
employment_type = rng.choice(["Salaried", "Self-Employed", "Gig"], N, p=[0.55, 0.30, 0.15])
inquiries_3m = rng.poisson(1.3, N)

df = pd.DataFrame(dict(
    age=age, income=income, bureau_score=bureau_score, existing_loans=existing_loans,
    dti=dti, avg_bank_balance=avg_bank_balance, bounced_txns_6m=bounced_txns_6m,
    vintage_months=vintage_months, loan_amount=loan_amount,
    employment_type=employment_type, inquiries_3m=inquiries_3m,
))

# ---------------------------------------------------------------
# 2. Simulate ground-truth default outcome from a latent risk score
#    (nonlinear combination + noise, so no single feature perfectly separates)
# ---------------------------------------------------------------
emp_risk = df.employment_type.map({"Salaried": -0.15, "Self-Employed": 0.05, "Gig": 0.30}).values
logit = (
    -3.4
    - 0.017 * (df.bureau_score - 700)
    + 0.028 * (df.dti - 35)
    + 0.22 * df.bounced_txns_6m
    - 0.16 * df.existing_loans.clip(upper=4)
    - 0.004 * (df.vintage_months / 12)
    + 0.20 * df.inquiries_3m
    - 0.35 * np.log(df.avg_bank_balance / df.income.clip(lower=1))
    + emp_risk
    + rng.normal(0, 0.55, N)
)
p_default = 1 / (1 + np.exp(-logit))
df["default"] = rng.binomial(1, p_default)

print(f"Base default rate: {df.default.mean():.3%}")

# ---------------------------------------------------------------
# 3. Train / test split + feature engineering
# ---------------------------------------------------------------
df = pd.get_dummies(df, columns=["employment_type"], drop_first=True)
feature_cols = [c for c in df.columns if c != "default"]
df["income_to_loan"] = df.income * 12 / df.loan_amount
df["balance_to_income"] = df.avg_bank_balance / df.income.clip(lower=1)
feature_cols = [c for c in df.columns if c != "default"]

X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], df["default"], test_size=0.25, random_state=42, stratify=df["default"]
)

# ---------------------------------------------------------------
# 4. Candidate scorecards: Logistic Regression (baseline, explainable)
#    vs XGBoost (challenger)
# ---------------------------------------------------------------
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)
auc_logreg = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

xgb = XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=42
)
xgb.fit(X_train, y_train)
auc_xgb = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

print(f"Logistic Regression AUC: {auc_logreg:.4f}")
print(f"XGBoost AUC: {auc_xgb:.4f}")

# Feature importance (explainability proxy for the "explainable AI" requirement)
importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 5 risk drivers (XGBoost):")
print(importances.head(5))

# ---------------------------------------------------------------
# 5. Policy Rule Engine (hard cutoffs a real risk team would set)
# ---------------------------------------------------------------
def rule_engine_flag(row):
    if row.bureau_score < 650 or row.dti > 60 or row.bounced_txns_6m >= 4:
        return "Reject"
    if row.bureau_score < 700 or row.dti > 45 or row.bounced_txns_6m >= 2:
        return "Refer"
    return "Approve"

test_df = X_test.copy()
test_df["default"] = y_test.values
test_df["ml_score"] = xgb.predict_proba(X_test)[:, 1]
test_df["rule_decision"] = test_df.apply(rule_engine_flag, axis=1)

# ---------------------------------------------------------------
# 6. Hybrid decision strategy: rule engine + ML score override
#    - Rule "Reject" always stays Reject (compliance floor)
#    - Rule "Refer"/"Approve" get refined by ML score bands
# ---------------------------------------------------------------
def hybrid_decision(row):
    if row.rule_decision == "Reject":
        return "Reject"
    if row.ml_score >= 0.35:
        return "Reject"
    if row.ml_score >= 0.15:
        return "Refer"
    return "Approve"

test_df["hybrid_decision"] = test_df.apply(hybrid_decision, axis=1)

def strategy_metrics(decisions, defaults):
    approved = decisions == "Approve"
    approval_rate = approved.mean()
    npa_rate = defaults[approved].mean() if approved.sum() else float("nan")
    return approval_rate, npa_rate

rule_appr, rule_npa = strategy_metrics(test_df.rule_decision, test_df.default)
hybrid_appr, hybrid_npa = strategy_metrics(test_df.hybrid_decision, test_df.default)

print("\n--- Strategy comparison on held-out test set ---")
print(f"Rules-only   : approval rate {rule_appr:.2%}, NPA rate {rule_npa:.2%}")
print(f"Hybrid (ML)  : approval rate {hybrid_appr:.2%}, NPA rate {hybrid_npa:.2%}")
print(f"Approval-rate delta: {(hybrid_appr - rule_appr)*100:+.2f} pp")
print(f"NPA-rate delta:      {(hybrid_npa - rule_npa)*100:+.2f} pp")

# ---------------------------------------------------------------
# 7. Threshold simulation / "what-if" sweep for the approve cutoff
#    (mirrors a decision-engine simulation module)
# ---------------------------------------------------------------
print("\n--- What-if sweep: ML approve-threshold vs approval/NPA trade-off ---")
for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    appr = (test_df.rule_decision != "Reject") & (test_df.ml_score < t)
    ar = appr.mean()
    npa = test_df.default[appr].mean() if appr.sum() else float("nan")
    print(f"threshold={t:.2f} -> approval rate {ar:.2%}, NPA rate {npa:.2%}")
