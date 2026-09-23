# credit-risk-hybrid-decision-engine
# Risk-Based Underwriting & Hybrid Decision Engine

A simulated end-to-end credit risk scorecard and rule-engine + ML hybrid decision
system for loan underwriting, built to mirror how a modern lending decision
platform (bureau intelligence + rule engine + ML scoring) evaluates and
approves/rejects/refers loan applicants.

## Overview

This project simulates a realistic loan-applicant population with bureau,
financial, and transactional features, trains a machine learning credit
scorecard, and blends it with a policy rule layer into a 3-way decision
(**Approve / Refer / Reject**). It then backtests the hybrid strategy against
a rules-only baseline to quantify the trade-off between approval rate and
portfolio risk (NPA).

## Problem Statement

Traditional underwriting relies purely on hard policy cutoffs (e.g. minimum
bureau score, maximum DTI), which are simple and explainable but leave
approval volume on the table by rejecting creditworthy borrowers who fall
just outside rigid thresholds. This project asks: **can an ML scorecard,
layered on top of — not instead of — the existing rule engine, safely expand
approvals without materially increasing default risk?**

## Data

Since no real bureau/banking data was available, a synthetic dataset of
25,000 applicants was generated with realistic feature distributions and a
non-linear latent default-risk function (with noise, so no single feature
perfectly separates good/bad):

| Feature | Description |
|---|---|
| `bureau_score` | Simulated credit bureau score (300–900) |
| `income` | Monthly income (INR) |
| `dti` | Debt-to-income ratio (%) |
| `existing_loans` | Number of active loans |
| `avg_bank_balance` | Average bank account balance |
| `bounced_txns_6m` | Bounced transactions in last 6 months |
| `vintage_months` | Length of credit history |
| `inquiries_3m` | Credit inquiries in last 3 months |
| `employment_type` | Salaried / Self-Employed / Gig |
| `loan_amount` | Requested loan amount |

Base simulated default rate: **15.6%**

## Methodology

1. **Feature engineering** — derived ratios (`income_to_loan`, `balance_to_income`) added to raw applicant features.
2. **Scorecard modeling** — trained and compared:
   - Logistic Regression (explainable baseline)
   - XGBoost (challenger model)
3. **Policy rule engine** — hard-coded cutoffs on bureau score, DTI, and bounced transactions (Approve / Refer / Reject), representing a typical existing risk policy.
4. **Hybrid decision engine** — rule engine's "Reject" is a compliance floor (never overridden); "Refer"/"Approve" applicants are re-scored by the ML model into refined bands.
5. **Threshold simulation** — swept the ML approve-cutoff across 6 values to map the full approval-rate vs. NPA-rate trade-off curve.

## Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| Test AUC | 0.800 | **0.834** |

**Top 5 risk drivers (XGBoost feature importance):** bureau score, bounced transactions (6m), DTI, credit inquiries (3m), employment type (salaried).

| Strategy | Approval Rate | NPA Rate |
|---|---|---|
| Rules-only (baseline) | 30.5% | 3.0% |
| Hybrid (rules + ML) | **58.6%** | 4.6% |

**Net impact:** +28.1 pp approval rate for a +1.6 pp increase in NPA — a materially favorable trade-off for portfolio growth.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost
python project1_credit_risk_engine.py
```

## Future Improvements

- Replace synthetic data with real/anonymized bureau data (e.g. Kaggle credit risk datasets) for external validity
- Add SHAP-based explainability per applicant decision
- Extend the rule engine to a configurable JSON/YAML policy spec
- Add cost-sensitive optimization (yield vs. loss) to choose the threshold automatically rather than manually
