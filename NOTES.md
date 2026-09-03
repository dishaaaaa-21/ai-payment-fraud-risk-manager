# BUILD LOG — AI Risk Manager for Behavioral Payment Fraud

This is a raw build log. Bugs, wrong assumptions, and surprises are logged as they happen.

---

## 2026-09-04 00:18 — Project Start

- **Workspace was completely empty.** No prior code, no dataset.
- PaySim CSV is NOT on disk anywhere — searched the entire user home directory. Will need to download it.
- Python 3.10.0 available. pandas, numpy, scikit-learn, flask, fastapi already installed.
- XGBoost and LightGBM are NOT installed — intentionally deferring that choice.
- No Kaggle CLI installed. Dataset acquisition will need an alternative approach.
- Git initialized fresh.

### Decision: Phase 0 + 1A only
Following the user's instruction to validate data and assumptions before building anything. Previous implementation plan was too eager — jumping to model training without profiling is risky.

---

## 2026-09-04 00:26 — Dataset Acquired

- PaySim dataset downloaded from Kaggle public API (no auth needed for this dataset).
- URL: `https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1`
- Downloaded as ZIP (177.8 MB), extracted to 470.7 MB CSV.
- Original filename: `PS_20174392719_1491204439457_log.csv`, renamed to `paysim.csv`.
- Schema validated: all 11 expected columns present.

---

## 2026-09-04 00:27 — Phase 1A Profiling: CRITICAL FINDINGS

### Dataset Stats
- 6,362,620 rows, 11 columns, zero missing values, zero duplicates.
- Steps 1-743 (743 unique hours, ~30 simulated days).
- 5 transaction types: CASH_OUT (35.2%), PAYMENT (33.8%), CASH_IN (22.0%), TRANSFER (8.4%), DEBIT (0.7%).
- Fraud: 8,213 cases (0.1291%). Extremely imbalanced.
- `isFlaggedFraud`: only 16 cases flagged (0.19% recall on actual fraud — essentially useless).

### CRITICAL BUG FOUND: is_full_balance_transfer has 100% fraud rate
- In the 10K sample: `is_full_balance_transfer=True` has fraud rate 1.0, False has fraud rate 0.0.
- This is suspiciously clean — it suggests PaySim's fraud mechanism IS "drain entire balance."
- If we include this feature, the model trivially learns a perfect rule. That's not learning fraud patterns, that's memorizing the simulation's fraud-injection rule.
- **Decision needed**: include but acknowledge this is a dataset artifact? Or exclude to force the model to learn from other signals? Need to check on the full dataset, not just a sample.

### CRITICAL FINDING: Behavioral features are DEAD ON ARRIVAL for PaySim
- 99.9% of customers (6,344,009 of 6,353,307) have exactly ONE transaction.
- Only 9,283 customers have 2 transactions. Only 15 have 3. NONE have more than 3.
- Max transactions per customer: 3. Mean: 1.00.
- 100% of sender-receiver pairs are unique. Zero repeat counterparties.
- Fraud customers: 99.7% have exactly 1 transaction.

**This means:**
1. Transaction velocity is meaningless — almost no customer has repeat transactions.
2. Counterparty repetition is impossible — every pair is unique.
3. Amount escalation cannot be computed — no history to compare against.
4. Customer baseline deviation is undefined — no baseline exists.
5. The entire behavioral engine as architected CANNOT operate on PaySim data.

This is the single most important finding. The behavioral engine is architecturally critical to our system, but PaySim cannot feed it. We need to:
- Build the engine anyway (for the synthetic scenario and production readiness).
- Acknowledge it adds zero value for PaySim evaluation.
- The synthetic scenario becomes the ONLY place to demonstrate behavioral detection.

### Fraud Distribution
- Fraud ONLY occurs in CASH_OUT (4,116) and TRANSFER (4,097). Zero fraud in CASH_IN, PAYMENT, DEBIT.
- Fraud is distributed roughly uniformly across steps (50.5% in first half, 49.5% in second).
- Fraud rate varies by bucket (0.06% to 2.2%) but fraud cases exist in all time periods.

### Leakage Analysis
- `newbalanceOrig`: LEAKS. Fraud match rate 94.7% vs non-fraud 19.8% for (old - amount == new).
- `newbalanceDest`: POST-TRANSACTION. Excluded.
- `isFlaggedFraud`: Effectively a second label. Only 16 cases, all fraud, all TRANSFER type with amount >353K. Excluded.
- `oldbalanceDest`: Pre-transaction but sender wouldn't know recipient's balance in production. Including with caveat.

### Proposed Time Split
- Train: steps 1-408 (5,987,417 txns, 4,589 fraud, 0.077%)
- Validation: steps 409-557 (181,264 txns, 1,562 fraud, 0.862%)
- Test: steps 558-743 (193,939 txns, 2,062 fraud, 1.063%)
- NOTE: Fraud RATE differs significantly across splits. Later steps have higher fraud concentration. This is a temporal pattern in the simulation — later "hours" see more fraud per transaction. This is realistic (fraud campaigns intensify) but means val/test fraud rates are 10x higher than train. The model will see a distribution shift.

### Redundancy Analysis
- `balance_change_expected` (oldbalanceOrg - amount): redundant linear combination, DO NOT INCLUDE.
- `amount_to_balance_ratio` vs `log_amount`: correlation 0.26, capture different info, INCLUDE BOTH.

### Step Bucket Anomaly
- Steps 50 and 725 show 100% fraud rate (336 and 230 txns respectively, ALL fraud).
- These are likely simulation artifacts — entire step-buckets with nothing but fraud.
- This is a dataset quirk, not a real-world pattern.

---

## 2026-09-04 00:28 — Architecture Impact Assessment

The behavioral engine cannot be validated on PaySim. This doesn't mean we shouldn't build it, but it changes the evaluation strategy:
- ML model evaluation: PaySim metrics (real).
- Behavioral engine evaluation: Synthetic scenario only (clearly labeled).
- Policy engine evaluation: Can fire on PaySim data (amount thresholds, full-balance-transfer detection) but counterparty-based rules can't fire (no repeat counterparties).
- Risk fusion evaluation: Only the ML component contributes signal on PaySim data.

This is an honest limitation, not a failure. Document it clearly.
