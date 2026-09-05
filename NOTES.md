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

### CRITICAL BUG FOUND: is_full_balance_transfer is a perfect fraud detector (dataset artifact)
- VERIFIED ON FULL DATASET (not just sample):
  - 8,018 full-balance-transfers exist. ALL 8,018 are fraud. Zero non-fraud.
  - 97.6% of ALL fraud (8,018/8,213) is a full-balance-transfer.
  - Only 195 fraud cases are NOT full-balance-transfers.
- This means PaySim's fraud injection mechanism IS "drain entire balance."
- Including this feature = model memorizes the simulation rule, not fraud patterns.
- **Decision**: INCLUDE in one model variant for comparison (to show the dataset's inherent structure),
  but PRIMARY model should EXCLUDE it to force learning from other signals.
  Otherwise we're just building a `if amount == balance: fraud` detector.
- This is honestly reported as a PaySim limitation.

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

---

## 2026-09-04 00:57 — Phase 1B Model Training Complete

### Results (Validation Set Only — Test UNTOUCHED)

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Baseline (rules-only) | N/A | 0.0352 | 0.6972 | 0.0670 | N/A |
| LogReg (Production) | 0.5629 | 0.5740 | 0.4219 | 0.4863 | 0.9866 |
| LogReg (Benchmark) | 0.9994 | 1.0000 | 0.9994 | 0.9997 | 0.9995 |
| **XGBoost (Production)** | **0.9915** | **0.9611** | **0.9334** | **0.9471** | **0.9999** |
| XGBoost (Benchmark) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Key observations:
- Naive baseline: high recall (70%) but terrible precision (3.5%). Rules-only catches fraud but generates ~30K false positives.
- LogReg Production: decent PR-AUC (0.56) but calibration is off (mean P(fraud)=0.088 vs actual 0.0086). Needs very high threshold (0.996) to get reasonable precision.
- **XGBoost Production: excellent. PR-AUC 0.9915, 96% precision at 93% recall.** Selected as primary.
- Benchmark models: near-perfect (as expected — `is_full_balance_transfer` is a perfect cheat code). XGBoost Benchmark converged in iteration 0 (!) — literally only needs one split.
- `amount_to_balance_ratio` dominates XGBoost Production feature importance (132K gain vs 4.8K for next feature). This confirms that fraud in PaySim is fundamentally about spending all your balance.
- Platt calibration works perfectly: post-calibration mean P(fraud)=0.008617 matches actual rate exactly.
- Benchmark model importance: `is_full_balance_transfer` has 3.6M gain vs 42K for `amount` — confirms the cheat-code nature.

### Approved decision: XGBoost Production (calibrated) as primary model.
Benchmark models retained as documented dataset-artifact reference only.

---

## 2026-09-05 12:19 — Starting Phases 3-8

Building behavioral engine, policy engine, risk decision fusion, synthetic scenario, AI investigation, and audit trail. All behavioral/policy evaluation will use the synthetic scenario, not PaySim metrics.
