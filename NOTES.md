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
