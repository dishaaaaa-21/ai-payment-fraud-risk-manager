"""
Time-Based Data Splitting
==========================
Implements chronological train/validation/test split based on the `step` column.

Split boundaries are data-driven from Phase 1A profiling:
  - Train:      steps 1-408   (5,987,417 txns, 4,589 fraud, 0.077%)
  - Validation: steps 409-557 (181,264 txns,   1,562 fraud, 0.862%)
  - Test:       steps 558-743 (193,939 txns,   2,062 fraud, 1.063%)

IMPORTANT: Fraud rate differs ~14x between train and test. This is an inherent
property of PaySim's temporal structure (later steps have fewer transactions
but similar fraud counts). Documented as a distribution-shift limitation.

Rules:
  - No random shuffling — chronological order preserved.
  - Model selection, hyperparameter tuning, calibration, and threshold
    selection use validation set ONLY.
  - Test set touched EXACTLY ONCE at the very end.
"""

import pandas as pd
from typing import Tuple

# Data-driven split boundaries from Phase 1A profiling
TRAIN_END_STEP = 408
VAL_END_STEP = 557
# Test: VAL_END_STEP+1 to max step (743)


def time_split(
    df: pd.DataFrame,
    train_end: int = TRAIN_END_STEP,
    val_end: int = VAL_END_STEP,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically by step.

    Returns:
        (train_df, val_df, test_df) — disjoint, covering all rows.
    """
    train = df[df["step"] <= train_end].copy()
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)].copy()
    test = df[df["step"] > val_end].copy()

    # Sanity checks
    total = len(train) + len(val) + len(test)
    assert total == len(df), f"Split lost rows: {total} != {len(df)}"
    assert train["step"].max() <= train_end
    assert val["step"].min() > train_end
    assert val["step"].max() <= val_end
    assert test["step"].min() > val_end

    return train, val, test


def print_split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Print split statistics."""
    print("\n--- Time-Based Split Summary ---")
    print(f"  {'Split':12s} {'Steps':>12s} {'Transactions':>14s} {'Fraud':>8s} {'FraudRate':>10s}")
    print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*8} {'-'*10}")

    for name, subset in [("Train", train), ("Validation", val), ("Test", test)]:
        f = subset["isFraud"].sum()
        r = f / len(subset) if len(subset) > 0 else 0
        s_min, s_max = subset["step"].min(), subset["step"].max()
        print(f"  {name:12s} {s_min:>5d}-{s_max:<5d}  {len(subset):>14,} {f:>8,} {r:>10.6f}")

    print(f"\n  Total: {len(train) + len(val) + len(test):,} transactions")
    train_fr = train["isFraud"].sum() / len(train)
    test_fr = test["isFraud"].sum() / len(test)
    print(f"  NOTE: Train fraud rate ({train_fr:.4%}) vs Test fraud rate ({test_fr:.4%})")
    print(f"  This {test_fr/train_fr:.0f}x shift is a documented PaySim limitation.")
