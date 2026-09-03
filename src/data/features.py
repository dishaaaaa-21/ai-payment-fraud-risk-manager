"""
Feature Engineering Pipeline — Leakage-Safe
=============================================
Builds two feature variants:
  - PRODUCTION: excludes is_full_balance_transfer (primary model)
  - BENCHMARK: includes is_full_balance_transfer (dataset reference)

Leakage rules enforced:
  - EXCLUDED: newbalanceOrig, newbalanceDest, isFlaggedFraud (post-transaction / second label)
  - INCLUDED: only pre-transaction attributes
  - No behavioral features here (those belong in the behavioral engine, and PaySim
    cannot support them — 99.9% of customers have exactly 1 transaction)

See NOTES.md and Phase 1A report for full leakage analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple

# ============================================================================
# LEAKAGE-SAFE COLUMNS
# ============================================================================

# Columns we NEVER use as features (post-transaction or labels)
EXCLUDED_COLUMNS = [
    "newbalanceOrig",   # Post-transaction — leaks fraud label (94.7% vs 19.8% match rate)
    "newbalanceDest",   # Post-transaction
    "isFlaggedFraud",   # Effectively a second label (16 cases, all fraud, 100% precision)
    "isFraud",          # Target variable
]

# Identifier columns — used for grouping, not as direct features
IDENTIFIER_COLUMNS = [
    "nameOrig",         # Customer ID — for behavioral grouping only
    "nameDest",         # Counterparty ID — for behavioral grouping only
]

# Transaction types in PaySim
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Production feature list (12 features) — the primary model
PRODUCTION_FEATURES = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_balance_ratio",
    "step",
    "step_mod_24",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

# Benchmark adds is_full_balance_transfer (13 features)
# NOTE: This feature is a near-perfect fraud detector in PaySim because
# PaySim's fraud mechanism IS "drain entire balance." 8,018/8,018 full-balance
# transfers are fraud. Including it gives the model a trivial shortcut.
# It is included ONLY as a reference benchmark, not as the primary model.
BENCHMARK_FEATURES = PRODUCTION_FEATURES + ["is_full_balance_transfer"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered features from raw PaySim data.

    All features here are pre-transaction (available at scoring time).
    No post-transaction columns are touched.

    Returns a new DataFrame with all features added (original columns preserved).
    """
    out = df.copy()

    # --- Derived numeric features ---

    # Log-scaled amount: reduces skew, helps linear models
    out["log_amount"] = np.log1p(out["amount"])

    # Ratio of amount to sender's balance: how much of their balance are they spending?
    # +1 to avoid division by zero when balance is 0
    out["amount_to_balance_ratio"] = out["amount"] / (out["oldbalanceOrg"] + 1.0)

    # Hour-of-day proxy (step is simulated hours, step % 24 gives rough hour)
    out["step_mod_24"] = out["step"] % 24

    # Full balance transfer flag: amount == oldbalanceOrg AND balance > 0
    # WARNING: This is a dataset artifact — 100% of these are fraud in PaySim.
    # See Phase 1A report for details. Only used in BENCHMARK variant.
    out["is_full_balance_transfer"] = (
        (out["amount"] == out["oldbalanceOrg"]) & (out["oldbalanceOrg"] > 0)
    ).astype(np.int8)

    # --- One-hot encode transaction type ---
    for t in TRANSACTION_TYPES:
        out[f"type_{t}"] = (out["type"] == t).astype(np.int8)

    return out


def get_feature_matrix(
    df: pd.DataFrame,
    variant: str = "production"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target y from a DataFrame with features built.

    Args:
        df: DataFrame with features already built (via build_features)
        variant: "production" (12 features) or "benchmark" (13 features)

    Returns:
        X: feature matrix
        y: target series (isFraud)
    """
    if variant == "production":
        feature_cols = PRODUCTION_FEATURES
    elif variant == "benchmark":
        feature_cols = BENCHMARK_FEATURES
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'production' or 'benchmark'.")

    # Validate all expected columns exist
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Did you run build_features()?")

    X = df[feature_cols].copy()
    y = df["isFraud"].copy()

    return X, y


def validate_no_leakage(df: pd.DataFrame, feature_cols: list) -> bool:
    """
    Programmatic check: ensure no leaked columns are in the feature set.
    """
    leaked = set(feature_cols) & set(EXCLUDED_COLUMNS)
    if leaked:
        raise ValueError(f"LEAKAGE DETECTED: {leaked} are in the feature set!")

    identifier_in_features = set(feature_cols) & set(IDENTIFIER_COLUMNS)
    if identifier_in_features:
        raise ValueError(
            f"IDENTIFIER LEAKAGE: {identifier_in_features} used as direct features! "
            f"Use for grouping only."
        )

    print(f"  Leakage check PASSED: {len(feature_cols)} features, none leaked.")
    return True
