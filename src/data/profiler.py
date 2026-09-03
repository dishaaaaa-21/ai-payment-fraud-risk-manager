"""
Phase 1A: Data Profiling and Validation
========================================
Comprehensive profiling of the PaySim dataset before any model building.
Covers: schema validation, distributions, fraud analysis, temporal analysis,
leakage analysis, feature availability, and PaySim limitations.

This script produces a structured report — all findings must be reviewed
before proceeding to model training.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from collections import OrderedDict

# ============================================================================
# CONFIGURATION
# ============================================================================
# Go up from src/data/ to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(_PROJECT_ROOT, "dataset", "paysim.csv")

# ============================================================================
# A. DATASET PROFILE
# ============================================================================

def profile_dataset(df: pd.DataFrame) -> dict:
    """Basic dataset profile: shape, types, missing values, duplicates."""
    print("\n" + "=" * 70)
    print("A. DATASET PROFILE")
    print("=" * 70)

    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
    }

    print(f"\nTotal rows:    {stats['total_rows']:,}")
    print(f"Total columns: {stats['total_columns']}")
    print(f"Columns:       {stats['column_names']}")

    # Data types
    print("\n--- Data Types ---")
    for col in df.columns:
        print(f"  {col:20s} {str(df[col].dtype):10s}  (nunique={df[col].nunique():,})")

    # Missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values in any column.")
    else:
        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count:,} ({100*count/len(df):.2f}%)")

    # Duplicate rows
    dup_count = df.duplicated().sum()
    print(f"\n--- Duplicate Rows ---")
    print(f"  Exact duplicates: {dup_count:,} ({100*dup_count/len(df):.4f}%)")

    # Step range
    print(f"\n--- Step (Time Proxy) ---")
    print(f"  Min step: {df['step'].min()}")
    print(f"  Max step: {df['step'].max()}")
    print(f"  Unique steps: {df['step'].nunique()}")

    # Transaction type distribution
    print(f"\n--- Transaction Type Distribution ---")
    type_counts = df['type'].value_counts()
    for t, c in type_counts.items():
        print(f"  {t:12s} {c:>10,}  ({100*c/len(df):6.2f}%)")

    # Fraud overview
    fraud_count = df['isFraud'].sum()
    fraud_rate = fraud_count / len(df)
    flagged_count = df['isFlaggedFraud'].sum()

    print(f"\n--- Fraud Overview ---")
    print(f"  Total fraud cases:     {fraud_count:,}")
    print(f"  Overall fraud rate:    {fraud_rate:.6f} ({100*fraud_rate:.4f}%)")
    print(f"  isFlaggedFraud count:  {flagged_count:,}")
    if fraud_count > 0:
        flagged_recall = flagged_count / fraud_count
        print(f"  isFlaggedFraud recall: {flagged_recall:.4f} ({100*flagged_recall:.2f}% of actual fraud)")

    stats["fraud_count"] = fraud_count
    stats["fraud_rate"] = fraud_rate
    stats["flagged_count"] = flagged_count

    return stats


# ============================================================================
# B. FRAUD DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_fraud_distribution(df: pd.DataFrame) -> dict:
    """Detailed fraud distribution by type, step, and time periods."""
    print("\n" + "=" * 70)
    print("B. FRAUD DISTRIBUTION ANALYSIS")
    print("=" * 70)

    findings = {}

    # B.1: Fraud by transaction type
    print("\n--- B.1: Fraud Count and Rate by Transaction Type ---")
    print(f"  {'Type':12s} {'Total':>10s} {'Fraud':>8s} {'FraudRate':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*10}")

    type_fraud = df.groupby('type').agg(
        total=('isFraud', 'count'),
        fraud=('isFraud', 'sum')
    )
    type_fraud['fraud_rate'] = type_fraud['fraud'] / type_fraud['total']

    for t, row in type_fraud.iterrows():
        print(f"  {t:12s} {row['total']:10,.0f} {row['fraud']:8,.0f} {row['fraud_rate']:10.6f}")

    fraud_types = type_fraud[type_fraud['fraud'] > 0].index.tolist()
    non_fraud_types = type_fraud[type_fraud['fraud'] == 0].index.tolist()
    print(f"\n  Types with fraud: {fraud_types}")
    print(f"  Types with NO fraud: {non_fraud_types}")
    findings["fraud_types"] = fraud_types
    findings["non_fraud_types"] = non_fraud_types

    # B.2: Fraud distribution over step (time)
    print("\n--- B.2: Fraud Distribution Over Time (Step) ---")
    # Divide steps into buckets
    max_step = df['step'].max()
    n_buckets = 10
    bucket_size = max_step // n_buckets + 1
    df_temp = df.copy()
    df_temp['step_bucket'] = (df_temp['step'] // bucket_size) * bucket_size

    step_fraud = df_temp.groupby('step_bucket').agg(
        total=('isFraud', 'count'),
        fraud=('isFraud', 'sum')
    )
    step_fraud['fraud_rate'] = step_fraud['fraud'] / step_fraud['total']

    print(f"  {'StepRange':>15s} {'Total':>10s} {'Fraud':>8s} {'FraudRate':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10}")
    for bucket, row in step_fraud.iterrows():
        end = min(bucket + bucket_size - 1, max_step)
        print(f"  {bucket:>6.0f}-{end:<6.0f}  {row['total']:10,.0f} {row['fraud']:8,.0f} {row['fraud_rate']:10.6f}")

    # Check if fraud is concentrated
    fraud_by_step = df[df['isFraud'] == 1]['step']
    print(f"\n  Fraud step statistics:")
    print(f"    Mean step:   {fraud_by_step.mean():.1f}")
    print(f"    Median step: {fraud_by_step.median():.1f}")
    print(f"    Std step:    {fraud_by_step.std():.1f}")
    print(f"    Min step:    {fraud_by_step.min()}")
    print(f"    Max step:    {fraud_by_step.max()}")

    # Is fraud uniformly distributed or concentrated?
    first_half = (fraud_by_step <= max_step / 2).sum()
    second_half = (fraud_by_step > max_step / 2).sum()
    total_fraud = len(fraud_by_step)
    print(f"\n  Fraud in first half (steps 1-{max_step//2}):  {first_half:,} ({100*first_half/total_fraud:.1f}%)")
    print(f"  Fraud in second half (steps {max_step//2+1}-{max_step}): {second_half:,} ({100*second_half/total_fraud:.1f}%)")

    findings["fraud_step_stats"] = {
        "mean": fraud_by_step.mean(),
        "median": fraud_by_step.median(),
        "std": fraud_by_step.std(),
        "min": fraud_by_step.min(),
        "max": fraud_by_step.max(),
    }

    return findings


# ============================================================================
# C. TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================================

def propose_temporal_split(df: pd.DataFrame) -> dict:
    """Data-driven temporal split based on actual fraud distribution."""
    print("\n" + "=" * 70)
    print("C. TIME-BASED TRAIN / VALIDATION / TEST SPLIT ANALYSIS")
    print("=" * 70)

    max_step = df['step'].max()
    total_fraud = df['isFraud'].sum()
    total_rows = len(df)

    # First, examine fraud density across steps in finer granularity
    print("\n--- Step-by-step fraud density (25-step buckets) ---")
    bucket_size = 25
    df_temp = df.copy()
    df_temp['step_bucket'] = (df_temp['step'] // bucket_size) * bucket_size

    step_detail = df_temp.groupby('step_bucket').agg(
        total=('isFraud', 'count'),
        fraud=('isFraud', 'sum')
    )
    step_detail['fraud_rate'] = step_detail['fraud'] / step_detail['total']
    step_detail['cum_fraud'] = step_detail['fraud'].cumsum()
    step_detail['cum_fraud_pct'] = step_detail['cum_fraud'] / total_fraud

    print(f"  {'StepBucket':>10s} {'Total':>10s} {'Fraud':>7s} {'Rate':>10s} {'CumFraud%':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*7} {'-'*10} {'-'*10}")
    for bucket, row in step_detail.iterrows():
        print(f"  {bucket:>10.0f} {row['total']:10,.0f} {row['fraud']:7,.0f} {row['fraud_rate']:10.6f} {row['cum_fraud_pct']:10.4f}")

    # Propose split: aim for ~70% train, ~15% val, ~15% test by transactions
    # But constrained by having enough fraud in each split
    print(f"\n--- Evaluating candidate splits ---")
    print(f"  Total steps: 1 to {max_step}")
    print(f"  Total transactions: {total_rows:,}")
    print(f"  Total fraud: {total_fraud:,}")

    # Try multiple split points
    candidates = [
        # (train_end, val_end)  — test is val_end+1 to max_step
        (int(max_step * 0.60), int(max_step * 0.80)),
        (int(max_step * 0.65), int(max_step * 0.82)),
        (int(max_step * 0.70), int(max_step * 0.85)),
        (400, 550),  # Previously proposed
        (int(max_step * 0.55), int(max_step * 0.75)),
    ]

    print(f"\n  {'TrainEnd':>8s} {'ValEnd':>7s} {'Train_N':>10s} {'Val_N':>10s} {'Test_N':>10s} "
          f"{'Train_F':>8s} {'Val_F':>7s} {'Test_F':>7s} "
          f"{'Train_FR':>9s} {'Val_FR':>9s} {'Test_FR':>9s}")
    print(f"  {'-'*8} {'-'*7} {'-'*10} {'-'*10} {'-'*10} "
          f"{'-'*8} {'-'*7} {'-'*7} "
          f"{'-'*9} {'-'*9} {'-'*9}")

    best_split = None
    best_score = -1

    for train_end, val_end in candidates:
        train = df[df['step'] <= train_end]
        val = df[(df['step'] > train_end) & (df['step'] <= val_end)]
        test = df[df['step'] > val_end]

        train_f = train['isFraud'].sum()
        val_f = val['isFraud'].sum()
        test_f = test['isFraud'].sum()

        train_fr = train_f / len(train) if len(train) > 0 else 0
        val_fr = val_f / len(val) if len(val) > 0 else 0
        test_fr = test_f / len(test) if len(test) > 0 else 0

        print(f"  {train_end:>8d} {val_end:>7d} {len(train):>10,} {len(val):>10,} {len(test):>10,} "
              f"{train_f:>8,} {val_f:>7,} {test_f:>7,} "
              f"{train_fr:>9.6f} {val_fr:>9.6f} {test_fr:>9.6f}")

        # Score: all splits must have fraud, prefer more balanced fraud rates
        if val_f >= 100 and test_f >= 100:
            # Prefer splits where val and test have reasonable fraud counts
            balance = 1.0 / (1.0 + abs(val_fr - test_fr) / max(val_fr, test_fr, 1e-9))
            score = min(val_f, test_f) * balance
            if score > best_score:
                best_score = score
                best_split = (train_end, val_end)

    if best_split is None:
        print("\n  WARNING: No candidate split has >=100 fraud cases in both val and test.")
        print("  Using 70/15/15 by step count as fallback.")
        best_split = (int(max_step * 0.70), int(max_step * 0.85))

    train_end, val_end = best_split
    train = df[df['step'] <= train_end]
    val = df[(df['step'] > train_end) & (df['step'] <= val_end)]
    test = df[df['step'] > val_end]

    print(f"\n--- RECOMMENDED SPLIT ---")
    print(f"  {'Split':12s} {'StepRange':>15s} {'Transactions':>14s} {'FraudCases':>12s} {'FraudRate':>12s}")
    print(f"  {'-'*12} {'-'*15} {'-'*14} {'-'*12} {'-'*12}")

    for name, subset, start, end in [
        ("Train", train, 1, train_end),
        ("Validation", val, train_end + 1, val_end),
        ("Test", test, val_end + 1, max_step),
    ]:
        f_count = subset['isFraud'].sum()
        f_rate = f_count / len(subset) if len(subset) > 0 else 0
        print(f"  {name:12s} {start:>6d}-{end:<6d}  {len(subset):>14,} {f_count:>12,} {f_rate:>12.6f}")

    split_info = {
        "train_end_step": train_end,
        "val_end_step": val_end,
        "max_step": max_step,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "train_fraud": int(train['isFraud'].sum()),
        "val_fraud": int(val['isFraud'].sum()),
        "test_fraud": int(test['isFraud'].sum()),
    }

    return split_info


# ============================================================================
# D. MANDATORY DATA LEAKAGE ANALYSIS
# ============================================================================

def leakage_analysis(df: pd.DataFrame) -> dict:
    """Analyze every PaySim column for leakage risk."""
    print("\n" + "=" * 70)
    print("D. MANDATORY DATA LEAKAGE ANALYSIS")
    print("=" * 70)

    columns_analysis = OrderedDict()

    # step
    columns_analysis["step"] = {
        "description": "Simulated hour of the simulation (1 step = 1 hour). "
                       "Max ~743 steps = ~30 simulated days.",
        "realtime_available": "YES — transaction timestamp is always known at scoring time.",
        "leakage_risk": "LOW — time of transaction is a legitimate feature. However, "
                        "step is a simulation artifact, not a real timestamp.",
        "classification": "SAFE FOR REAL-TIME MODEL (with caveats — it's a simulation time proxy, "
                          "not a real-world time feature)",
        "recommendation": "INCLUDE as feature. In production, this would be hour-of-day, day-of-week, etc."
    }

    # type
    columns_analysis["type"] = {
        "description": "Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER.",
        "realtime_available": "YES — transaction type is known when the transaction is initiated.",
        "leakage_risk": "NONE — this is a pre-transaction attribute.",
        "classification": "SAFE FOR REAL-TIME MODEL",
        "recommendation": "INCLUDE (one-hot or label encoded)."
    }

    # amount
    columns_analysis["amount"] = {
        "description": "Transaction amount in local currency.",
        "realtime_available": "YES — amount is specified by the initiator.",
        "leakage_risk": "NONE — this is a pre-transaction attribute.",
        "classification": "SAFE FOR REAL-TIME MODEL",
        "recommendation": "INCLUDE as primary feature."
    }

    # nameOrig
    columns_analysis["nameOrig"] = {
        "description": "Originator account identifier (e.g., C1234567890).",
        "realtime_available": "YES — customer ID is known at transaction time.",
        "leakage_risk": "NONE for identification. HIGH if used directly as a feature "
                        "(model would memorize specific customers).",
        "classification": "IDENTIFIER / SPECIAL HANDLING",
        "recommendation": "DO NOT use as a direct model feature. USE for grouping to compute "
                          "behavioral features (velocity, history, etc.)."
    }

    # oldbalanceOrg
    columns_analysis["oldbalanceOrg"] = {
        "description": "Origin account balance BEFORE the transaction.",
        "realtime_available": "YES — account balance is queryable before executing a transaction.",
        "leakage_risk": "LOW — this is pre-transaction state.",
        "classification": "SAFE FOR REAL-TIME MODEL",
        "recommendation": "INCLUDE. Useful for amount-to-balance ratio."
    }

    # newbalanceOrig
    print("\n  CRITICAL LEAKAGE CHECK: newbalanceOrig")
    # Check if newbalanceOrig = oldbalanceOrg - amount for non-fraud
    sample = df.sample(min(10000, len(df)), random_state=42)
    expected_new = sample['oldbalanceOrg'] - sample['amount']
    actual_new = sample['newbalanceOrig']
    match_rate = (np.abs(expected_new - actual_new) < 0.01).mean()
    fraud_sample = sample[sample['isFraud'] == 1]
    if len(fraud_sample) > 0:
        fraud_expected = fraud_sample['oldbalanceOrg'] - fraud_sample['amount']
        fraud_actual = fraud_sample['newbalanceOrig']
        fraud_match = (np.abs(fraud_expected - fraud_actual) < 0.01).mean()
    else:
        fraud_match = float('nan')

    non_fraud_sample = sample[sample['isFraud'] == 0]
    nf_expected = non_fraud_sample['oldbalanceOrg'] - non_fraud_sample['amount']
    nf_actual = non_fraud_sample['newbalanceOrig']
    nf_match = (np.abs(nf_expected - nf_actual) < 0.01).mean()

    print(f"    newbalanceOrig == oldbalanceOrg - amount ?")
    print(f"      Overall match rate:   {match_rate:.4f}")
    print(f"      Non-fraud match rate: {nf_match:.4f}")
    print(f"      Fraud match rate:     {fraud_match:.4f}")
    print(f"    If fraud transactions show different patterns in newbalanceOrig vs expected,")
    print(f"    then newbalanceOrig LEAKS the fraud label.")

    columns_analysis["newbalanceOrig"] = {
        "description": "Origin account balance AFTER the transaction.",
        "realtime_available": "NO — this is only known after the transaction executes.",
        "leakage_risk": f"HIGH — post-transaction value. Match rate with expected (old-amount): "
                        f"overall={match_rate:.4f}, non-fraud={nf_match:.4f}, fraud={fraud_match:.4f}. "
                        f"Discrepancy between fraud and non-fraud indicates this column encodes "
                        f"information about the transaction outcome.",
        "classification": "POST-TRANSACTION / UNSAFE",
        "recommendation": "EXCLUDE from model features."
    }

    # nameDest
    columns_analysis["nameDest"] = {
        "description": "Destination account identifier.",
        "realtime_available": "YES — recipient is known at transaction time.",
        "leakage_risk": "NONE for identification. HIGH if used directly as feature.",
        "classification": "IDENTIFIER / SPECIAL HANDLING",
        "recommendation": "DO NOT use as direct feature. USE for counterparty analysis "
                          "(repetition, new counterparty detection, etc.)."
    }

    # oldbalanceDest
    print("\n  CRITICAL LEAKAGE CHECK: oldbalanceDest")
    # In a real payment system, the sender may NOT know the recipient's balance
    # But it IS pre-transaction data (exists before the txn executes)
    dest_zero = (df['oldbalanceDest'] == 0).mean()
    fraud_dest_zero = (df[df['isFraud'] == 1]['oldbalanceDest'] == 0).mean()
    non_fraud_dest_zero = (df[df['isFraud'] == 0]['oldbalanceDest'] == 0).mean()
    print(f"    oldbalanceDest == 0 rates:")
    print(f"      Overall:   {dest_zero:.4f}")
    print(f"      Fraud:     {fraud_dest_zero:.4f}")
    print(f"      Non-fraud: {non_fraud_dest_zero:.4f}")

    columns_analysis["oldbalanceDest"] = {
        "description": "Destination account balance BEFORE the transaction.",
        "realtime_available": "QUESTIONABLE — in real payment systems, the sender typically "
                              "does NOT have access to the recipient's balance. In PaySim simulation, "
                              "this is available.",
        "leakage_risk": f"MEDIUM — pre-transaction data but may not be realistically available. "
                        f"Zero-balance rates differ: fraud={fraud_dest_zero:.4f} vs "
                        f"non-fraud={non_fraud_dest_zero:.4f}.",
        "classification": "SAFE FOR REAL-TIME MODEL (with caveat: not realistic in production)",
        "recommendation": "INCLUDE with documentation caveat. In a real system, this would be "
                          "replaced with merchant/recipient risk score from internal data."
    }

    # newbalanceDest
    print("\n  CRITICAL LEAKAGE CHECK: newbalanceDest")
    dest_sample = sample.copy()
    dest_expected = dest_sample['oldbalanceDest'] + dest_sample['amount']
    dest_actual = dest_sample['newbalanceDest']
    dest_match = (np.abs(dest_expected - dest_actual) < 0.01).mean()

    columns_analysis["newbalanceDest"] = {
        "description": "Destination account balance AFTER the transaction.",
        "realtime_available": "NO — post-transaction value.",
        "leakage_risk": "HIGH — post-transaction. Same issues as newbalanceOrig.",
        "classification": "POST-TRANSACTION / UNSAFE",
        "recommendation": "EXCLUDE from model features."
    }

    # isFraud
    columns_analysis["isFraud"] = {
        "description": "Ground truth fraud label. 1 = fraudulent, 0 = legitimate.",
        "realtime_available": "NO — this is the label we are trying to predict.",
        "leakage_risk": "N/A — this IS the target variable.",
        "classification": "LABEL / TARGET",
        "recommendation": "USE as target variable only. Never as input feature."
    }

    # isFlaggedFraud
    print("\n  CRITICAL LEAKAGE CHECK: isFlaggedFraud")
    flagged = df['isFlaggedFraud'].sum()
    flagged_and_fraud = ((df['isFlaggedFraud'] == 1) & (df['isFraud'] == 1)).sum()
    flagged_not_fraud = ((df['isFlaggedFraud'] == 1) & (df['isFraud'] == 0)).sum()
    print(f"    isFlaggedFraud == 1:  {flagged:,}")
    print(f"    Flagged AND fraud:    {flagged_and_fraud:,}")
    print(f"    Flagged NOT fraud:    {flagged_not_fraud:,}")
    if flagged > 0:
        print(f"    Precision of flag:    {flagged_and_fraud/flagged:.4f}")

    # Check what triggers isFlaggedFraud
    if flagged > 0:
        flagged_df = df[df['isFlaggedFraud'] == 1]
        print(f"    Flagged transaction types: {flagged_df['type'].unique().tolist()}")
        print(f"    Flagged amount range: {flagged_df['amount'].min():,.2f} to {flagged_df['amount'].max():,.2f}")
        print(f"    Flagged amount mean:  {flagged_df['amount'].mean():,.2f}")

    columns_analysis["isFlaggedFraud"] = {
        "description": "PaySim's built-in fraud flagging mechanism. Flags transactions over "
                       "200,000 that attempt to transfer the entire origin balance.",
        "realtime_available": "AMBIGUOUS — in PaySim, this is computed by the simulation engine "
                              "during transaction processing. It acts as a post-hoc label.",
        "leakage_risk": f"HIGH — this is essentially another label derived from the simulation. "
                        f"Only {flagged:,} transactions flagged, {flagged_and_fraud:,} of which "
                        f"are actual fraud. Using it as a feature would be circular.",
        "classification": "POST-TRANSACTION / UNSAFE (effectively a second label)",
        "recommendation": "EXCLUDE from model features. Could be used as a baseline comparator "
                          "(how well does PaySim's own flag perform vs our model?)."
    }

    # Print summary table
    print("\n--- LEAKAGE ANALYSIS SUMMARY ---")
    print(f"  {'Column':20s} {'Classification':40s} {'Decision':10s}")
    print(f"  {'-'*20} {'-'*40} {'-'*10}")
    for col, info in columns_analysis.items():
        decision = "INCLUDE" if "SAFE" in info["classification"] else \
                   "EXCLUDE" if "UNSAFE" in info["classification"] or "LABEL" in info["classification"] else \
                   "SPECIAL"
        print(f"  {col:20s} {info['classification']:40s} {decision:10s}")

    return columns_analysis


# ============================================================================
# E. FEATURE AVAILABILITY ANALYSIS
# ============================================================================

def feature_availability_analysis(df: pd.DataFrame) -> dict:
    """Separate features into base ML features and behavioral engine features."""
    print("\n" + "=" * 70)
    print("E. FEATURE AVAILABILITY ANALYSIS")
    print("=" * 70)

    # --- E.1: Base ML Model Features ---
    print("\n--- E.1: Base ML Model Features (available at transaction scoring time) ---")
    print(f"  {'Feature':30s} {'Source':20s} {'PreTxn?':8s} {'LeakRisk':10s} {'Include?':8s} {'Reason'}")
    print(f"  {'-'*30} {'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*40}")

    base_features = [
        ("amount", "raw column", "YES", "NONE", "YES", "Primary transaction attribute"),
        ("type (one-hot)", "raw column", "YES", "NONE", "YES", "Transaction category"),
        ("oldbalanceOrg", "raw column", "YES", "LOW", "YES", "Pre-txn sender balance"),
        ("oldbalanceDest", "raw column", "YES", "MEDIUM", "YES*", "Pre-txn receiver balance (caveat: not realistic in prod)"),
        ("step", "raw column", "YES", "LOW", "YES", "Time proxy — sim hour"),
        ("step_mod_24", "derived: step%24", "YES", "NONE", "YES", "Hour-of-day proxy"),
        ("amount_to_balance_ratio", "amount/(oldbalanceOrg+1)", "YES", "NONE", "YES", "How much of balance is being spent"),
        ("is_full_balance_transfer", "amount==oldbalanceOrg", "YES", "NONE", "YES", "Transferring entire balance (common fraud pattern)"),
        ("log_amount", "log1p(amount)", "YES", "NONE", "YES", "Log-scaled amount for better distribution"),
    ]

    for feat in base_features:
        print(f"  {feat[0]:30s} {feat[1]:20s} {feat[2]:8s} {feat[3]:10s} {feat[4]:8s} {feat[5]}")

    # --- E.2: Behavioral Engine Features ---
    print("\n--- E.2: Behavioral Engine Features (computed from PAST transactions only) ---")
    print(f"  {'Feature':35s} {'Computation':35s} {'PastOnly?':10s} {'PaySimSupport':14s}")
    print(f"  {'-'*35} {'-'*35} {'-'*10} {'-'*14}")

    behavioral_features = [
        ("tx_velocity_1h", "Count of txns by nameOrig in last 1 step", "YES", "YES"),
        ("tx_velocity_3h", "Count of txns by nameOrig in last 3 steps", "YES", "YES"),
        ("tx_velocity_24h", "Count of txns by nameOrig in last 24 steps", "YES", "YES"),
        ("counterparty_repetition", "Count of prev txns to same nameDest", "YES", "YES"),
        ("is_new_counterparty", "First-ever txn to this nameDest", "YES", "YES"),
        ("amount_escalation_ratio", "amount / customer rolling mean amount", "YES", "YES"),
        ("amount_vs_max", "amount / customer historical max amount", "YES", "YES"),
        ("customer_mean_amount", "Rolling mean of customer's past amounts", "YES", "YES"),
        ("customer_std_amount", "Rolling std of customer's past amounts", "YES", "YES"),
        ("amount_zscore", "(amount - mean) / std per customer", "YES", "YES"),
        ("dest_incoming_velocity", "Count of txns TO nameDest in last N steps", "YES", "YES"),
        ("device_fingerprint", "Device/IP-based features", "N/A", "NO — not in PaySim"),
        ("geo_distance", "Distance between sender/receiver locations", "N/A", "NO — not in PaySim"),
        ("time_since_last_tx", "Steps since customer's last transaction", "YES", "YES"),
    ]

    for feat in behavioral_features:
        print(f"  {feat[0]:35s} {feat[1]:35s} {feat[2]:10s} {feat[3]:14s}")

    # Actually compute some behavioral feature feasibility stats
    print("\n--- E.3: Behavioral Feature Feasibility Check ---")

    # How many customers have multiple transactions?
    orig_counts = df['nameOrig'].value_counts()
    print(f"  Customer transaction count distribution:")
    print(f"    Customers with 1 txn:    {(orig_counts == 1).sum():,} ({100*(orig_counts == 1).mean():.1f}%)")
    print(f"    Customers with 2 txns:   {(orig_counts == 2).sum():,}")
    print(f"    Customers with 3+ txns:  {(orig_counts >= 3).sum():,}")
    print(f"    Customers with 5+ txns:  {(orig_counts >= 5).sum():,}")
    print(f"    Customers with 10+ txns: {(orig_counts >= 10).sum():,}")
    print(f"    Max txns per customer:   {orig_counts.max()}")
    print(f"    Mean txns per customer:  {orig_counts.mean():.2f}")

    # What about fraud customers?
    fraud_customers = df[df['isFraud'] == 1]['nameOrig'].unique()
    fraud_customer_counts = orig_counts.loc[orig_counts.index.isin(fraud_customers)]
    print(f"\n  FRAUD customer transaction count distribution:")
    print(f"    Fraud customers total:     {len(fraud_customers):,}")
    print(f"    With 1 txn:  {(fraud_customer_counts == 1).sum():,} ({100*(fraud_customer_counts == 1).mean():.1f}%)")
    print(f"    With 2 txns: {(fraud_customer_counts == 2).sum():,}")
    print(f"    With 3+ txns:{(fraud_customer_counts >= 3).sum():,}")
    print(f"    Mean txns:   {fraud_customer_counts.mean():.2f}")

    # Counterparty repetition feasibility
    pair_counts = df.groupby(['nameOrig', 'nameDest']).size()
    print(f"\n  Counterparty pair analysis:")
    print(f"    Unique sender-receiver pairs: {len(pair_counts):,}")
    print(f"    Pairs with 1 txn:   {(pair_counts == 1).sum():,} ({100*(pair_counts == 1).mean():.1f}%)")
    print(f"    Pairs with 2+ txns: {(pair_counts >= 2).sum():,}")
    print(f"    Pairs with 3+ txns: {(pair_counts >= 3).sum():,}")
    print(f"    Max txns for a pair: {pair_counts.max()}")

    return {
        "base_features": [f[0] for f in base_features if f[4].startswith("YES")],
        "behavioral_features": [f[0] for f in behavioral_features if f[3].startswith("YES")],
        "unsupported_features": [f[0] for f in behavioral_features if f[3].startswith("NO")],
        "customer_txn_mean": orig_counts.mean(),
        "fraud_customer_single_txn_pct": (fraud_customer_counts == 1).mean() if len(fraud_customer_counts) > 0 else 0,
    }


# ============================================================================
# F. REDUNDANCY ANALYSIS
# ============================================================================

def redundancy_analysis(df: pd.DataFrame):
    """Check engineered features for redundancy with existing columns."""
    print("\n" + "=" * 70)
    print("F. REDUNDANCY ANALYSIS")
    print("=" * 70)

    print("\n--- Checking: balance_change_expected = oldbalanceOrg - amount ---")
    # This is a mathematical restatement. But is it redundant with newbalanceOrig?
    # If we're EXCLUDING newbalanceOrig (leakage), then oldbalanceOrg - amount
    # is just a linear combination of two features the model already has.
    # A tree model will learn this automatically. A linear model might benefit
    # from having it pre-computed, but it's still perfectly correlated with
    # the inputs.
    df_temp = df.sample(min(10000, len(df)), random_state=42)
    bce = df_temp['oldbalanceOrg'] - df_temp['amount']
    corr_old = np.corrcoef(bce, df_temp['oldbalanceOrg'])[0, 1]
    corr_amt = np.corrcoef(bce, df_temp['amount'])[0, 1]
    print(f"  Correlation with oldbalanceOrg: {corr_old:.4f}")
    print(f"  Correlation with amount:        {corr_amt:.4f}")
    print(f"  VERDICT: This feature is a LINEAR COMBINATION of two existing features.")
    print(f"           Tree models will learn this split automatically.")
    print(f"           For linear models, amount_to_balance_ratio is more informative.")
    print(f"           RECOMMENDATION: DO NOT INCLUDE (redundant).")

    print("\n--- Checking: amount_to_balance_ratio vs log_amount ---")
    ratio = df_temp['amount'] / (df_temp['oldbalanceOrg'] + 1)
    log_amt = np.log1p(df_temp['amount'])
    corr = np.corrcoef(ratio, log_amt)[0, 1]
    print(f"  Correlation: {corr:.4f}")
    print(f"  VERDICT: These capture different information (relative spend vs absolute scale).")
    print(f"           RECOMMENDATION: INCLUDE BOTH.")

    print("\n--- Checking: is_full_balance_transfer ---")
    full_transfer = (df_temp['amount'] == df_temp['oldbalanceOrg'])
    fraud_full = df_temp[full_transfer]['isFraud'].mean()
    fraud_not_full = df_temp[~full_transfer]['isFraud'].mean()
    print(f"  Fraud rate when full balance transfer: {fraud_full:.6f}")
    print(f"  Fraud rate otherwise:                  {fraud_not_full:.6f}")
    if fraud_full > 0 and fraud_not_full > 0:
        print(f"  Lift: {fraud_full / fraud_not_full:.1f}x")
    print(f"  VERDICT: Binary flag encoding a specific threshold of amount_to_balance_ratio.")
    print(f"           If it shows meaningful lift, INCLUDE. Otherwise, ratio alone suffices.")


# ============================================================================
# G. PAYSIM LIMITATIONS
# ============================================================================

def paysim_limitations(df: pd.DataFrame):
    """Critically analyze where PaySim does NOT represent real payment systems."""
    print("\n" + "=" * 70)
    print("G. PAYSIM DATASET LIMITATIONS")
    print("=" * 70)

    limitations = [
        (
            "Synthetic Data",
            "PaySim is a simulation, not real transaction data. It was generated using "
            "agent-based modeling calibrated on aggregate statistics from a real mobile money "
            "service in an undisclosed African country. Individual transaction patterns may "
            "not reflect real-world fraud behavior. The fraud labeling mechanism is rule-based "
            "within the simulation, not derived from actual investigations.",
            "Model performance on PaySim may not transfer to real payment data. "
            "PR-AUC/recall numbers should be interpreted as 'performance on this simulation', "
            "not 'expected production performance'."
        ),
        (
            "Mobile Money Simulation",
            "PaySim simulates a mobile money platform (M-Pesa-like), not a card payment "
            "gateway like Razorpay. Transaction types (CASH_IN, CASH_OUT, TRANSFER, PAYMENT, "
            "DEBIT) are mobile money operations, not UPI/card/netbanking transactions.",
            "Transaction type semantics differ from Razorpay. 'CASH_OUT' in mobile money "
            "means withdrawing to cash, not a card purchase. Our model learns mobile money "
            "fraud patterns, which may differ from UPI/card fraud."
        ),
        (
            "No Device Fingerprints",
            "PaySim contains no device ID, browser fingerprint, or app version information. "
            "Real fraud detection systems heavily rely on device intelligence.",
            "Our architecture's behavioral engine cannot implement device-based features. "
            "This is a significant gap vs production fraud detection."
        ),
        (
            "No IP/Geographic Data",
            "No IP addresses, GPS coordinates, or geographic information. Real systems use "
            "impossible-travel detection, VPN detection, and geo-velocity checks.",
            "Geographic fraud signals (impossible travel, unusual location) cannot be built "
            "or evaluated."
        ),
        (
            "Limited Merchant Information",
            "Destination accounts are identified only by opaque IDs (e.g., M123456789 for "
            "merchants, C123456789 for customers). No merchant category codes (MCC), "
            "merchant names, or business types.",
            "Merchant-risk features (high-risk MCC, new merchant, merchant fraud rate) "
            "cannot be implemented. Policy rule 'blocklisted entity' will only work with "
            "synthetic blocklist."
        ),
        (
            "Transaction Identifiers Are Opaque",
            "nameOrig and nameDest are synthetic IDs with no real-world meaning. No account "
            "age, registration date, or KYC information is available.",
            "Account-age-based features and KYC risk signals are not possible."
        ),
        (
            "Step as Time Proxy",
            "'step' represents simulated hours (1–743, ~30 days). It is NOT a real timestamp. "
            "There is no date, no day-of-week, no seasonal pattern. All 'temporal' features "
            "are proxies within a flat 30-day simulation window.",
            "Time-of-day and day-of-week patterns may be artificial. Seasonal fraud patterns "
            "cannot be studied."
        ),
        (
            "Limited Behavioral History",
            "The simulation runs for ~30 days. Real fraud detection benefits from months/years "
            "of customer history. PaySim customers may have very few transactions.",
            "Behavioral features (velocity, escalation, baseline deviation) will be computed "
            "on thin history. Many customers may have only 1-2 transactions, making per-customer "
            "statistics unreliable."
        ),
        (
            "Fraud Labeling Mechanism",
            "PaySim's fraud is injected by the simulation: specific agents are designated as "
            "fraudsters who drain victim accounts via TRANSFER and CASH_OUT. The fraud pattern "
            "is mostly 'drain entire balance', which is ONE specific fraud type.",
            "The model will learn to detect balance-draining fraud specifically. It may not "
            "generalize to other fraud types (card testing, account takeover, friendly fraud, "
            "etc.). Our task/commission scam scenario is deliberately synthetic because PaySim "
            "does not contain this pattern."
        ),
        (
            "Class Imbalance",
            "Fraud is extremely rare in PaySim (~0.13%). This is realistic for payment fraud "
            "but creates challenges for model training and evaluation.",
            "Must use appropriate handling: class weights, PR-AUC as primary metric, "
            "stratified considerations in time-based split."
        ),
    ]

    for i, (title, description, impact) in enumerate(limitations, 1):
        print(f"\n  {i}. {title}")
        print(f"     Description: {description}")
        print(f"     Impact: {impact}")

    # Architecture modifications needed
    print("\n" + "=" * 70)
    print("ARCHITECTURE MODIFICATIONS RECOMMENDED BASED ON PAYSIM LIMITATIONS")
    print("=" * 70)

    modifications = [
        (
            "Behavioral Engine: Reduced Effectiveness",
            "Many customers have very few transactions. Velocity and escalation features "
            "will be sparse or undefined for first-time transactors. The behavioral engine "
            "should handle cold-start gracefully (default/population-level baselines)."
        ),
        (
            "Policy Engine: Blocklist Rule is Synthetic",
            "No real entity data exists. The 'blocklisted entity' rule must use a synthetic "
            "blocklist or be driven by computed signals (e.g., accounts with high incoming "
            "fraud rate). Document this honestly."
        ),
        (
            "Counterparty Risk: Limited Signal",
            "With opaque IDs and no merchant metadata, counterparty risk is limited to "
            "statistical patterns (how many fraud txns go to this dest account). This is "
            "still useful but less rich than real-world merchant risk scoring."
        ),
        (
            "Synthetic Scenario: Necessary but Separate",
            "Task/commission scam patterns DO NOT exist in PaySim. Our synthetic scenario "
            "MUST go through the real pipeline but its risk tier assignments will depend on "
            "signals the behavioral engine detects, not on PaySim-trained fraud patterns. "
            "This is expected and honest."
        ),
        (
            "Amount-Based Features Will Dominate",
            "Given the limited feature space, amount and balance-related features will likely "
            "dominate. This is a known limitation of working with PaySim."
        ),
    ]

    for i, (title, description) in enumerate(modifications, 1):
        print(f"\n  {i}. {title}")
        print(f"     {description}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 1A: DATA PROFILING AND VALIDATION REPORT")
    print(f"Dataset: {DATASET_PATH}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    start = time.time()
    df = pd.read_csv(DATASET_PATH)
    load_time = time.time() - start
    print(f"Loaded in {load_time:.1f}s")

    # A. Dataset Profile
    stats = profile_dataset(df)

    # B. Fraud Distribution
    fraud_findings = analyze_fraud_distribution(df)

    # C. Temporal Split
    split_info = propose_temporal_split(df)

    # D. Leakage Analysis
    leakage = leakage_analysis(df)

    # E. Feature Availability
    features = feature_availability_analysis(df)

    # F. Redundancy Analysis
    redundancy_analysis(df)

    # G. PaySim Limitations
    paysim_limitations(df)

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 1A COMPLETE — AWAITING REVIEW BEFORE PROCEEDING")
    print("=" * 70)
    print(f"\n  Key findings to review:")
    print(f"    1. Total rows: {stats['total_rows']:,}")
    print(f"    2. Fraud cases: {stats['fraud_count']:,} ({100*stats['fraud_rate']:.4f}%)")
    print(f"    3. Fraud only in types: {fraud_findings['fraud_types']}")
    print(f"    4. Proposed split: train <={split_info['train_end_step']}, "
          f"val <={split_info['val_end_step']}, test >{split_info['val_end_step']}")
    print(f"    5. Train fraud: {split_info['train_fraud']:,}, "
          f"Val fraud: {split_info['val_fraud']:,}, "
          f"Test fraud: {split_info['test_fraud']:,}")
    print(f"    6. Excluded (leakage): newbalanceOrig, newbalanceDest, isFlaggedFraud")
    print(f"    7. Customer history: mean {features['customer_txn_mean']:.1f} txns/customer")
    print(f"    8. Fraud customers with only 1 txn: {100*features['fraud_customer_single_txn_pct']:.1f}%")
    print(f"\n  DO NOT proceed to model training until these findings are reviewed.")


if __name__ == "__main__":
    main()
