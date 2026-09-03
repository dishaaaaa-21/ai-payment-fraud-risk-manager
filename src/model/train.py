"""
Model Training Pipeline — Phase 1B
====================================
Trains and evaluates fraud detection models on PaySim data.

Models:
  1. Naive baseline (rules-only, no ML)
  2. Logistic Regression baseline
  3. XGBoost main model

Each model is trained in two variants:
  - PRODUCTION: 12 features, excludes is_full_balance_transfer
  - BENCHMARK: 13 features, includes is_full_balance_transfer

All model selection, tuning, and threshold setting uses validation set ONLY.
Test set is NOT touched here.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
import xgboost as xgb

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src.data.features import (
    build_features, get_feature_matrix, validate_no_leakage,
    PRODUCTION_FEATURES, BENCHMARK_FEATURES
)
from src.data.split import time_split, print_split_summary


# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = os.path.join(_PROJECT_ROOT, "dataset", "paysim.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Threshold for binary classification — will be tuned on validation
DEFAULT_THRESHOLD = 0.5


# ============================================================================
# NAIVE BASELINE (rules-only, no ML)
# ============================================================================

def naive_baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Rules-only baseline: flag as fraud if:
      - type is CASH_OUT or TRANSFER, AND
      - amount > 200,000

    This is deliberately simple to show whether ML adds value.
    Returns binary predictions (0/1).
    """
    is_risky_type = df["type"].isin(["CASH_OUT", "TRANSFER"])
    is_high_amount = df["amount"] > 200_000
    return (is_risky_type & is_high_amount).astype(int).values


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = "Model",
    latency_ms: float = None,
) -> dict:
    """Compute all required evaluation metrics."""
    # Binary predictions from probabilities if provided
    if y_prob is not None:
        y_pred_binary = (y_prob >= threshold).astype(int)
    else:
        y_pred_binary = y_pred

    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    # PR-AUC (headline metric) — requires probability scores
    if y_prob is not None:
        pr_auc = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        pr_auc = None
        roc_auc = None

    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_positive": int(tp + fn),
        "total_negative": int(tn + fp),
        "latency_ms": latency_ms,
    }

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print a metrics dictionary."""
    print(f"\n  --- {metrics['model']} ---")
    print(f"  Threshold:  {metrics['threshold']:.3f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    if metrics["pr_auc"] is not None:
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}  (headline)")
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}  (secondary)")
    print(f"  Confusion:  TP={metrics['tp']:,}  FP={metrics['fp']:,}  "
          f"FN={metrics['fn']:,}  TN={metrics['tn']:,}")
    if metrics["latency_ms"] is not None:
        print(f"  Latency:    {metrics['latency_ms']:.2f} ms/transaction")


def find_best_threshold(y_true, y_prob, metric="f1"):
    """Find the threshold that maximizes F1 on validation data."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 for each threshold
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    best_idx = np.argmax(f1s[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    return best_threshold, best_f1


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_logistic_regression(X_train, y_train, X_val, y_val, variant_name):
    """Train a logistic regression with StandardScaler."""
    print(f"\n{'='*60}")
    print(f"Training Logistic Regression ({variant_name})")
    print(f"{'='*60}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train samples: {len(X_train):,} (fraud: {y_train.sum():,})")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train with class_weight='balanced' to handle imbalance
    start = time.time()
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")

    # Predict probabilities on validation
    start = time.time()
    y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
    pred_time = time.time() - start
    latency_ms = (pred_time / len(X_val)) * 1000

    # Find best threshold on validation
    best_thresh, best_f1 = find_best_threshold(y_val, y_prob_val)
    print(f"  Best F1 threshold (validation): {best_thresh:.4f} (F1={best_f1:.4f})")

    # Metrics at default 0.5 and best threshold
    metrics_default = compute_metrics(
        y_val, None, y_prob_val, 0.5,
        f"LogReg ({variant_name}) @0.5", latency_ms
    )
    metrics_best = compute_metrics(
        y_val, None, y_prob_val, best_thresh,
        f"LogReg ({variant_name}) @best", latency_ms
    )

    print_metrics(metrics_default)
    print_metrics(metrics_best)

    # Calibration check: mean predicted prob vs actual fraud rate
    mean_prob = y_prob_val.mean()
    actual_rate = y_val.mean()
    print(f"\n  Calibration: mean P(fraud)={mean_prob:.6f}, actual rate={actual_rate:.6f}")

    return {
        "model": model,
        "scaler": scaler,
        "best_threshold": best_thresh,
        "metrics_default": metrics_default,
        "metrics_best": metrics_best,
        "y_prob_val": y_prob_val,
    }


def train_xgboost(X_train, y_train, X_val, y_val, variant_name):
    """Train XGBoost with class imbalance handling."""
    print(f"\n{'='*60}")
    print(f"Training XGBoost ({variant_name})")
    print(f"{'='*60}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Train samples: {len(X_train):,} (fraud: {y_train.sum():,})")

    # scale_pos_weight = # negatives / # positives
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    # Train XGBoost
    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        tree_method="hist",  # Fast histogram-based training
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - start
    best_iteration = model.best_iteration
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best iteration: {best_iteration} / 300")

    # Predict probabilities on validation
    start = time.time()
    y_prob_val = model.predict_proba(X_val)[:, 1]
    pred_time = time.time() - start
    latency_ms = (pred_time / len(X_val)) * 1000

    # Find best threshold on validation
    best_thresh, best_f1 = find_best_threshold(y_val, y_prob_val)
    print(f"  Best F1 threshold (validation): {best_thresh:.4f} (F1={best_f1:.4f})")

    # Metrics at default 0.5 and best threshold
    metrics_default = compute_metrics(
        y_val, None, y_prob_val, 0.5,
        f"XGBoost ({variant_name}) @0.5", latency_ms
    )
    metrics_best = compute_metrics(
        y_val, None, y_prob_val, best_thresh,
        f"XGBoost ({variant_name}) @best", latency_ms
    )

    print_metrics(metrics_default)
    print_metrics(metrics_best)

    # Feature importance
    print(f"\n  --- Feature Importance (gain) ---")
    importance = model.get_booster().get_score(importance_type="gain")
    # Map f0, f1, ... to feature names
    feature_names = list(X_train.columns)
    named_importance = {}
    for fname, score in importance.items():
        idx = int(fname.replace("f", ""))
        if idx < len(feature_names):
            named_importance[feature_names[idx]] = score
    for feat, score in sorted(named_importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:30s} {score:>12.1f}")

    # Calibration check
    mean_prob = y_prob_val.mean()
    actual_rate = y_val.mean()
    print(f"\n  Calibration: mean P(fraud)={mean_prob:.6f}, actual rate={actual_rate:.6f}")

    # Probability calibration using Platt scaling on validation
    print(f"\n  Applying Platt scaling calibration...")
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    y_prob_cal = calibrated.predict_proba(X_val)[:, 1]
    cal_mean = y_prob_cal.mean()
    print(f"  Post-calibration: mean P(fraud)={cal_mean:.6f} (target: {actual_rate:.6f})")

    # Re-compute metrics with calibrated probabilities
    best_thresh_cal, best_f1_cal = find_best_threshold(y_val, y_prob_cal)
    metrics_calibrated = compute_metrics(
        y_val, None, y_prob_cal, best_thresh_cal,
        f"XGBoost ({variant_name}) calibrated @best", latency_ms
    )
    print_metrics(metrics_calibrated)

    return {
        "model": model,
        "calibrated_model": calibrated,
        "best_threshold": best_thresh,
        "best_threshold_calibrated": best_thresh_cal,
        "metrics_default": metrics_default,
        "metrics_best": metrics_best,
        "metrics_calibrated": metrics_calibrated,
        "y_prob_val": y_prob_val,
        "y_prob_cal": y_prob_cal,
        "feature_importance": named_importance,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 1B: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # --- Step 1: Load Data ---
    print("\n[1/5] Loading dataset...")
    start = time.time()
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df):,} rows in {time.time()-start:.1f}s")

    # --- Step 2: Build Features ---
    print("\n[2/5] Building leakage-safe features...")
    df = build_features(df)
    print(f"  Features built. Columns now: {len(df.columns)}")

    # Validate no leakage in both variants
    print("  Validating production features...")
    validate_no_leakage(df, PRODUCTION_FEATURES)
    print("  Validating benchmark features...")
    validate_no_leakage(df, BENCHMARK_FEATURES)

    # --- Step 3: Time-Based Split ---
    print("\n[3/5] Applying time-based split...")
    train_df, val_df, test_df = time_split(df)
    print_split_summary(train_df, val_df, test_df)

    # Extract feature matrices for both variants
    X_train_prod, y_train = get_feature_matrix(train_df, "production")
    X_val_prod, y_val = get_feature_matrix(val_df, "production")

    X_train_bench, _ = get_feature_matrix(train_df, "benchmark")
    X_val_bench, _ = get_feature_matrix(val_df, "benchmark")

    # --- Step 4: Naive Baseline ---
    print("\n[4/5] Evaluating naive baseline (rules-only)...")
    naive_pred_val = naive_baseline_predict(val_df)
    naive_metrics = compute_metrics(
        y_val.values, naive_pred_val, None, 0.5,
        "Naive Baseline (rules-only)"
    )
    print_metrics(naive_metrics)

    # --- Step 5: Train ML Models ---
    print("\n[5/5] Training ML models...")

    all_results = {"naive_baseline": naive_metrics}

    # 5a. Logistic Regression — Production
    lr_prod = train_logistic_regression(
        X_train_prod, y_train, X_val_prod, y_val, "Production"
    )
    all_results["lr_production"] = lr_prod["metrics_best"]

    # 5b. Logistic Regression — Benchmark
    lr_bench = train_logistic_regression(
        X_train_bench, y_train, X_val_bench, y_val, "Benchmark"
    )
    all_results["lr_benchmark"] = lr_bench["metrics_best"]

    # 5c. XGBoost — Production
    xgb_prod = train_xgboost(
        X_train_prod, y_train, X_val_prod, y_val, "Production"
    )
    all_results["xgb_production"] = xgb_prod["metrics_best"]
    all_results["xgb_production_calibrated"] = xgb_prod["metrics_calibrated"]

    # 5d. XGBoost — Benchmark
    xgb_bench = train_xgboost(
        X_train_bench, y_train, X_val_bench, y_val, "Benchmark"
    )
    all_results["xgb_benchmark"] = xgb_bench["metrics_best"]
    all_results["xgb_benchmark_calibrated"] = xgb_bench["metrics_calibrated"]

    # --- Save Models ---
    print("\n\nSaving models...")
    joblib.dump(lr_prod["model"], os.path.join(MODELS_DIR, "lr_production.pkl"))
    joblib.dump(lr_prod["scaler"], os.path.join(MODELS_DIR, "lr_production_scaler.pkl"))
    joblib.dump(lr_bench["model"], os.path.join(MODELS_DIR, "lr_benchmark.pkl"))
    joblib.dump(lr_bench["scaler"], os.path.join(MODELS_DIR, "lr_benchmark_scaler.pkl"))
    xgb_prod["model"].save_model(os.path.join(MODELS_DIR, "xgb_production.json"))
    joblib.dump(xgb_prod["calibrated_model"], os.path.join(MODELS_DIR, "xgb_production_calibrated.pkl"))
    xgb_bench["model"].save_model(os.path.join(MODELS_DIR, "xgb_benchmark.json"))
    joblib.dump(xgb_bench["calibrated_model"], os.path.join(MODELS_DIR, "xgb_benchmark_calibrated.pkl"))

    # Save thresholds and feature lists
    config = {
        "production_features": PRODUCTION_FEATURES,
        "benchmark_features": BENCHMARK_FEATURES,
        "xgb_production_threshold": float(xgb_prod["best_threshold"]),
        "xgb_production_threshold_calibrated": float(xgb_prod["best_threshold_calibrated"]),
        "xgb_benchmark_threshold": float(xgb_bench["best_threshold"]),
        "xgb_benchmark_threshold_calibrated": float(xgb_bench["best_threshold_calibrated"]),
        "lr_production_threshold": float(lr_prod["best_threshold"]),
        "lr_benchmark_threshold": float(lr_bench["best_threshold"]),
    }
    with open(os.path.join(MODELS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("  Models and config saved to models/")

    # --- Summary Table ---
    print("\n" + "=" * 70)
    print("PHASE 1B VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  NOTE: All metrics below are on VALIDATION set only (steps 409-557).")
    print(f"  Test set has NOT been touched.")
    print(f"  Threshold for each model is optimized for F1 on validation.")

    print(f"\n  {'Model':45s} {'PR-AUC':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'ROC-AUC':>8s}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    summary_order = [
        ("Naive Baseline (rules-only)", "naive_baseline"),
        ("LogReg (Production) @best", "lr_production"),
        ("LogReg (Benchmark) @best", "lr_benchmark"),
        ("XGBoost (Production) @best", "xgb_production"),
        ("XGBoost (Production) calibrated", "xgb_production_calibrated"),
        ("XGBoost (Benchmark) @best", "xgb_benchmark"),
        ("XGBoost (Benchmark) calibrated", "xgb_benchmark_calibrated"),
    ]

    for label, key in summary_order:
        m = all_results[key]
        pr = f"{m['pr_auc']:.4f}" if m.get("pr_auc") is not None else "N/A"
        roc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "N/A"
        print(f"  {label:45s} {pr:>8s} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {roc:>8s}")

    # Save summary
    summary_path = os.path.join(MODELS_DIR, "validation_results.json")
    # Convert for JSON serialization
    serializable_results = {}
    for key, val in all_results.items():
        serializable_results[key] = {k: (float(v) if isinstance(v, (np.floating, float)) else
                                         int(v) if isinstance(v, (np.integer, int)) else
                                         str(v) if v is not None else None)
                                     for k, v in val.items()}
    with open(summary_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n  Results saved to {summary_path}")

    print(f"\n  SELECTED PRIMARY MODEL: XGBoost (Production) calibrated")
    print(f"  REASONING: Best PR-AUC among production variants; calibrated")
    print(f"  probabilities are needed for risk decision engine downstream.")
    print(f"\n  DO NOT evaluate on test set until all architecture decisions are final.")


if __name__ == "__main__":
    main()
