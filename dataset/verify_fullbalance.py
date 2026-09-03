"""Verify the is_full_balance_transfer finding on the full dataset."""
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/paysim.csv")

# Full dataset check: is_full_balance_transfer
full_transfer = (df["amount"] == df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)
print("=== is_full_balance_transfer on FULL dataset ===")
print(f"Full balance transfers: {full_transfer.sum():,}")
print(f"Fraud among full transfers: {df[full_transfer]['isFraud'].sum():,}")
nf_full = (full_transfer & (df["isFraud"] == 0)).sum()
print(f"Non-fraud among full transfers: {nf_full:,}")
if full_transfer.sum() > 0:
    print(f"Fraud rate (full transfer): {df[full_transfer]['isFraud'].mean():.6f}")

not_full = ~full_transfer
print(f"Fraud among NON-full transfers: {df[not_full]['isFraud'].sum():,}")
print(f"Fraud rate (not full transfer): {df[not_full]['isFraud'].mean():.6f}")

# How many fraud cases are full balance transfers?
fraud = df[df["isFraud"] == 1]
fraud_full = (fraud["amount"] == fraud["oldbalanceOrg"]) & (fraud["oldbalanceOrg"] > 0)
print(f"\nFraud that are full balance transfers: {fraud_full.sum():,} / {len(fraud):,} ({100*fraud_full.mean():.1f}%)")

# Fraud cases NOT full balance
fraud_not_full = fraud[~fraud_full]
print(f"Fraud cases NOT full balance: {len(fraud_not_full):,}")
if len(fraud_not_full) > 0:
    print(f"  Types: {fraud_not_full['type'].value_counts().to_dict()}")
    print(f"  Amount range: {fraud_not_full['amount'].min():.2f} to {fraud_not_full['amount'].max():.2f}")
    print(f"  oldbalanceOrg range: {fraud_not_full['oldbalanceOrg'].min():.2f} to {fraud_not_full['oldbalanceOrg'].max():.2f}")
    zero_bal = (fraud_not_full["oldbalanceOrg"] == 0).sum()
    print(f"  oldbalanceOrg == 0: {zero_bal:,}")

# Check: amount > oldbalanceOrg (overdraft-like fraud)
fraud_over = fraud[fraud["amount"] > fraud["oldbalanceOrg"]]
print(f"\nFraud where amount > oldbalanceOrg: {len(fraud_over):,}")
if len(fraud_over) > 0:
    print(f"  oldbalanceOrg == 0: {(fraud_over['oldbalanceOrg'] == 0).sum():,}")
    print(f"  Mean amount: {fraud_over['amount'].mean():,.2f}")
