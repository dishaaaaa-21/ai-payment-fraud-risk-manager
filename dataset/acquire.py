# Dataset Acquisition — PaySim Fraud Detection Dataset
# =====================================================
#
# The PaySim dataset simulates mobile money transactions based on a sample
# of real transactions from a mobile money service in Africa. It contains
# ~6.3 million rows with labeled fraud.
#
# Source: https://www.kaggle.com/datasets/ealaxi/paysim1
# Paper: "PaySim: A financial mobile money simulator for fraud detection"
#        by E. A. Lopez-Rojas, A. Elmir, S. Axelsson (2016)
#
# HOW TO ACQUIRE THE DATASET
# ===========================
#
# Option 1: Kaggle CLI (recommended)
#   1. Install kaggle: pip install kaggle
#   2. Set up Kaggle API credentials:
#      - Go to https://www.kaggle.com/settings → Create New Token
#      - Save kaggle.json to ~/.kaggle/ (Linux/Mac) or
#        C:\Users\<username>\.kaggle\ (Windows)
#   3. Run: kaggle datasets download -d ealaxi/paysim1 -p dataset/ --unzip
#   4. Rename the CSV if needed: the file should be at dataset/paysim.csv
#
# Option 2: Manual download
#   1. Go to https://www.kaggle.com/datasets/ealaxi/paysim1
#   2. Click "Download" (requires free Kaggle account)
#   3. Extract the ZIP file
#   4. Place the CSV at: dataset/paysim.csv
#
# Option 3: Run this script (uses kaggle CLI under the hood)
#   python dataset/acquire.py
#
# EXPECTED SCHEMA
# ===============
# step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,
# oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
#
# Expected size: ~470 MB, ~6.3 million rows

import os
import sys
import subprocess


DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATASET_DIR, "paysim.csv")

EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud"
]


def validate_dataset(path: str) -> bool:
    """Validate the dataset exists and has the expected schema."""
    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at {path}")
        return False

    import pandas as pd
    # Read only the header to validate schema
    df_head = pd.read_csv(path, nrows=5)
    missing_cols = set(EXPECTED_COLUMNS) - set(df_head.columns)
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return False

    extra_cols = set(df_head.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        print(f"WARNING: Unexpected extra columns: {extra_cols}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Dataset found at {path}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Columns: {list(df_head.columns)}")
    print(f"Schema validation: PASSED")
    return True


def acquire_via_kaggle():
    """Download dataset using kaggle CLI."""
    print("Attempting download via kaggle CLI...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "ealaxi/paysim1",
             "-p", DATASET_DIR, "--unzip"],
            check=True
        )
        # The downloaded file may have a different name
        for fname in os.listdir(DATASET_DIR):
            if fname.endswith(".csv") and fname != "paysim.csv":
                old_path = os.path.join(DATASET_DIR, fname)
                os.rename(old_path, DATASET_PATH)
                print(f"Renamed {fname} -> paysim.csv")
                break
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"kaggle CLI failed: {e}")
        return False


def main():
    print("=" * 60)
    print("PaySim Dataset Acquisition")
    print("=" * 60)

    # Check if already present
    if os.path.exists(DATASET_PATH):
        print("Dataset already exists. Validating...")
        if validate_dataset(DATASET_PATH):
            print("Dataset is ready.")
            return 0
        else:
            print("Existing dataset failed validation.")
            return 1

    # Try kaggle CLI
    if acquire_via_kaggle():
        if validate_dataset(DATASET_PATH):
            print("Dataset acquired and validated successfully.")
            return 0

    # Manual fallback
    print()
    print("MANUAL DOWNLOAD REQUIRED")
    print("-" * 40)
    print("1. Go to: https://www.kaggle.com/datasets/ealaxi/paysim1")
    print("2. Click 'Download' (requires free Kaggle account)")
    print("3. Extract the ZIP file")
    print(f"4. Place the CSV at: {DATASET_PATH}")
    print()
    print("Then re-run: python dataset/acquire.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
