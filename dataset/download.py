"""Download PaySim dataset from Kaggle public API."""
import urllib.request
import zipfile
import os
import glob

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(DATASET_DIR, "paysim.zip")
CSV_PATH = os.path.join(DATASET_DIR, "paysim.csv")

def download():
    if os.path.exists(CSV_PATH):
        size_mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
        print(f"Dataset already exists: {size_mb:.1f} MB")
        return

    url = "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1"
    print(f"Downloading from Kaggle API ({url})...")
    print("This is ~186 MB compressed, may take a few minutes...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    urllib.request.urlretrieve(url, ZIP_PATH)
    zip_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"Downloaded ZIP: {zip_size:.1f} MB")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATASET_DIR)
    print("Extracted.")

    # Find the CSV (name may vary)
    csvs = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    for csv in csvs:
        if csv != CSV_PATH and "paysim" not in os.path.basename(csv).lower():
            os.rename(csv, CSV_PATH)
            print(f"Renamed {os.path.basename(csv)} -> paysim.csv")
            break
        elif csv != CSV_PATH:
            os.rename(csv, CSV_PATH)
            print(f"Renamed {os.path.basename(csv)} -> paysim.csv")
            break

    # If still not found, check if extracted file has different name
    if not os.path.exists(CSV_PATH):
        csvs = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
        if csvs:
            os.rename(csvs[0], CSV_PATH)
            print(f"Renamed {os.path.basename(csvs[0])} -> paysim.csv")

    # Cleanup zip
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("Removed ZIP file.")

    final_size = os.path.getsize(CSV_PATH) / (1024 * 1024)
    print(f"Final dataset: {final_size:.1f} MB")

if __name__ == "__main__":
    download()
