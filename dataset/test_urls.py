import urllib.request
import os

urls_to_try = [
    "https://storage.googleapis.com/kaggle-data-sets/1069/1924/compressed/PS_20174392719_1491204439457_log.csv.zip",
    "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1",
]
for url in urls_to_try:
    print(f"Trying: {url[:80]}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=15)
        content_length = response.headers.get("Content-Length", "unknown")
        print(f"Response code: {response.status}, size: {content_length}")
        break
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {str(e)[:100]}")
