"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/."""
import os
import sys

import requests

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_FILE = os.path.join(OUT_DIR, "processed.cleveland.data")


def download(url: str = URL, out_file: str = OUT_FILE) -> str:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(out_file, "wb") as f:
        f.write(resp.content)
    return out_file


if __name__ == "__main__":
    try:
        path = download()
    except Exception as exc:  # fail loudly for CI
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Saved to {path}")
