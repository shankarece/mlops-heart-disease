"""Command-line inference: score one patient (JSON file) with the saved pipeline.

Usage:  python -m src.predict sample_input.json
"""
import json
import os
import sys

import joblib
import pandas as pd

from src.data import FEATURES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")


def predict(record: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([record])[FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    return {"prediction": int(proba >= 0.5), "confidence": round(proba, 4)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.predict sample_input.json")
    with open(sys.argv[1]) as f:
        print(json.dumps(predict(json.load(f))))
