"""FastAPI model-serving app with logging and Prometheus metrics."""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel
from starlette.responses import Response

from src.data import FEATURES

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models",
                               "model.joblib"))

app = FastAPI(title="Heart Disease Risk API")
REQUESTS = Counter("predict_requests_total", "Prediction requests",
                   ["status"])
LATENCY = Histogram("predict_latency_seconds", "Prediction latency")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class Patient(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Patient):
    start = time.time()
    try:
        row = pd.DataFrame([p.model_dump()])[FEATURES]
        proba = float(get_model().predict_proba(row)[0, 1])
    except Exception as exc:
        REQUESTS.labels(status="error").inc()
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    LATENCY.observe(time.time() - start)
    REQUESTS.labels(status="ok").inc()
    result = {"prediction": int(proba >= 0.5), "confidence": round(proba, 4)}
    log.info("input=%s output=%s", p.model_dump(), result)
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
