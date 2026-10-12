import os

import pytest
from fastapi.testclient import TestClient

from src.app import app

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
          "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
          "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_endpoint():
    assert client.get("/metrics").status_code == 200


@pytest.mark.skipif(not os.path.exists("models/model.joblib"),
                    reason="model not trained yet")
def test_predict():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1
