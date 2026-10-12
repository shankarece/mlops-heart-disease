# Heart Disease Risk - MLOps Assignment 01

End-to-end ML pipeline: data, training with MLflow, FastAPI service, Docker, Kubernetes, CI/CD, monitoring.

## Setup
```bash
git clone https://github.com/shankarece/mlops-heart-disease.git
cd mlops-heart-disease
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline
```bash
python -m src.download_data      # fetches UCI Cleveland data into data/raw/
python -m src.train              # trains LR + RF, logs to MLflow, saves models/model.joblib
mlflow ui                        # view runs at http://127.0.0.1:5000
python -m pytest -v              # unit tests
flake8 src tests --max-line-length=100
```

## Serve locally
```bash
uvicorn src.app:app --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```
Endpoints: `/predict`, `/health`, `/metrics` (Prometheus).

## Docker
```bash
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
```

## Kubernetes (Minikube / Docker Desktop)
```bash
kubectl apply -f k8s/
kubectl get pods,svc
```

## Monitoring
`monitoring/prometheus.yml` scrapes `/metrics`. Run Prometheus and Grafana in Docker, add Prometheus as a Grafana data source, and chart `predict_requests_total` and `predict_latency_seconds`.

## Repo layout
`src/` code, `tests/` pytest, `notebooks/` EDA, `data/` datasets, `k8s/` manifests, `monitoring/` Prometheus config, `screenshots/` report evidence, `.github/workflows/ci.yml` CI.
