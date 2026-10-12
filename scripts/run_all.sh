#!/usr/bin/env bash
# One-command pipeline: checks -> install -> data -> EDA -> train -> lint -> tests
# -> docker build -> run -> API smoke test.  Usage: bash scripts/run_all.sh [--skip-docker]
set -Eeuo pipefail
cd "$(dirname "$0")/.."

SKIP_DOCKER=0
[ "${1:-}" = "--skip-docker" ] && SKIP_DOCKER=1
PORT="${PORT:-8000}"
IMAGE=heart-api:latest
NAME=heart-api
mkdir -p logs
LOG="logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== [$1] $2"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
trap 'echo "FAILED near line $LINENO - see $LOG" >&2' ERR

step 1 "Pre-flight checks"
command -v python >/dev/null || die "python not found"
python -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || die "Python 3.10+ required"
echo "python: $(python --version 2>&1)"
case "$PWD" in
  "$HOME"/workspace/*) ;;
  *) echo "WARNING: repo is outside ~/workspace; the lab deletes it on shutdown." ;;
esac
if [ ! -s data/raw/processed.cleveland.data ]; then
  curl -sS -o /dev/null -m 20 https://archive.ics.uci.edu \
    || die "cannot reach archive.ics.uci.edu (needed to download the dataset)"
fi
if [ "$SKIP_DOCKER" = 0 ]; then
  docker info >/dev/null 2>&1 || die "Docker daemon not reachable (try: docker ps)"
  echo "docker: OK"
fi

step 2 "Install dependencies"
python -m pip install -q -r requirements.txt

step 3 "Download and clean data"
if [ -s data/raw/processed.cleveland.data ]; then
  echo "dataset already present, skipping download"
else
  python -m src.download_data
fi

step 4 "EDA (plots -> reports/eda/)"
python -m src.eda

step 5 "Train models (MLflow tracking -> mlruns/)"
export MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_DISABLE_TELEMETRY=true
python -m src.train
[ -s models/model.joblib ] || die "models/model.joblib was not created"

step 6 "Lint"
python -m flake8 src tests --max-line-length=100

step 7 "Unit tests"
python -m pytest -q

if [ "$SKIP_DOCKER" = 1 ]; then
  echo; echo "Done (Docker steps skipped). Log: $LOG"; exit 0
fi

step 8 "Docker build (trains the model inside the image)"
docker build -t "$IMAGE" .

step 9 "Run container and smoke-test the API"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d -p "$PORT:8000" --name "$NAME" "$IMAGE" >/dev/null \
  || die "could not start container (is port $PORT already in use? use PORT=8001)"
ok=0
for _ in $(seq 1 30); do
  if curl -sf "localhost:$PORT/health" >/dev/null; then ok=1; break; fi
  sleep 2
done
[ "$ok" = 1 ] || { docker logs "$NAME" | tail -30; die "API did not become healthy"; }
echo "health: $(curl -s localhost:$PORT/health)"
resp=$(curl -sf -X POST "localhost:$PORT/predict" \
  -H "Content-Type: application/json" -d @sample_input.json) \
  || { docker logs "$NAME" | tail -30; die "/predict failed"; }
echo "predict: $resp"
echo "$resp" | python -c \
  'import sys, json; d = json.load(sys.stdin); assert d["prediction"] in (0, 1) and 0 <= d["confidence"] <= 1' \
  || die "unexpected /predict response"
curl -sf "localhost:$PORT/metrics" | grep -q '^predict_requests_total' \
  || die "/metrics missing predict_requests_total"
echo "metrics: OK"
docker ps --filter "name=$NAME"
echo; echo "ALL CHECKS PASSED. Log: $LOG"
echo "Next: bash scripts/k8s_deploy.sh   then   bash scripts/monitoring_up.sh"
