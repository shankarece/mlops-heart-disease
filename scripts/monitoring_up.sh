#!/usr/bin/env bash
# Start Prometheus + Grafana (pre-provisioned dashboard), send traffic, verify.
# Requires the API container from run_all.sh to be running on localhost:8000.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
die() { echo "ERROR: $*" >&2; exit 1; }
trap 'echo "FAILED near line $LINENO" >&2' ERR

curl -sf localhost:8000/health >/dev/null \
  || die "API not running on :8000 - run: bash scripts/run_all.sh"
if docker compose version >/dev/null 2>&1; then DC="docker compose"
elif command -v docker-compose >/dev/null; then DC="docker-compose"
else die "neither 'docker compose' nor 'docker-compose' is available"; fi

$DC -f monitoring/docker-compose.yml up -d
echo "waiting for Prometheus..."
for _ in $(seq 1 30); do
  curl -sf localhost:9090/-/ready >/dev/null && break; sleep 2
done
curl -sf localhost:9090/-/ready >/dev/null || die "Prometheus did not start"

bash scripts/generate_traffic.sh 80
sleep 15   # let Prometheus scrape (interval 10s)
n=$(curl -s 'localhost:9090/api/v1/query?query=sum(predict_requests_total)' \
  | python -c 'import sys, json; r = json.load(sys.stdin)["data"]["result"]; print(r[0]["value"][1] if r else 0)')
echo "Prometheus sees predict_requests_total = $n"
python -c "import sys; sys.exit(0 if float('$n') > 0 else 1)" \
  || die "Prometheus has no data (check http://localhost:9090/targets)"
echo
echo "OK. Open in Firefox in the lab:"
echo "  Prometheus: http://localhost:9090/targets   (target should be UP)"
echo "  Grafana:    http://localhost:3000           (admin / admin) -> Dashboards -> Heart Disease API"
