#!/usr/bin/env bash
# Send N prediction requests (default 60) so monitoring has data.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-60}"; URL="${URL:-http://localhost:8000}"
for i in $(seq 1 "$N"); do
  curl -sf -X POST "$URL/predict" -H "Content-Type: application/json" \
    -d @sample_input.json >/dev/null || echo "request $i failed"
  sleep 0.1
done
echo "sent $N requests to $URL"
