#!/usr/bin/env bash
# Build reports/MLOps_Assignment1_Report.docx from your real results.
# Run AFTER scripts/run_all.sh (needs mlruns/ and reports/eda/).
# Screenshots: put PNGs named as in README into screenshots/ first (missing ones get a marked placeholder).
set -Eeuo pipefail
cd "$(dirname "$0")/.."
[ -d reports/eda ] || { echo "ERROR: run 'bash scripts/run_all.sh' first (EDA plots missing)" >&2; exit 1; }
python -m pip install -q -r requirements-report.txt
MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_DISABLE_TELEMETRY=true python -m src.make_report
