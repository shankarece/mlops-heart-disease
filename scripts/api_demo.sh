#!/usr/bin/env bash
# Evidence for the Docker screenshot: container status, /health, /predict.
cd "$(dirname "$0")/.." || exit 1
clear
docker ps
echo
echo '$ curl localhost:8000/health'
curl -s localhost:8000/health; echo
echo
echo '$ curl -X POST localhost:8000/predict -d @sample_input.json'
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json; echo
