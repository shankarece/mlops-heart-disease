#!/usr/bin/env bash
# Deploy to a local Minikube cluster and smoke-test.  Usage: bash scripts/k8s_deploy.sh
set -Eeuo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs "$HOME/workspace/bin"
LOG="logs/k8s_deploy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export PATH="$HOME/workspace/bin:$PATH"
IMAGE=heart-api:latest
LPORT="${LPORT:-8080}"

step() { echo; echo "=== [$1] $2"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
trap 'echo "FAILED near line $LINENO - see $LOG" >&2' ERR
PF_PID=""
cleanup() { [ -n "$PF_PID" ] && kill "$PF_PID" 2>/dev/null || true; }
trap cleanup EXIT

step 1 "Pre-flight"
docker info >/dev/null 2>&1 || die "Docker daemon not reachable"
cpus=$(nproc); mem_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
echo "cpus=$cpus available_mem_mb=$mem_mb"
[ "$cpus" -ge 2 ] || die "Minikube needs >= 2 CPUs (found $cpus)"
[ "$mem_mb" -ge 2000 ] || die "Minikube needs ~2GB free RAM (found ${mem_mb}MB)"

step 2 "Install kubectl / minikube if missing (into ~/workspace/bin)"
if ! command -v kubectl >/dev/null; then
  ver=$(curl -sfL https://dl.k8s.io/release/stable.txt) || die "cannot reach dl.k8s.io"
  curl -sfL -o "$HOME/workspace/bin/kubectl" \
    "https://dl.k8s.io/release/$ver/bin/linux/amd64/kubectl" || die "kubectl download failed"
  chmod +x "$HOME/workspace/bin/kubectl"
fi
if ! command -v minikube >/dev/null; then
  curl -sfL -o "$HOME/workspace/bin/minikube" \
    https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 \
    || die "minikube download failed (install it manually or use Docker Desktop on your laptop)"
  chmod +x "$HOME/workspace/bin/minikube"
fi
kubectl version --client | head -1
minikube version | head -1

step 3 "Start Minikube (docker driver)"
if ! minikube status >/dev/null 2>&1; then
  minikube start --driver=docker
fi
kubectl get nodes

step 4 "Make sure the image exists, then load it into Minikube"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" .
minikube image load "$IMAGE"

step 5 "Apply manifests and wait for rollout"
kubectl apply -f k8s/
kubectl rollout status deployment/heart-api --timeout=240s \
  || { kubectl describe pods -l app=heart-api | tail -40; die "rollout failed"; }

step 6 "Smoke test through the Service (port-forward)"
kubectl port-forward svc/heart-api "$LPORT:80" >/dev/null 2>&1 &
PF_PID=$!
ok=0
for _ in $(seq 1 20); do
  if curl -sf "localhost:$LPORT/health" >/dev/null; then ok=1; break; fi
  sleep 1
done
[ "$ok" = 1 ] || die "service not reachable on localhost:$LPORT"
echo "health:  $(curl -s localhost:$LPORT/health)"
echo "predict: $(curl -sf -X POST localhost:$LPORT/predict -H 'Content-Type: application/json' -d @sample_input.json)"

step 7 "Evidence for screenshots"
kubectl get deployments,pods,svc -o wide
echo
echo "External access (for a LoadBalancer screenshot) - run in a second terminal:"
echo "  minikube tunnel        # gives the Service an EXTERNAL-IP"
echo "  minikube service heart-api --url"
echo; echo "K8S DEPLOY OK. Log: $LOG"
