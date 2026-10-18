"""Capture the report screenshots automatically.

Web UIs (MLflow, Prometheus, Grafana, GitHub Actions) are captured with a
headless Chrome/Chromium via Playwright. Terminal evidence (training output,
Docker, Kubernetes) is produced by running the real commands and rendering
their actual output as a terminal-style image. Nothing is invented: every image
shows live output or UI. Run:  python -m src.capture_screenshots
"""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time

import mlflow
import requests
from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHOTS = os.path.join(ROOT, "screenshots")
REPO = "shankarece/mlops-heart-disease"
MLFLOW_PORT = 5055
PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")
GRAFANA_URL = os.environ.get("GRAFANA_URL", "http://localhost:3000")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
results = {}
os.environ["PATH"] = os.path.expanduser("~/workspace/bin") + os.pathsep + os.environ["PATH"]


def log(msg):
    print(msg, flush=True)


def launch(p):
    args = ["--no-sandbox", "--disable-dev-shm-usage"]
    for name in ("google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser", "chrome"):
        path = shutil.which(name)
        if path:
            try:
                return p.chromium.launch(executable_path=path, args=args)
            except Exception as exc:  # try the next option
                log(f"  could not launch {path}: {str(exc)[:120]}")
    try:
        return p.chromium.launch(args=args)
    except Exception:
        log("  no usable browser found; trying 'playwright install chromium' ...")
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"],
                       check=False, timeout=900)
        return p.chromium.launch(args=args)


def run(cmd, timeout=180):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout, cwd=ROOT, executable="/bin/bash")
    out = ANSI.sub("", (r.stdout or "") + (r.stderr or ""))
    return r.returncode, out.rstrip("\n")


# ---------------------------------------------------------------- terminal ----
def terminal(browser, name, blocks, footer=None):
    """blocks: list of (command, output). Renders them as a terminal image."""
    parts = []
    for cmd, out in blocks:
        parts.append(f'<span style="color:#6ab04c">$ {html.escape(cmd)}</span>')
        if out:
            parts.append(html.escape(out))
        parts.append("")
    if footer:
        parts.append(f'<span style="color:#888">{html.escape(footer)}</span>')
    body = "\n".join(parts)
    page = browser.new_page(viewport={"width": 1200, "height": 400})
    page.set_content(
        "<html><body style='margin:0;background:#1e1e1e'>"
        "<pre id='t' style='margin:0;padding:18px 22px;color:#d4d4d4;"
        "font:14px/1.4 \"DejaVu Sans Mono\",Menlo,Consolas,monospace;"
        f"white-space:pre-wrap;word-break:break-all;width:1150px'>{body}</pre>"
        "</body></html>")
    path = os.path.join(SHOTS, name)
    page.locator("#t").screenshot(path=path)
    page.close()
    results[name] = "ok"
    log(f"  captured {name}")


def latest_log(prefix):
    logs = sorted(f for f in os.listdir(os.path.join(ROOT, "logs"))
                  if f.startswith(prefix)) if os.path.isdir(os.path.join(ROOT, "logs")) else []
    return os.path.join(ROOT, "logs", logs[-1]) if logs else None


def shot_train(browser):
    log("[train output] running src.train (about a minute)...")
    env = "MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_DISABLE_TELEMETRY=true "
    rc, out = run(env + f"{sys.executable} -m src.train 2>&1", timeout=900)
    kept = "\n".join(ln for ln in out.splitlines() if "WARNING" not in ln)
    if rc != 0:
        results["08_train_output.png"] = "FAILED (training error)"
        log(out[-800:])
        return
    terminal(browser, "08_train_output.png",
             [("python -m src.train", kept)],
             footer="(MLflow deprecation WARNING lines omitted)")


def shot_docker(browser):
    log("[docker evidence]")
    rc0, imgs = run("docker images heart-api")
    rc1, ps = run("docker ps --filter name=heart-api "
                  "--format 'table {{.ID}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}\\t{{.Names}}'")
    rc2, health = run(f"curl -s {API_URL}/health")
    rc3, pred = run(f"curl -s -X POST {API_URL}/predict -H 'Content-Type: application/json' "
                    "-d @sample_input.json")
    rc4, logs = run("docker logs --tail 4 heart-api 2>&1")
    if rc2 != 0 or "ok" not in health or "prediction" not in pred:
        results["04_docker_build_run.png"] = "FAILED (API not running on :8000)"
        return
    blocks = [("docker build -t heart-api:latest .", ""),
              ("docker images heart-api", imgs),
              ("docker run -d -p 8000:8000 --name heart-api heart-api:latest", ""),
              ("docker ps --filter name=heart-api", ps),
              (f"curl {API_URL}/health", health),
              (f"curl -X POST {API_URL}/predict -H 'Content-Type: application/json' "
               "-d @sample_input.json", pred),
              ("docker logs --tail 4 heart-api", logs)]
    lg = latest_log("run_all_")
    footer = ("(build and run executed by scripts/run_all.sh; log: "
              f"{os.path.basename(lg)})") if lg else None
    # drop the two action-only lines that show no output (keeps the image honest)
    blocks = [b for b in blocks if b[1]]
    terminal(browser, "04_docker_build_run.png", blocks, footer)


def shot_k8s(browser):
    log("[kubernetes evidence]")
    rc, live = run("kubectl get deployments,pods,svc -o wide", timeout=60)
    if rc == 0 and "heart-api" in live:
        blocks = [("kubectl get deployments,pods,svc -o wide", live)]
        rc2, hc = run("curl -s -m 10 http://$(minikube ip):$(kubectl get svc heart-api "
                      "-o jsonpath='{.spec.ports[0].nodePort}')/health", timeout=40)
        if rc2 == 0 and "ok" in hc:
            blocks.append(("curl http://<minikube-ip>:<nodePort>/health  # via the Service",
                           hc))
        terminal(browser, "05_k8s_deploy.png", blocks, footer="(live cluster output)")
        return
    lg = latest_log("k8s_deploy_")
    if not lg:
        results["05_k8s_deploy.png"] = ("FAILED (no cluster and no k8s log; "
                                        "run scripts/k8s_deploy.sh)")
        return
    text = ANSI.sub("", open(lg, errors="replace").read())
    start = text.find("=== [5]")
    end = text.find("K8S DEPLOY OK")
    if start < 0 or end < 0:
        results["05_k8s_deploy.png"] = "FAILED (k8s log incomplete)"
        return
    section = text[start:end + len("K8S DEPLOY OK")].rstrip()
    terminal(browser, "05_k8s_deploy.png", [("bash scripts/k8s_deploy.sh", section)],
             footer=f"(cluster not running now; output from {os.path.basename(lg)})")


# --------------------------------------------------------------------- web ----
def web_shot(browser, name, url, wait_text=None, settle_ms=2500, full=False,
             creds=None, width=1500, height=900, close_assistant=False, before_shot=None):
    log(f"[{name}] {url}")
    ctx = browser.new_context(viewport={"width": width, "height": height},
                              http_credentials=creds)
    page = ctx.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        status = "ok"
        if wait_text:
            try:
                page.get_by_text(re.compile(wait_text)).first.wait_for(timeout=30000)
            except Exception:
                status = f"captured, but text '{wait_text}' not seen (check image)"
        if close_assistant:  # MLflow 3 opens an "Assistant" side panel; hide it
            try:
                page.locator("[aria-label='Close']").first.click(timeout=4000)
            except Exception:
                pass
        page.wait_for_timeout(settle_ms)
        if before_shot:
            try:
                before_shot(page)
            except Exception as exc:
                log(f"  (optional UI step skipped: {str(exc)[:80]})")
        page.screenshot(path=os.path.join(SHOTS, name), full_page=full)
        results[name] = status
    except Exception as exc:
        results[name] = f"FAILED ({str(exc)[:100]})"
    finally:
        ctx.close()


def start_mlflow_ui():
    uri = mlflow.get_tracking_uri()
    env = dict(os.environ, MLFLOW_DISABLE_AGENT_HINT="1", MLFLOW_DISABLE_TELEMETRY="true")
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "ui", "--host", "127.0.0.1", "--port",
         str(MLFLOW_PORT), "--backend-store-uri", uri],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{MLFLOW_PORT}/health", timeout=2).ok:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    return None


def shot_mlflow(browser):
    exp = mlflow.get_experiment_by_name("heart-disease")
    if exp is None:
        for n in ("01_mlflow_runs.png", "02_mlflow_run_detail.png"):
            results[n] = "FAILED (no MLflow experiment; run src.train)"
        return
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                              order_by=["start_time DESC"])
    meta_path = os.path.join(ROOT, "models", "metadata.json")
    best = json.load(open(meta_path)).get("model") if os.path.exists(meta_path) else None
    pick = runs[runs["tags.mlflow.runName"] == best] if best else runs
    run_id = (pick if len(pick) else runs).iloc[0]["run_id"]
    proc = start_mlflow_ui()
    if proc is None:
        for n in ("01_mlflow_runs.png", "02_mlflow_run_detail.png"):
            results[n] = "FAILED (mlflow ui did not start)"
        return
    try:
        base = f"http://127.0.0.1:{MLFLOW_PORT}/#/experiments/{exp.experiment_id}"

        def show_metrics(page):
            page.get_by_text("Show more columns").first.click(timeout=4000)
            page.wait_for_timeout(2500)

        def open_roc(page):
            page.get_by_text("roc_curve.png").first.click(timeout=8000)
            page.wait_for_timeout(2500)

        web_shot(browser, "01_mlflow_runs.png", f"{base}/runs",
                 wait_text="logistic_regression|random_forest", settle_ms=3000,
                 width=1800, height=620, close_assistant=True,
                 before_shot=show_metrics)
        web_shot(browser, "02_mlflow_run_detail.png", f"{base}/runs/{run_id}",
                 wait_text="About this run", settle_ms=3000,
                 width=1500, height=1500, close_assistant=True)
        web_shot(browser, "02b_mlflow_artifacts.png", f"{base}/runs/{run_id}/artifacts",
                 wait_text="roc_curve.png", settle_ms=2500, width=1500, height=900,
                 close_assistant=True, before_shot=open_roc)
    finally:
        proc.terminate()


def shot_github(browser):
    url = f"https://github.com/{REPO}/actions"
    try:
        r = requests.get(f"https://api.github.com/repos/{REPO}/actions/runs",
                         params={"status": "success", "per_page": 1}, timeout=20)
        runs = r.json().get("workflow_runs", [])
        if runs:
            url = runs[0]["html_url"]
    except Exception:
        log("  GitHub API not reachable; capturing the Actions list instead")
    web_shot(browser, "03_github_actions.png", url, settle_ms=3000, full=True,
             width=1400, height=900)


def main():
    os.makedirs(SHOTS, exist_ok=True)
    with sync_playwright() as p:
        browser = launch(p)
        try:
            shot_train(browser)
            shot_docker(browser)
            shot_k8s(browser)
            shot_mlflow(browser)
            web_shot(browser, "06_prometheus_targets.png", f"{PROM_URL}/targets",
                     wait_text="heart-api", settle_ms=3000)
            web_shot(browser, "07_grafana_dashboard.png",
                     f"{GRAFANA_URL}/d/heart-api/heart-disease-api?orgId=1&from=now-15m"
                     "&to=now&kiosk", wait_text="Total predictions", settle_ms=9000,
                     creds={"username": "admin", "password": "admin"})
            shot_github(browser)
        finally:
            browser.close()
    print("\nScreenshot summary:")
    for name in sorted(results):
        print(f"  {name:28s} {results[name]}")
    if not any(v.startswith("ok") for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
