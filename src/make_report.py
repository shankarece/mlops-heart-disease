"""Build the assignment report (.docx) from real pipeline results.

Reads MLflow runs, EDA outputs and dataset statistics, embeds plots and any
screenshots found in screenshots/, and marks missing screenshots clearly.
Run:  python -m src.make_report
"""
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

from src.data import FEATURES, NUMERIC, load_clean, load_raw  # noqa: E402

# ---- edit these two lines if needed ---------------------------------------
STUDENT_NAME = "Siva Sankar K"
STUDENT_ID = "s2025ae05092"
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/shankarece/mlops-heart-disease"
ROOT = os.path.join(os.path.dirname(__file__), "..")
EDA = os.path.join(ROOT, "reports", "eda")
SHOTS = os.path.join(ROOT, "screenshots")
OUT = os.path.join(ROOT, "reports", "MLOps_Assignment1_Report.docx")
missing = []


def p(doc, text, bold=False, italic=False):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold, run.italic = bold, italic
    return para


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def code(doc, text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9)


def table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    doc.add_paragraph()


def picture(doc, path, caption, width=6.0):
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.runs[0].italic = True


def shot(doc, filename, caption):
    path = os.path.join(SHOTS, filename)
    if os.path.exists(path):
        picture(doc, path, caption)
    else:
        missing.append(filename)
        para = doc.add_paragraph()
        run = para.add_run(f"[INSERT SCREENSHOT: {filename} - {caption}]")
        run.bold = True
        run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)


def draw_architecture(path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    w, h = 0.12, 0.22
    top, bot = 0.62, 0.16
    cols = [0.02, 0.19, 0.36, 0.53, 0.70, 0.86]
    boxes = {
        "uci": (cols[0], top, "UCI dataset"),
        "clean": (cols[1], top, "Clean +\npreprocess"),
        "train": (cols[2], top, "Train LR / RF\n(CV + grid)"),
        "model": (cols[3], top, "Model\npipeline"),
        "api": (cols[4], top, "FastAPI\n/predict"),
        "mon": (cols[5], top, "Prometheus\n+ Grafana"),
        "mlf": (cols[2], bot, "MLflow\ntracking"),
        "ci": (cols[3], bot, "GitHub Actions\nCI"),
        "dock": (cols[4], bot, "Docker\nimage"),
        "k8s": (cols[5], bot, "Kubernetes\n(LoadBalancer)"),
    }
    for x, y, label in boxes.values():
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#dbe9f6", ec="#2b5c8a", lw=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def arrow(a, b, style="-"):
        xa, ya, _ = boxes[a]
        xb, yb, _ = boxes[b]
        if ya == yb:  # horizontal: right edge of a -> left edge of b
            p1, p2 = (xa + w, ya + h / 2), (xb, yb + h / 2)
        else:         # vertical: bottom of a -> top of b
            p1, p2 = (xa + w / 2, ya), (xb + w / 2, yb + h)
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(
            arrowstyle="->", lw=1.4, color="#333", linestyle=style))

    for a, b in [("uci", "clean"), ("clean", "train"), ("train", "model"),
                 ("model", "api"), ("api", "mon"), ("train", "mlf"),
                 ("api", "dock"), ("dock", "k8s")]:
        arrow(a, b)
    arrow("ci", "dock", "--")
    ax.text(cols[3] + w + 0.025, bot + h / 2 + 0.04, "builds", fontsize=8,
            ha="center", color="#555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("System architecture", fontsize=12)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def mlflow_runs():
    mlflow.set_experiment("heart-disease")
    exp = mlflow.get_experiment_by_name("heart-disease")
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                            order_by=["start_time DESC"])
    if df.empty:
        sys.exit("No MLflow runs found. Run:  python -m src.train")
    return df, df.drop_duplicates(subset="tags.mlflow.runName")


def main():
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    if not os.path.isdir(EDA):
        sys.exit("EDA outputs missing. Run:  python -m src.eda")
    raw = load_raw()
    df = load_clean()
    all_runs, latest = mlflow_runs()
    meta_path = os.path.join(ROOT, "models", "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    n_tests = sum(len(re.findall(r"^def test_", open(f).read(), re.M))
                  for f in glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    pos = int(df["target"].sum())
    neg = len(df) - pos
    arch = os.path.join(ROOT, "reports", "architecture.png")
    draw_architecture(arch)

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    t = doc.add_heading("MLOps Assignment 01", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in ("End-to-End ML Model Development, CI/CD and Production Deployment",
                 "Course: Machine Learning Operations (MLOps) AIMLCZG523, BITS Pilani",
                 f"Student: {STUDENT_NAME} ({STUDENT_ID})",
                 f"Code repository: {REPO_URL}"):
        c = doc.add_paragraph(line)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    doc.add_heading("1. Problem statement and approach", 1)
    p(doc, "For this assignment I built a classifier that predicts whether a patient "
           "has heart disease from routine clinical measurements, and deployed it as "
           "a monitored API on Kubernetes. I tried to treat it like a small "
           "production project rather than a notebook exercise: the pipeline is "
           "scripted from end to end, every training run is tracked, the code is "
           "tested and linted automatically, and the service runs in a container.")
    bullets(doc, [
        "Data: the UCI Heart Disease (Cleveland) dataset, fetched by a download "
        "script.",
        "Models: logistic regression and random forest, tuned with grid search and "
        "5-fold stratified cross-validation, tracked in MLflow.",
        "Serving: FastAPI with /predict, /health and /metrics, packaged in Docker.",
        "Delivery: GitHub Actions (lint, tests, training, Docker build) and a "
        "Kubernetes Deployment with a LoadBalancer Service on Minikube.",
        "Observability: request logging, Prometheus metrics and a Grafana dashboard.",
    ])

    doc.add_heading("2. Setup and installation", 1)
    p(doc, "Everything can be run from a fresh clone. I worked in the BITS Prayogshala "
           "virtual lab (Docker available, Python 3.12). Because the lab session "
           "resets when it is idle, I wrapped each stage in a script that checks its "
           "prerequisites first, so the whole setup can be rebuilt with a few "
           "commands:")
    code(doc, f"git clone {REPO_URL}.git\ncd mlops-heart-disease\n"
              "bash scripts/run_all.sh        # install, data, EDA, train, lint, tests, "
              "docker build/run, API smoke test\n"
              "bash scripts/monitoring_up.sh  # Prometheus + Grafana\n"
              "bash scripts/k8s_deploy.sh     # Minikube deployment")
    p(doc, "Each script stops with a clear ERROR message if something is wrong and "
           "writes a log file to logs/. The manual equivalents are:")
    code(doc, "pip install -r requirements.txt\npython -m src.download_data\n"
              "python -m src.eda\npython -m src.train\nmlflow ui\npython -m pytest -v\n"
              "docker build -t heart-api:latest .\n"
              "docker run -d -p 8000:8000 --name heart-api heart-api:latest")

    doc.add_heading("3. Data acquisition and exploratory data analysis", 1)
    p(doc, f"The dataset has {raw.shape[0]} patients, {len(FEATURES)} features and a "
           "target column. src/download_data.py downloads processed.cleveland.data "
           "from the UCI repository so nobody has to fetch it by hand. In "
           "src/data.py I turn the original 0-4 target into a binary label "
           "(0 = no disease, 1-4 = disease), drop duplicate rows and save the "
           "cleaned table as data/heart_clean.csv.")
    miss = raw.isna().sum()
    miss = miss[miss > 0]
    p(doc, "Missing values: " + (", ".join(f"{k} ({v})" for k, v in miss.items())
                                 if len(miss) else "none") +
      ". I left them in the data and impute inside the model pipeline (median for "
      "numeric columns, most frequent for categorical ones). That way the imputation "
      "values are learned from the training folds only, and the same rule is "
      "applied when the API receives a request.")
    p(doc, f"After cleaning there are {pos} disease and {neg} no-disease cases "
           f"({100 * pos / len(df):.1f}% positive), so the classes are close to "
           "balanced and accuracy is a fair metric. I still report precision, recall "
           "and ROC-AUC, because missing a sick patient is the costly mistake here.")
    for img, cap in [("histograms.png", "Figure 1: Distributions of numeric features"),
                     ("correlation_heatmap.png", "Figure 2: Correlation heatmap"),
                     ("class_balance.png", "Figure 3: Class balance"),
                     ("features_by_target.png", "Figure 4: Numeric features by target")]:
        path = os.path.join(EDA, img)
        if os.path.exists(path):
            picture(doc, path, cap, 5.2 if img == "class_balance.png" else 6.0)
    corr = df.corr()["target"].drop("target").sort_values(key=abs, ascending=False)
    p(doc, "Features most correlated with the target: " +
      ", ".join(f"{k} ({v:+.2f})" for k, v in corr.head(5).items()) + ".")

    doc.add_heading("4. Feature engineering and model development", 1)
    p(doc, "Preprocessing is a scikit-learn ColumnTransformer. The numeric features "
           f"({', '.join(NUMERIC)}) are median-imputed and standardised; the "
           "categorical features are imputed with the most frequent value and "
           "one-hot encoded, ignoring unseen categories at inference time. I fit the "
           "transformer and the classifier together as a single Pipeline, so the API "
           "applies exactly the same preprocessing as training.")
    p(doc, "I used an 80/20 stratified train/test split with seed 42. For each model "
           "I ran GridSearchCV with 5-fold stratified cross-validation on the "
           "training part, scored on ROC-AUC, then cross-validated the best setting "
           "for the full set of metrics and scored it once on the held-out test set. "
           "I chose the model to deploy using cross-validation AUC only, so the test "
           "set stayed an unbiased final check.")
    table(doc, ["Model", "Search space"],
          [["Logistic Regression", "C in {0.01, 0.1, 1, 10}"],
           ["Random Forest", "n_estimators in {100, 300}; max_depth in {3, 5, None}"]])
    rows = []
    for _, r in latest.iterrows():
        rows.append([r["tags.mlflow.runName"].replace("_", " ").title()] + [
            f"{r.get('metrics.' + m, float('nan')):.3f}"
            for m in ("cv_accuracy", "cv_precision", "cv_recall",
                      "cv_roc_auc", "test_roc_auc")])
    table(doc, ["Model", "CV accuracy", "CV precision", "CV recall",
                "CV ROC-AUC", "Test ROC-AUC"], rows)
    if meta:
        cvs = {r["tags.mlflow.runName"]: r.get("metrics.cv_roc_auc")
               for _, r in latest.iterrows()}
        other = [k for k in cvs if k != meta["model"]]
        txt = (f"I selected {meta['model'].replace('_', ' ')} "
               f"(CV ROC-AUC {meta['cv_roc_auc']:.3f}, held-out test ROC-AUC "
               f"{meta['test_roc_auc']:.3f}).")
        if other:
            gap = meta["cv_roc_auc"] - cvs[other[0]]
            txt += (f" It is ahead of {other[0].replace('_', ' ')} by {gap:.3f} CV "
                    "ROC-AUC")
            if gap < 0.02 and meta["model"] == "logistic_regression":
                txt += (", which is within noise, so I preferred the simpler and "
                        "more interpretable linear model.")
            else:
                txt += "."
        p(doc, txt)
    shot(doc, "08_train_output.png", "Figure 5: Training output")

    doc.add_heading("5. Experiment tracking (MLflow)", 1)
    p(doc, "Every run of the training script is logged to MLflow under the experiment "
           "'heart-disease': the best hyper-parameters, the cross-validated metrics, "
           "the test ROC-AUC, the ROC curve and confusion matrix plots, and the "
           "fitted model. I ran the script several times while developing, so the "
           "MLflow UI shows more than two runs; the tables in this report use the "
           "latest run of each model.")
    prm = [c for c in latest.columns if c.startswith("params.")]
    rows = []
    for _, r in latest.iterrows():
        used = {c.replace("params.", ""): r[c] for c in prm
                if str(r[c]) not in ("None", "nan")}
        name = r["tags.mlflow.runName"].replace("_", " ").title()
        rows.append([name, ", ".join(f"{k}={v}" for k, v in used.items())])
    table(doc, ["Run", "Best parameters"], rows)
    shot(doc, "01_mlflow_runs.png",
         "Figure 6: MLflow comparison of the two models "
         "(left column: random forest, right column: logistic regression)")
    shot(doc, "02_mlflow_run_detail.png", "Figure 7: MLflow run with parameters, metrics and plots")

    doc.add_heading("6. Model packaging and reproducibility", 1)
    bullets(doc, [
        "I saved the final model as the complete Pipeline (preprocessing plus "
        "classifier) with joblib in models/model.joblib, and also logged it to MLflow.",
        "All dependencies are listed in requirements.txt and every script installs "
        "from it.",
        "I fixed the random seed (42) for the split, the cross-validation and the "
        "forest.",
        "The Docker image downloads the data and trains the model while it is being "
        "built, so training and serving always use the same library versions (see "
        "section 8 for why).",
    ])

    doc.add_heading("7. CI/CD pipeline and automated testing", 1)
    p(doc, f"I wrote {n_tests} pytest tests. They cover the data cleaning (target "
           "binarisation), the preprocessing pipeline (missing values and output "
           "shape), model training and probability output, and the API (health, "
           "metrics, prediction and input validation).")
    p(doc, "The GitHub Actions workflow (.github/workflows/ci.yml) runs on every push "
           "and pull request. It installs the dependencies, runs flake8, downloads "
           "the data, trains the model, runs pytest, builds the Docker image and "
           "uploads the test report, the model and the cleaned data as artifacts. It "
           "fails on any lint, test or training error, and the logs of every run are "
           "kept.")
    p(doc, "The very first run failed: the newer MLflow version I had installed would "
           "not save the scikit-learn pipeline in its default format. I reproduced "
           "the error locally, switched the model logging to the cloudpickle format, "
           "and the next run passed. That is the kind of problem I wanted the "
           "pipeline to catch.")
    shot(doc, "03_github_actions.png",
         "Figure 8: GitHub Actions run history "
         "(the latest runs pass; the first run failed and was fixed in run 2)")

    doc.add_heading("8. Containerisation", 1)
    p(doc, "The Dockerfile is based on python:3.12-slim. It installs the "
           "requirements, copies the source, trains the model inside the image and "
           "starts uvicorn on port 8000. POST /predict takes the 13 patient features "
           "as JSON and returns the predicted class and a confidence value (the "
           "probability of disease); invalid input returns HTTP 422. Example request:")
    code(doc, 'curl -X POST localhost:8000/predict -H "Content-Type: application/json" '
              "-d @sample_input.json\n"
              '-> {"prediction": ..., "confidence": ...}')
    p(doc, "I ran into one problem here. My first image used Python 3.10, but the "
           "model had been trained with Python 3.12 and a newer scikit-learn, and "
           "/predict returned HTTP 500 inside the container. Training the model "
           "during the image build removed that mismatch.")
    shot(doc, "04_docker_build_run.png",
         "Figure 9: Running containers with /health and /predict responses")

    doc.add_heading("9. Production deployment (Kubernetes)", 1)
    p(doc, "I deployed the API on a local Minikube cluster (Docker driver). "
           "k8s/deployment.yaml runs 2 replicas with readiness and liveness probes on "
           "/health plus resource requests and limits, and k8s/service.yaml exposes "
           "them through a Service of type LoadBalancer (port 80 to container port "
           "8000). scripts/k8s_deploy.sh starts the cluster, loads the image, applies "
           "the manifests, waits for the rollout and checks /health and /predict "
           "through the Service. On Minikube the EXTERNAL-IP stays 'pending' until "
           "'minikube tunnel' is running, which is expected.")
    p(doc, "Access: the API is not on a public URL. To try it locally, run "
           "scripts/run_all.sh (the container listens on http://localhost:8000 and "
           "the interactive docs are at /docs) or scripts/k8s_deploy.sh (Kubernetes, "
           "then 'minikube service heart-api --url'). The same model can also be "
           "scored from the command line with "
           "'python -m src.predict sample_input.json'.")
    shot(doc, "05_k8s_deploy.png", "Figure 10: Kubernetes rollout and resources")

    doc.add_heading("10. Monitoring and logging", 1)
    p(doc, "Every request to the API is logged (input and result), and failures are "
           "logged with a stack trace. The /metrics endpoint exposes Prometheus "
           "metrics: predict_requests_total, labelled by status, and a latency "
           "histogram, predict_latency_seconds. The docker-compose file in monitoring/ "
           "starts Prometheus, which scrapes the API every 10 seconds, and Grafana "
           "with a provisioned datasource and a 'Heart Disease API' dashboard (total "
           "predictions, errors, request rate and p95 latency). "
           "scripts/monitoring_up.sh sends test traffic and checks that Prometheus "
           "received it.")
    shot(doc, "06_prometheus_targets.png", "Figure 11: Prometheus target UP")
    shot(doc, "07_grafana_dashboard.png", "Figure 12: Grafana dashboard")

    doc.add_heading("11. Architecture", 1)
    picture(doc, arch, "Figure 13: System architecture", 6.3)

    doc.add_heading("12. Conclusion and links", 1)
    p(doc, "The result is a reproducible path from raw data to a monitored model "
           "service on Kubernetes, with lint, tests and training checked automatically "
           "on every push. If I carried on with this project, I would add a model "
           "registry with promotion rules, data-drift monitoring and automated "
           "retraining.")
    p(doc, f"Code repository: {REPO_URL}")
    doc.save(OUT)
    print(f"Report written to {os.path.abspath(OUT)}")
    if missing:
        print("Screenshots not found in screenshots/ (placeholders inserted):")
        for m in missing:
            print("  -", m)


if __name__ == "__main__":
    main()
