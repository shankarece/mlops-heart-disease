"""Train models, track with MLflow, save the best full pipeline."""
import json
import os

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402,E501
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402,E501
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data import build_preprocessor, load_clean, split_xy  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SEED = 42

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None]},
    ),
}
SCORING = ["accuracy", "precision", "recall", "roc_auc"]


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("heart-disease")
    X, y = split_xy(load_clean())
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    best_name, best_cv, best_test, best_model = None, -1.0, 0.0, None
    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc")
            search.fit(X_tr, y_tr)
            model = search.best_estimator_
            cvres = cross_validate(model, X_tr, y_tr, cv=cv, scoring=SCORING)
            proba = model.predict_proba(X_te)[:, 1]
            test_auc = roc_auc_score(y_te, proba)

            mlflow.log_params(search.best_params_)
            for m in SCORING:
                mlflow.log_metric(f"cv_{m}", float(np.mean(cvres[f"test_{m}"])))
            mlflow.log_metric("test_roc_auc", test_auc)

            for plot, disp in (
                ("roc_curve.png",
                 lambda: RocCurveDisplay.from_estimator(model, X_te, y_te)),
                ("confusion_matrix.png",
                 lambda: ConfusionMatrixDisplay.from_estimator(
                     model, X_te, y_te)),
            ):
                disp()
                plt.savefig(plot, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(plot)
                os.remove(plot)
            mlflow.sklearn.log_model(
                model, "model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
            print(f"{name}: cv_auc={np.mean(cvres['test_roc_auc']):.3f} "
                  f"test_auc={test_auc:.3f}")
            cv_auc = float(np.mean(cvres["test_roc_auc"]))
            # select on cross-validation AUC; the test set is reported only
            if cv_auc > best_cv:
                best_name, best_cv, best_test, best_model = (
                    name, cv_auc, test_auc, model)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.joblib"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({"model": best_name, "cv_roc_auc": best_cv,
                   "test_roc_auc": best_test}, f)
    print(f"Saved best model: {best_name}")


if __name__ == "__main__":
    main()
