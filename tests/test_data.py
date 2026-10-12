import numpy as np
import pandas as pd

from src.data import COLUMNS, FEATURES, build_preprocessor, clean


def _sample(n=20):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(30, 70, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 300, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(100, 190, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.random(n) * 3,
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n), "target": rng.integers(0, 5, n),
    })[COLUMNS]
    df.loc[0, "ca"] = np.nan
    return df


def test_clean_binarizes_target():
    out = clean(_sample())
    assert set(out["target"].unique()) <= {0, 1}


def test_preprocessor_handles_missing_and_shape():
    df = clean(_sample())
    X = build_preprocessor().fit_transform(df[FEATURES])
    assert not np.isnan(np.asarray(X.todense() if hasattr(X, "todense")
                                   else X)).any()
    assert X.shape[0] == len(df)
