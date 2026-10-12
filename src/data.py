"""Data loading, cleaning and preprocessing pipeline."""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                        "processed.cleveland.data")
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                          "heart_clean.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize target (0 = no disease, 1-4 -> 1) and drop duplicates."""
    df = df.copy()
    df["target"] = (df["target"] > 0).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC),
                              ("cat", cat, CATEGORICAL)])


def load_clean() -> pd.DataFrame:
    df = clean(load_raw())
    df.to_csv(CLEAN_PATH, index=False)
    return df


def split_xy(df: pd.DataFrame):
    return df[FEATURES], df["target"].to_numpy(dtype=np.int64)
