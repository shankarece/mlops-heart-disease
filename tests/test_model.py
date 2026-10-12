from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import FEATURES, build_preprocessor, clean
from tests.test_data import _sample


def test_pipeline_trains_and_predicts_probabilities():
    df = clean(_sample(60))
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(df[FEATURES], df["target"])
    proba = pipe.predict_proba(df[FEATURES])
    assert proba.shape == (len(df), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
