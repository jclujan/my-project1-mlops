import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.train import train_model


def simple_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="mean")),
                    ("scale", StandardScaler())
                ]),
                ["feature1", "feature2"]
            )
        ]
    )


def sample_regression_data():
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    })
    y = pd.Series([100, 200, 300, 400, 500])
    return X, y


def sample_classification_data():
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    })
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y


def test_train_regression_runs():
    X, y = sample_regression_data()
    model = train_model(X, y, simple_preprocessor(), problem_type="regression")

    # GridSearchCV has best_estimator_ after fitting
    assert hasattr(model, "best_estimator_")


def test_train_classification_runs():
    X, y = sample_classification_data()
    model = train_model(X, y, simple_preprocessor(), problem_type="classification")

    assert hasattr(model, "best_estimator_")


def test_invalid_problem_type():
    X, y = sample_regression_data()

    try:
        train_model(X, y, simple_preprocessor(), problem_type="invalid")
    except ValueError:
        assert True