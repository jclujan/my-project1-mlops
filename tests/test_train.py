import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pytest

from src.train import train_model


def simple_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="mean")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["feature1", "feature2"],
            )
        ]
    )


def sample_regression_data():
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
        }
    )
    y = pd.Series([100, 200, 300, 400, 500])
    return X, y


def sample_classification_data():
    # Need at least 5 samples PER CLASS for StratifiedKFold(n_splits=5)
    X = pd.DataFrame(
        {
            "feature1": list(range(1, 11)),
            "feature2": [10 * i for i in range(1, 11)],
        }
    )
    y = pd.Series([0, 1] * 5)  # 5 zeros, 5 ones
    return X, y


def test_train_regression_runs():
    X, y = sample_regression_data()
    model = train_model(X, y, simple_preprocessor(), problem_type="regression")
    assert hasattr(model, "best_estimator_")


def test_train_classification_runs():
    X, y = sample_classification_data()
    model = train_model(X, y, simple_preprocessor(), problem_type="classification")
    assert hasattr(model, "best_estimator_")


def test_invalid_problem_type():
    X, y = sample_regression_data()
    with pytest.raises(ValueError):
        train_model(X, y, simple_preprocessor(), problem_type="invalid")