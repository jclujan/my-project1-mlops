import numpy as np
import pytest

from src.evaluate import evaluate_regression


def test_evaluate_returns_metrics():
    y_true = [100000, 150000, 200000]
    y_pred = [110000, 140000, 210000]
    metrics = evaluate_regression(y_true, y_pred, compute_rmsle=True)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "rmsle" in metrics


def test_evaluate_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_regression([1, 2, 3], [1, 2], compute_rmsle=False)


def test_evaluate_nan_raises():
    y_true = np.array([1.0, np.nan, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        evaluate_regression(y_true, y_pred)
        