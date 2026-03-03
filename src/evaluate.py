"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.Series, list]


def _to_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    if x is None:
        raise ValueError(f"Evaluation failed: {name} is None.")

    if isinstance(x, pd.DataFrame):
        # common mistake: passing a (n,1) df
        if x.shape[1] != 1:
            raise ValueError(f"Evaluation failed: {name} is a DataFrame with shape {x.shape}; expected 1 column.")
        x = x.iloc[:, 0]

    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Evaluation failed: {name} is empty.")
    return arr


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike, *, compute_rmsle: bool = True) -> Dict[str, float]:
    """
    Inputs:
    - y_true: ground truth targets
    - y_pred: model predictions
    Outputs:
    - metrics: dict with rmse, mae, r2 (and rmsle if applicable)

    Why this contract matters:
    - Standard metrics allow reliable comparisons between models/runs and catch regressions early.
    """
    print("[evaluate.evaluate_regression] Evaluating predictions")  # TODO: replace with logging later

    y_true_arr = _to_1d_array(y_true, "y_true")
    y_pred_arr = _to_1d_array(y_pred, "y_pred")

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"Evaluation failed: length mismatch y_true={y_true_arr.shape[0]} vs y_pred={y_pred_arr.shape[0]}"
        )

    # NaN checks
    if np.isnan(y_true_arr).any():
        n = int(np.isnan(y_true_arr).sum())
        raise ValueError(f"Evaluation failed: y_true contains {n} NaN values.")
    if np.isnan(y_pred_arr).any():
        n = int(np.isnan(y_pred_arr).sum())
        raise ValueError(f"Evaluation failed: y_pred contains {n} NaN values.")

    # Core metrics
    residuals = y_true_arr - y_pred_arr
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    # R^2
    y_mean = float(np.mean(y_true_arr))
    ss_tot = float(np.sum((y_true_arr - y_mean) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

    metrics: Dict[str, float] = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    # Optional RMSLE (useful for SalePrice; requires positives)
    if compute_rmsle:
        if (y_true_arr <= 0).any() or (y_pred_arr <= 0).any():
            print("[evaluate.evaluate_regression] Skipping RMSLE: requires y_true>0 and y_pred>0")
        else:
            log_true = np.log1p(y_true_arr)
            log_pred = np.log1p(y_pred_arr)
            rmsle = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
            metrics["rmsle"] = rmsle

    print(f"[evaluate.evaluate_regression] Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    y_true = [100000, 150000, 200000]
    y_pred = [110000, 140000, 210000]
    print(evaluate_regression(y_true, y_pred, compute_rmsle=True))