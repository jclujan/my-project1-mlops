"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import joblib

from src.utils import ensure_parent_dir


def load_artifact(model_path: str) -> Dict[str, Any]:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact or "metadata" not in artifact:
        raise TypeError("Model artifact must be a dict with keys: 'pipeline' and 'metadata'.")
    return artifact


def run_inference(
    input_df: pd.DataFrame,
    artifact: Dict[str, Any],
    id_col: str = "Id",
    pred_col: str = "SalePrice",
) -> pd.DataFrame:
    pipeline = artifact["pipeline"]
    meta = artifact["metadata"]

    # Never require Id. Keep if present.
    ids = input_df[id_col].copy() if id_col in input_df.columns else None

    # Predict in log space then invert
    preds_log = pipeline.predict(input_df)
    preds = np.expm1(np.asarray(preds_log).reshape(-1)) if meta.get("target_transform") == "log1p" else np.asarray(preds_log).reshape(-1)

    out = pd.DataFrame({pred_col: preds})
    if ids is not None:
        out.insert(0, id_col, ids.values)
    return out


def predict_csv(input_path: str, model_path: str, output_path: str, id_col: str = "Id", pred_col: str = "SalePrice") -> None:
    df = pd.read_csv(input_path)
    artifact = load_artifact(model_path)
    out = run_inference(df, artifact, id_col=id_col, pred_col=pred_col)

    ensure_parent_dir(output_path)
    out.to_csv(output_path, index=False)