"""
Module: Data Cleaner

Role:
- Make raw data structurally safe for the pipeline.
- No imputation or feature engineering here.

Input: pandas.DataFrame (Raw)
Output: CleanResult(X, y, dropped_columns)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


class DataCleanError(RuntimeError):
    """Raised when data cannot be cleaned safely."""


@dataclass(frozen=True)
class CleanResult:
    X: pd.DataFrame
    y: Optional[pd.Series]
    dropped_columns: list[str]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )
    return df


def clean_housing_data(
    df: pd.DataFrame,
    *,
    target_col: str = "SalePrice",
    drop_cols: Optional[list[str]] = None,
    require_target: bool = False,
) -> CleanResult:
    if df is None or not isinstance(df, pd.DataFrame):
        raise DataCleanError("Input must be a pandas DataFrame.")
    if df.empty:
        raise DataCleanError("Input DataFrame is empty.")

    df2 = _standardize_columns(df)

    # Drop duplicates
    df2 = df2.drop_duplicates().reset_index(drop=True)

    dropped_columns: list[str] = []
    if drop_cols:
        existing = [c for c in drop_cols if c in df2.columns]
        if existing:
            df2 = df2.drop(columns=existing)
            dropped_columns.extend(existing)

    if require_target and target_col not in df2.columns:
        raise DataCleanError(f"Missing required target column: {target_col}")

    y: Optional[pd.Series] = None
    if target_col in df2.columns:
        y = df2[target_col].copy()
        df2 = df2.drop(columns=[target_col])

    return CleanResult(X=df2, y=y, dropped_columns=dropped_columns)