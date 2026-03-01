"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""
"""
Module: Data Cleaner

Role:
- Make raw data compliant so the pipeline behaves predictably.
- Keep it LIGHT: no scaling/encoding/feature engineering here.
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


# In Ames Housing, these missing values often mean "feature not present"
AMES_NONE_COLS = {
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill "None" for known NA-means-absent categorical columns (if they exist)
    none_cols = [c for c in AMES_NONE_COLS if c in df.columns]
    if none_cols:
        df[none_cols] = df[none_cols].fillna("None")

    # Numeric columns -> median
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical/object columns -> mode (most frequent) or "Unknown" if mode missing
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_vals = df[col].mode(dropna=True)
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    return df


def clean_housing_data(
    df: pd.DataFrame,
    *,
    target_col: str = "SalePrice",
    drop_cols: Optional[list[str]] = None,
) -> CleanResult:
    """
    Light cleaning for (Ames) housing-style tabular data.

    Args:
        df: raw dataframe
        target_col: target column name (present in train, absent in test)
        drop_cols: columns to drop if present (e.g., ["Id"])

    Returns:
        CleanResult(X, y, dropped_columns)

    Raises:
        DataCleanError: for schema issues that make cleaning unsafe
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise DataCleanError("Input must be a pandas DataFrame.")

    if df.empty:
        raise DataCleanError("Input DataFrame is empty.")

    df2 = _standardize_columns(df)

    # Drop duplicates (keep first)
    df2 = df2.drop_duplicates()

    dropped = []
    if drop_cols:
        for c in drop_cols:
            if c in df2.columns:
                df2 = df2.drop(columns=[c])
                dropped.append(c)

    y = None
    if target_col in df2.columns:
        y = df2[target_col].copy()
        df2 = df2.drop(columns=[target_col])

    df2 = _fill_missing_values(df2)

    # Final sanity checks
    if df2.isna().any().any():
        # If something still missing, stop early so validation can catch it explicitly
        remaining = df2.columns[df2.isna().any()].tolist()
        raise DataCleanError(f"Cleaning incomplete: still missing values in columns: {remaining}")

    return CleanResult(X=df2, y=y, dropped_columns=dropped)