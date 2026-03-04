"""
Module: Data Cleaner

Role:
- Make raw data compliant so the pipeline behaves predictably.
- Keep it LIGHT: no scaling/encoding/feature engineering here.

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


# In Ames Housing, these missing values often mean "feature not present"
AMES_NONE_COLS = {
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
}

# In Ames Housing, NA in these numeric columns often means "feature not present" -> fill 0
AMES_ZERO_COLS = {
    "GarageYrBlt",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "GarageArea",
    "GarageCars",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )
    return df


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Categorical NA means "None"
    none_cols = [c for c in AMES_NONE_COLS if c in df.columns]
    if none_cols:
        df[none_cols] = df[none_cols].fillna("None")

    # 2) Numeric NA means "feature absent" -> fill 0
    zero_cols = [c for c in AMES_ZERO_COLS if c in df.columns]
    if zero_cols:
        df[zero_cols] = df[zero_cols].fillna(0)

    # 3) Remaining numeric -> median
    num_cols = df.select_dtypes(include=["number"]).columns
    remaining_num_cols = [c for c in num_cols if c not in AMES_ZERO_COLS]
    for col in remaining_num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 4) Remaining categorical/object -> mode (or "Unknown")
    cat_cols = df.select_dtypes(include=["object"]).columns
    remaining_cat_cols = [c for c in cat_cols if c not in AMES_NONE_COLS]
    for col in remaining_cat_cols:
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
    require_target: bool = False,
) -> CleanResult:
    """
    Light cleaning for (Ames) housing-style tabular data.

    Args:
        df: raw dataframe
        target_col: target column name (present in train, absent in test)
        drop_cols: columns to drop if present (e.g., ["Id"])
        require_target: if True, raise error if target_col is missing

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

    # Drop duplicates + reset index (safe + prevents misalignment later)
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

    df2 = _fill_missing_values(df2)

    # Final sanity check: no missing values left
    if df2.isna().any().any():
        remaining = df2.columns[df2.isna().any()].tolist()
        raise DataCleanError(f"Cleaning incomplete: still missing values in columns: {remaining}")

    return CleanResult(X=df2, y=y, dropped_columns=dropped_columns)