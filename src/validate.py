"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: DataFrame to validate
    - required_columns: List of columns that must exist
    Outputs:
    - True if valid, otherwise raises ValueError

    Why this contract matters for reliable ML delivery:
    - Prevents silent schema drift (broken training + incorrect inference).
    - Fail fast on obvious issues to avoid wasting compute and producing bad models.
    """
    logger.debug("Validating dataframe (fail fast)")

    # ----------------------------
    # Basic structural checks
    # ----------------------------
    if df is None:
        raise ValueError("Validation failed: df is None (upstream step did not return a DataFrame).")

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Validation failed: df must be a pandas DataFrame, got {type(df)}")

    if df.empty:
        raise ValueError("Validation failed: DataFrame is empty. Check ingestion and upstream filtering.")

    if required_columns is None:
        raise ValueError("Validation failed: required_columns is None.")

    if not isinstance(required_columns, list) or not all(isinstance(c, str) for c in required_columns):
        raise ValueError("Validation failed: required_columns must be a list of strings.")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Validation failed: Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # 1) Required columns should not contain nulls
    null_counts = df[required_columns].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        raise ValueError(
            "Validation failed: Null values found in required columns. "
            f"Null counts: {cols_with_nulls}"
        )

    # 2) Numeric sanity: no inf/-inf in numeric columns (common after preprocessing bugs)
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        has_inf = (numeric_cols == float("inf")).any().any() or (numeric_cols == float("-inf")).any().any()
        if has_inf:
            raise ValueError("Validation failed: Found inf/-inf in numeric columns.")

    # 3) Target check (your notebook uses SalePrice as y)
    if "SalePrice" in df.columns:
        if df["SalePrice"].isna().any():
            n_null = int(df["SalePrice"].isna().sum())
            raise ValueError(f"Validation failed: 'SalePrice' contains {n_null} nulls.")
        if (df["SalePrice"] <= 0).any():
            n_bad = int((df["SalePrice"] <= 0).sum())
            raise ValueError(f"Validation failed: 'SalePrice' has {n_bad} values <= 0 (must be positive).")

    # 4) Id check (Ames dataset typically includes Id; you drop it in the notebook)
    if "Id" in df.columns:
        if df["Id"].isna().any():
            raise ValueError("Validation failed: 'Id' contains nulls.")
        if df["Id"].duplicated().any():
            dup = int(df["Id"].duplicated().sum())
            raise ValueError(f"Validation failed: 'Id' must be unique, found {dup} duplicated Ids.")
        # Usually Id is positive
        if pd.api.types.is_numeric_dtype(df["Id"]) and (df["Id"] <= 0).any():
            raise ValueError("Validation failed: 'Id' contains values <= 0.")

    # 5) Optional: categorical column used in your EDA (Neighborhood)
    if "Neighborhood" in df.columns:
        if df["Neighborhood"].isna().any():
            n_null = int(df["Neighborhood"].isna().sum())
            raise ValueError(f"Validation failed: 'Neighborhood' contains {n_null} nulls.")
        # avoid empty strings
        if (df["Neighborhood"].astype(str).str.strip() == "").any():
            raise ValueError("Validation failed: 'Neighborhood' contains empty strings.")

    # 6) Optional: known house-quality constraints (only if these columns exist)
    if "OverallQual" in df.columns and pd.api.types.is_numeric_dtype(df["OverallQual"]):
        if (~df["OverallQual"].between(1, 10)).any():
            raise ValueError("Validation failed: 'OverallQual' must be between 1 and 10.")

    if "OverallCond" in df.columns and pd.api.types.is_numeric_dtype(df["OverallCond"]):
        if (~df["OverallCond"].between(1, 10)).any():
            raise ValueError("Validation failed: 'OverallCond' must be between 1 and 10.")

    if "YearBuilt" in df.columns and pd.api.types.is_numeric_dtype(df["YearBuilt"]):
        # keep it simple and future-proof
        if ((df["YearBuilt"] < 1800) | (df["YearBuilt"] > 2100)).any():
            raise ValueError("Validation failed: 'YearBuilt' has values outside [1800, 2100].")

    # 7) Optional: non-negative check for common area/size columns (only if present)
    nonneg_like = ["LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageArea"]
    for c in nonneg_like:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            if (df[c] < 0).any():
                n_bad = int((df[c] < 0).sum())
                raise ValueError(f"Validation failed: '{c}' has {n_bad} negative values (must be >= 0).")

    logger.debug("Validation OK")
    return True

