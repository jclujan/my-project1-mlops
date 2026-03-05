import numpy as np
import pandas as pd
import pytest

from src.validate import validate_dataframe


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_df():
    """Minimal valid DataFrame that passes all checks."""
    return pd.DataFrame({
        "LotArea":     [8450, 9600, 11250],
        "GrLivArea":   [1710, 1262, 1786],
        "OverallQual": [5, 7, 4],
        "YearBuilt":   [1960, 2003, 1920],
        "Neighborhood": ["NAmes", "CollgCr", "OldTown"],
    })


# ── Structural checks ─────────────────────────────────────────────────────────

def test_valid_dataframe_passes():
    assert validate_dataframe(_base_df(), ["LotArea", "GrLivArea"]) is True


def test_none_dataframe_raises():
    with pytest.raises(ValueError, match="None"):
        validate_dataframe(None, ["LotArea"])


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(pd.DataFrame(), ["LotArea"])


def test_missing_required_column_raises():
    df = _base_df().drop(columns=["LotArea"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df, ["LotArea"])


def test_none_required_columns_raises():
    with pytest.raises(ValueError, match="required_columns is None"):
        validate_dataframe(_base_df(), None)


def test_required_columns_not_list_raises():
    with pytest.raises(ValueError, match="list of strings"):
        validate_dataframe(_base_df(), "LotArea")


# ── Null checks ───────────────────────────────────────────────────────────────

def test_null_in_required_column_raises():
    df = _base_df().copy()
    df.loc[0, "LotArea"] = None
    with pytest.raises(ValueError, match="Null values"):
        validate_dataframe(df, ["LotArea"])


# ── Infinity checks ───────────────────────────────────────────────────────────

def test_inf_in_numeric_column_raises():
    df = _base_df().copy()
    df["LotArea"] = df["LotArea"].astype(float)
    df.loc[0, "LotArea"] = float("inf")
    with pytest.raises(ValueError, match="inf"):
        validate_dataframe(df, ["GrLivArea"])


# ── Domain constraint checks ──────────────────────────────────────────────────

def test_overall_qual_out_of_range_raises():
    df = _base_df().copy()
    df.loc[0, "OverallQual"] = 11  # max is 10
    with pytest.raises(ValueError, match="OverallQual"):
        validate_dataframe(df, [])


def test_year_built_too_old_raises():
    df = _base_df().copy()
    df.loc[0, "YearBuilt"] = 1700  # below 1800
    with pytest.raises(ValueError, match="YearBuilt"):
        validate_dataframe(df, [])


def test_lot_area_negative_raises():
    df = _base_df().copy()
    df.loc[0, "LotArea"] = -1
    with pytest.raises(ValueError, match="LotArea"):
        validate_dataframe(df, [])


def test_neighborhood_empty_string_raises():
    df = _base_df().copy()
    df.loc[0, "Neighborhood"] = "   "
    with pytest.raises(ValueError, match="Neighborhood"):
        validate_dataframe(df, [])
