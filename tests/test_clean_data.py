import pandas as pd
import pytest

from src.clean_data import clean_housing_data, DataCleanError


def test_clean_removes_duplicates():
    df = pd.DataFrame({
        "Id": [1, 1],
        "SalePrice": [100000, 100000],
    })

    result = clean_housing_data(df, drop_cols=["Id"], require_target=True)

    assert result.X.shape[0] == 1
    assert "Id" not in result.X.columns


def test_clean_extracts_target():
    df = pd.DataFrame({
        "FeatureA": [1, 2],
        "SalePrice": [100000, 200000],
    })

    result = clean_housing_data(df, require_target=True)

    assert result.y is not None
    assert result.y.tolist() == [100000, 200000]
    assert "SalePrice" not in result.X.columns


def test_clean_raises_if_target_missing_when_required():
    df = pd.DataFrame({
        "FeatureA": [1, 2],
    })

    with pytest.raises(DataCleanError):
        clean_housing_data(df, require_target=True)


def test_clean_standardizes_column_names():
    df = pd.DataFrame({
        "  Lot Area  ": [1000],
        "SalePrice": [200000],
    })

    result = clean_housing_data(df, require_target=True)

    assert "Lot_Area" in result.X.columns


def test_clean_allows_missing_values():
    df = pd.DataFrame({
        "SalePrice": [100000],
        "LotArea": [None],  # cleaning should NOT fix this anymore
    })

    result = clean_housing_data(df, require_target=True)

    assert result.X.isna().any().any()  # missing values should still exist