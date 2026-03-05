import pandas as pd
import pytest

from src.validate import validate_dataframe


def test_validate_ok_minimal():
    df = pd.DataFrame({"Id": [1, 2], "SalePrice": [100000, 150000]})
    assert validate_dataframe(df, required_columns=["Id", "SalePrice"]) is True


def test_validate_empty_df_raises():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Id", "SalePrice"])


def test_validate_missing_required_columns_raises():
    df = pd.DataFrame({"Id": [1, 2]})
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Id", "SalePrice"])


def test_validate_null_in_required_raises():
    df = pd.DataFrame({"Id": [1, None], "SalePrice": [100000, 150000]})
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Id", "SalePrice"])
        