import pandas as pd
from src.clean_data import clean_housing_data


def test_clean_removes_duplicates():
    df = pd.DataFrame({
        "Id": [1, 1],
        "SalePrice": [100000, 100000],
    })

    result = clean_housing_data(df, drop_cols=["Id"], require_target=True)

    assert result.X.shape[0] == 1


def test_clean_fills_missing_numeric():
    df = pd.DataFrame({
        "SalePrice": [100000],
        "LotArea": [None],
    })

    result = clean_housing_data(df, require_target=True)

    assert not result.X.isna().any().any()