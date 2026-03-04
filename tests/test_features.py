import pandas as pd
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer

from src.features import get_feature_preprocessor


def sample_dataframe():
    return pd.DataFrame({
        "num1": [1, 2, 3, np.nan],
        "num2": [10, 20, 30, 40],
        "cat1": ["A", "B", "A", None],
    })


def test_returns_column_transformer():
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["num1"],
        categorical_onehot_cols=["cat1"],
        numeric_passthrough_cols=["num2"],
    )
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_fits_and_transforms():
    df = sample_dataframe()

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["num1"],
        categorical_onehot_cols=["cat1"],
        numeric_passthrough_cols=["num2"],
    )

    X_transformed = preprocessor.fit_transform(df)

    # Should return numpy array
    assert isinstance(X_transformed, np.ndarray)

    # Should have same number of rows
    assert X_transformed.shape[0] == df.shape[0]


def test_empty_lists_do_not_crash():
    df = sample_dataframe()

    preprocessor = get_feature_preprocessor()

    X_transformed = preprocessor.fit_transform(df)

    # With remainder="drop" and no transformers, result should be empty array
    assert X_transformed.shape[0] == df.shape[0]