"""
Educational Goal:
- Why this module exists in an MLOps system: Feature transformations must be reproducible and applied identically in training and inference.
- Responsibility (separation of concerns): Define the preprocessing recipe only, without fitting on data.
- Pipeline contract (inputs and outputs): Returns a ColumnTransformer that will be used inside a Pipeline.
"""

#test comment

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

#adding the preprocessor function
def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols: numeric columns to bin using quantiles
    - categorical_onehot_cols: categorical columns for one hot encoding
    - numeric_passthrough_cols: numeric columns to impute and scale
    - n_bins: number of quantile bins for KBinsDiscretizer
    Outputs:
    - preprocessor: A scikit-learn ColumnTransformer (not fitted)
    Why this contract matters for reliable ML delivery:
    - Keeping preprocessing inside the model Pipeline prevents leakage and ensures training and inference use identical transforms.
    """
    print("[features.get_feature_preprocessor] Building feature preprocessing recipe")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    transformers = []

    if len(quantile_bin_cols) > 0:
        quantile_bin_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("bin", KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="quantile")),
            ]
        )
        transformers.append(("quantile_bin", quantile_bin_pipe, quantile_bin_cols))

    if len(categorical_onehot_cols) > 0:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe),
            ]
        )
        transformers.append(("categorical_onehot", categorical_pipe, categorical_onehot_cols))

    if len(numeric_passthrough_cols) > 0:
        numeric_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric_scaled", numeric_pipe, numeric_passthrough_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # --------------------------------------------------------
    # PROJECT SPECIFIC MODEL LOGIC (FEATURES)
    # --------------------------------------------------------
    # Purpose:
    # - This module defines the "default" preprocessing recipe (above).
    # - Every project/dataset usually needs 1 to 2 extra transformations that are business-specific.
    # - This section is the only place teammates should modify to match our notebook decisions.
    #
    # What belongs here:
    # - Extra feature transformations that must be identical in train and inference
    # - Dataset-specific cleaning decisions we want baked into the pipeline
    #
    # What does NOT belong here:
    # - .fit() calls (never fit inside this function)
    # - Target (y) transformations
    # - Model training or evaluation logic
    #
    # If we haven't implemented this yet:
    # - Leave it as a warning for now, but remove the warning once implemented.
    print("Warning: Project specific logic not implemented yet")
    # --------------------------------------------------------
    # END PROJECT SPECIFIC MODEL LOGIC
    # --------------------------------------------------------

    return preprocessor