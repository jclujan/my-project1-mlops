from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

from src.load_data import load_raw_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from src.utils import save_csv, save_model


SETTINGS = {
    "paths": {
        "raw": Path("data/raw/data.csv"),
        "processed": Path("data/processed/clean.csv"),
        "model": Path("models/model.joblib"),
        "predictions": Path("reports/predictions.csv"),
    },
    "target_column": "target",
    "problem_type": "regression",  # "regression" | "classification"
    "split": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "features": {
        # Estas listas DEBEN corresponder a columnas reales tras clean_dataframe
        "quantile_bin": ["num_feature"],
        "categorical_onehot": ["cat_feature"],
        "numeric_passthrough": [],
        "n_bins": 3,
    },
}


def main():
    SETTINGS["paths"]["processed"].parent.mkdir(parents=True, exist_ok=True)
    SETTINGS["paths"]["model"].parent.mkdir(parents=True, exist_ok=True)
    SETTINGS["paths"]["predictions"].parent.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(SETTINGS["paths"]["raw"])
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])
    save_csv(df_clean, SETTINGS["paths"]["processed"])

    required_cols = [SETTINGS["target_column"]] + (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
    )
    validate_dataframe(df_clean, required_columns=required_cols)

    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]

    stratify = None
    if SETTINGS["problem_type"] == "classification":
        stratify = y
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["split"]["test_size"],
            random_state=SETTINGS["split"]["random_state"],
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["split"]["test_size"],
            random_state=SETTINGS["split"]["random_state"],
            stratify=None,
        )

    # checks mínimos para evitar configs rotas
    configured = (
        SETTINGS["features"]["quantile_bin"]
        + SETTINGS["features"]["categorical_onehot"]
        + SETTINGS["features"]["numeric_passthrough"]
    )
    missing = [c for c in configured if c not in X_train.columns]
    if missing:
        raise ValueError(f"Configured feature columns missing from data: {missing}")

    for col in SETTINGS["features"]["quantile_bin"]:
        if not is_numeric_dtype(X_train[col]):
            raise ValueError(f"Quantile bin column is not numeric: {col}")

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        n_bins=SETTINGS["features"]["n_bins"],
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=SETTINGS["problem_type"],
    )
    save_model(model, SETTINGS["paths"]["model"])

    metric = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        problem_type=SETTINGS["problem_type"],
    )
    print(f"Metric: {metric}")

    preds_df = run_inference(model, X_test.head(10))
    save_csv(preds_df, SETTINGS["paths"]["predictions"])


if __name__ == "__main__":
    main()