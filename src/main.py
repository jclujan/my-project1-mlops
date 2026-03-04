""" Educational Goal:
• Why this module exists in an MLOps system: Provide a single, repeatable entrypoint that stitches together data → cleaning → validation → features → training → evaluation → inference → artifact saving.
• Responsibility (separation of concerns): Orchestrates the workflow only; all real logic lives in src/load_data.py, src/clean_data.py, src/features.py, src/train.py, src/evaluate.py, src/infer.py.
• Pipeline contract (inputs and outputs): Reads raw CSV(s), writes clean CSV + model artifact + predictions CSV, prints metrics to console.
TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session """

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.load_data import load_dataset
from src.clean_data import clean_housing_data
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_regression
from src.infer import run_inference
from src.utils import ensure_parent_dir


# ------------------------------
# CONFIGURATION (bridge to YAML)
# ------------------------------
SETTINGS = {
    "problem_type": "regression",  # "regression" or "classification"
    "target_column": "SalePrice",
    "id_column": "Id",
    "paths": {
        "train_csv": "data/raw/train.csv",
        "test_csv": "data/raw/test.csv",  # optional
        "processed_clean_csv": "data/processed/clean.csv",
        "model_artifact": "models/model.joblib",
        "predictions_csv": "reports/predictions.csv",
    },
    "split": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "features": {
        # Preconfigured to work with the dummy dataset created below (if train.csv is missing).
        # LOUD REMINDER: You MUST map these lists to your real dataset columns and paths.
        "quantile_bin": ["LotArea", "GrLivArea"],
        "categorical_onehot": ["Neighborhood"],
        "numeric_passthrough": ["OverallQual", "YearBuilt"],
        "n_bins": 3,
    },
    "cleaning": {
        "drop_cols": ["Id"],        # cleaned out of X, but inference can still keep Id if present
        "require_target": True,     # True for training data; False for test/inference data
    },
}


def _make_dummy_ames_like_csv(path: Path) -> None: # will be used if train.csv is missing, to keep the repo runnable 
    """ Inputs:
    • path: Where to write a dummy CSV if the real one is missing.
    Outputs:
    • None (writes a tiny deterministic CSV).
    Why this contract matters for reliable ML delivery:
    • Keeps the repo runnable end-to-end so CI and onboarding work even before real data is wired. """
    print(f"[main._make_dummy_ames_like_csv] Creating dummy dataset at: {path}")  # TODO: replace with logging later
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6, 7, 8],
            "Neighborhood": ["NAmes", "CollgCr", "OldTown", "NAmes", "Somerst", "Edwards", "NAmes", "Sawyer"],
            "OverallQual": [5, 7, 4, 6, 8, 5, 6, 7],
            "YearBuilt": [1960, 2003, 1920, 1975, 2007, 1950, 1985, 1995],
            "LotArea": [8450, 9600, 11250, 9550, 14260, 14115, 10084, 10382],
            "GrLivArea": [1710, 1262, 1786, 1717, 2198, 1362, 1694, 2090],
            "SalePrice": [208500, 181500, 223500, 140000, 250000, 143000, 307000, 200000],
        }
    )
    df.to_csv(path, index=False)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("LOUD WARNING: Dummy dataset created for scaffolding ONLY.")
    print("Replace data/raw/train.csv (and test.csv) with your real dataset and")
    print("update SETTINGS['features'] + SETTINGS['target_column'] accordingly.")
    print("Dummy columns:", list(df.columns))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def _fail_fast_feature_checks(X: pd.DataFrame, *, quantile_bin: List[str], categorical_onehot: List[str], numeric_passthrough: List[str]) -> None:
    # will be used after train/test split to check that the feature config matches the actual dataframe schema
    """ Inputs:
    • X: Feature dataframe (already cleaned, target removed).
    • quantile_bin/categorical_onehot/numeric_passthrough: configured feature lists.
    Outputs:
    • None (raises ValueError on misconfiguration).
    Why this contract matters for reliable ML delivery:
    • Most pipeline failures are config/schema mismatches; failing fast saves time and avoids silent bugs. """
    print("[main._fail_fast_feature_checks] Checking feature configuration vs dataframe schema")  # TODO: replace with logging later

    configured = list(dict.fromkeys(quantile_bin + categorical_onehot + numeric_passthrough))
    missing = [c for c in configured if c not in X.columns]
    if missing:
        raise ValueError(
            f"Feature config error: these configured columns are missing from X: {missing}. "
            f"Available columns: {list(X.columns)}"
        )

    non_numeric_bins = [c for c in quantile_bin if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric_bins:
        raise ValueError(
            f"Feature config error: quantile_bin columns must be numeric dtypes, but these are not: {non_numeric_bins}"
        )


def main() -> None:
    """ Inputs:
    • None (uses SETTINGS dict as a config bridge for now).
    Outputs:
    • Writes:
      - data/processed/clean.csv
      - models/model.joblib
      - reports/predictions.csv
    Why this contract matters for reliable ML delivery:
    • A single entrypoint is easy to run in CI, easy to reproduce across machines, and easy to schedule/deploy. """
    print("[main.main] Starting pipeline")  # TODO: replace with logging later

    # ------------------------------
    # 0) Paths + directories
    # ------------------------------
    print("[main.main] Step 0 - Ensure output directories exist")  # TODO: replace with logging later
    train_path = Path(SETTINGS["paths"]["train_csv"])
    test_path = Path(SETTINGS["paths"]["test_csv"])
    clean_out_path = Path(SETTINGS["paths"]["processed_clean_csv"])
    model_out_path = Path(SETTINGS["paths"]["model_artifact"])
    preds_out_path = Path(SETTINGS["paths"]["predictions_csv"])

    clean_out_path.parent.mkdir(parents=True, exist_ok=True)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_out_path.parent.mkdir(parents=True, exist_ok=True)


    # ------------------------------
    # 1) Load (or create dummy)
    # ------------------------------
    print("[main.main] Step 1 - Load raw data")  # TODO: replace with logging later
    if not train_path.exists():
        _make_dummy_ames_like_csv(train_path)

    data = load_dataset(train_path, test_path if test_path.exists() else None)
    df_train_raw = data["train"].df

   # No custom ingestion logic required for this dataset. 
   # If your dataset requires special parsing (e.g., multiple CSVs, nested JSON, SQL), 
   # implement that logic in src/load_data.py and call it here.


    # ------------------------------
    # 2) Clean (train)
    # ------------------------------
    print("[main.main] Step 2 - Clean training data")  # TODO: replace with logging later
    clean_train = clean_housing_data(
        df_train_raw,
        target_col=SETTINGS["target_column"],
        drop_cols=SETTINGS["cleaning"]["drop_cols"],
        require_target=True,
    )
    X_all = clean_train.X
    y_all = clean_train.y
    if y_all is None:
        raise ValueError("Cleaning returned y=None for training data. Check target_column / require_target.")

    # Save processed clean.csv (include target for traceability)
    print("[main.main] Saving processed clean CSV")  # TODO: replace with logging later
    df_clean_materialized = X_all.copy()
    df_clean_materialized[SETTINGS["target_column"]] = y_all.values
    df_clean_materialized.to_csv(clean_out_path, index=False)

    # ------------------------------
    # 3) Validate (fail fast)
    # ------------------------------
    print("[main.main] Step 3 - Validate features dataframe")  # TODO: replace with logging later
    required_cols = list(
        dict.fromkeys(
            SETTINGS["features"]["quantile_bin"]
            + SETTINGS["features"]["categorical_onehot"]
            + SETTINGS["features"]["numeric_passthrough"]
        )
    )
    validate_dataframe(X_all, required_cols)

    # ------------------------------
    # 4) Train/test split (before fitting any preprocess)
    # ------------------------------
    print("[main.main] Step 4 - Train/test split")  # TODO: replace with logging later
    stratify = y_all if SETTINGS["problem_type"] == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=SETTINGS["split"]["test_size"],
            random_state=SETTINGS["split"]["random_state"],
            stratify=stratify,
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=SETTINGS["split"]["test_size"],
            random_state=SETTINGS["split"]["random_state"],
            stratify=None,
        )

    # ------------------------------
    # 5) Fail-fast feature config checks (post-split)
    # ------------------------------
    _fail_fast_feature_checks(
        X_train,
        quantile_bin=SETTINGS["features"]["quantile_bin"],
        categorical_onehot=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough=SETTINGS["features"]["numeric_passthrough"],
    )

    # ------------------------------
    # 6) Build preprocessor 
    # ------------------------------
    print("[main.main] Step 6 - Build feature preprocessor")  # TODO: replace with logging later
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        n_bins=int(SETTINGS["features"]["n_bins"]),
    )

    # ------------------------------
    # 7) Train model (Pipeline fits on train split only)
    # ------------------------------
    print("[main.main] Step 7 - Train model")  # TODO: replace with logging later
    model = train_model(X_train, y_train, preprocessor, SETTINGS["problem_type"])

    # Save artifact for infer.py (dict with pipeline + metadata)
    print("[main.main] Saving model artifact")  # TODO: replace with logging later
    artifact = {
        "pipeline": model,
        "metadata": {
            "problem_type": SETTINGS["problem_type"],
            "target_transform": "log1p" if SETTINGS["problem_type"] == "regression" else "none",
        },
    }
    joblib.dump(artifact, model_out_path)

    # ------------------------------
    # 9) Evaluate (console only)
    # ------------------------------
    print("[main.main] Step 8 - Evaluate on held-out test")  # TODO: replace with logging later
    if SETTINGS["problem_type"] == "regression":
        # train.py trains y in log-space -> predictions are log-space
        y_pred_log = model.predict(X_test)
        y_pred_price = pd.Series(y_pred_log, index=X_test.index).pipe(lambda s: s.apply(lambda v: v)).values
        # invert here so evaluate_regression runs on price-scale
        y_pred_price = (pd.Series(y_pred_log).apply(lambda v: v)).values  # keep simple, avoid over-abstraction
        y_pred_price = pd.Series(y_pred_log).pipe(lambda s: s).values  # noop for readability
        y_pred_price = pd.Series(y_pred_log).values  # log predictions
        y_pred_price = pd.Series(y_pred_price).apply(lambda v: v).values  # noop
        y_pred_price = pd.Series(y_pred_price).values  # still log
        y_pred_price = pd.Series(y_pred_price).apply(lambda v: v).values  # still log
        # actual inversion (the only important line)
        import numpy as np  # local import to keep main readable

        y_pred_price = np.expm1(y_pred_log)
        y_true_price = y_test.astype(float).values  # original y is in price-scale

        metrics = evaluate_regression(y_true_price, y_pred_price, compute_rmsle=True)
        print(f"[main.main] Regression metrics (price-scale): {metrics}")  # TODO: replace with logging later
    else:
        print("[main.main] Classification: evaluation not wired to a metrics function yet.")  # TODO: replace with logging later

    # -------------------------------------------------------
    # START STUDENT CODE
    # -------------------------------------------------------
    # TODO_STUDENT: Swap/extend metrics (e.g., MAE only, custom business loss, classification F1).
    # Why: “Best” metric depends on product goals (ranking vs calibration vs cost-sensitive errors).
    # Examples:
    # 1. Add MAPE for price forecasting
    # 2. Add confusion matrix reporting for classification
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    print("Warning: Student has not implemented this section yet")
    # -------------------------------------------------------
    # END STUDENT CODE
    # -------------------------------------------------------

    # ------------------------------
    # 10) Inference + save predictions
    # ------------------------------
    print("[main.main] Step 9 - Inference + save predictions")  # TODO: replace with logging later
    if "test" in data:
        print("[main.main] Using provided test.csv for inference")  # TODO: replace with logging later
        df_test_raw = data["test"].df
        clean_test = clean_housing_data(
            df_test_raw,
            target_col=SETTINGS["target_column"],
            drop_cols=SETTINGS["cleaning"]["drop_cols"],
            require_target=False,
        )
        X_infer = clean_test.X
    else:
        print("[main.main] No test.csv found; inferring on X_test split")  # TODO: replace with logging later
        X_infer = X_test.copy()

    preds_df = run_inference(
        input_df=X_infer,
        artifact=artifact,
        id_col=SETTINGS["id_column"],
        pred_col=SETTINGS["target_column"],
    )

    ensure_parent_dir(str(preds_out_path))
    preds_df.to_csv(preds_out_path, index=False)
    print(f"[main.main] Saved predictions to: {preds_out_path}")  # TODO: replace with logging later


if __name__ == "__main__":
    main()