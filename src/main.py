"""
Educational Goal:
- Why this module exists in an MLOps system: Provide a single, repeatable
  entrypoint that stitches together data -> cleaning -> validation ->
  features -> training -> evaluation -> inference -> artifact saving.
- Responsibility (separation of concerns): Orchestrates the workflow only;
  all real logic lives in the other src/ modules.
- Pipeline contract: Reads raw CSV(s), writes clean CSV + model artifact
  + predictions CSV, logs metrics and artifacts to Weights & Biases.

What this module owns:
- Load and validate config.yaml
- Initialise W&B run and finish it cleanly
- Orchestrate: load -> clean -> validate -> split -> features -> train
  -> evaluate -> save -> log artifact -> infer

What this module does not own:
- Data cleaning rules (src/clean_data.py)
- Feature engineering (src/features.py)
- Training algorithm (src/train.py)
- Evaluation metrics (src/evaluate.py)
- Inference logic (src/infer.py)
- Validation rules (src/validate.py)
- Logging setup (src/logger.py)

Usage:
    python -m src.main
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.clean_data import clean_housing_data
from src.evaluate import evaluate_regression
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_dataset, load_csv
from src.logger import configure_logging
from src.train import train_model
from src.utils import (
    ensure_parent_dir,
    fail_fast_feature_checks,
    load_config,
    make_dummy_ames_like_csv,
)
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

# CONFIGURATION — loaded from config.yaml, never hardcoded here
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def main() -> None:
    """
    Inputs:
    - None (reads all settings from config.yaml, secrets from .env)

    Outputs:
    - Writes: data/processed/clean.csv, models/model.joblib,
      reports/predictions.csv, logs/pipeline.log
    - Logs: metrics, artifact, and run metadata to Weights & Biases

    Why this contract matters for reliable ML delivery:
    - A single entrypoint is easy to run in CI, easy to reproduce across
      machines, and easy to schedule or deploy.
    - W&B tracking ensures every run is auditable and every model artifact
      is versioned and promotable to production.
    """
    load_dotenv()

    # ------------------------------------------------------------------
    # 0) Config + logging
    # ------------------------------------------------------------------
    cfg = load_config(_CONFIG_PATH)

    configure_logging(
        log_level=cfg["logging"]["level"],
        log_file=Path(cfg["logging"]["log_file"]),
    )

    logger.info("[main] Starting pipeline")

    train_path     = Path(cfg["data"]["raw"]["train_path"])
    test_path      = Path(cfg["data"]["raw"]["test_path"])
    clean_out_path = Path(cfg["data"]["processed"]["clean_path"])
    model_out_path = Path(cfg["output"]["model_path"])
    preds_out_path = Path(cfg["output"]["predictions_path"])

    problem_type  = cfg["pipeline"]["problem_type"]
    target_column = cfg["pipeline"]["target_column"]
    id_column     = cfg["pipeline"]["id_column"]

    ensure_parent_dir(clean_out_path)
    ensure_parent_dir(model_out_path)
    ensure_parent_dir(preds_out_path)

    # ------------------------------------------------------------------
    # W&B init
    # Names come from config.yaml — never hardcoded in this file.
    # Set WANDB_MODE=disabled in .env to skip tracking (used in CI).
    # ------------------------------------------------------------------
    wandb_project       = cfg["wandb"]["project"]
    wandb_artifact_name = cfg["wandb"]["model_artifact_name"]

    run = wandb.init(
        project=wandb_project,
        job_type="training",
        config=cfg,
        mode=os.getenv("WANDB_MODE", "online"),
    )
    logger.info("[main] W&B run initialised | name=%s", run.name)

    try:
        # ------------------------------------------------------------------
        # 1) Load (or create dummy data for CI smoke-tests)
        # ------------------------------------------------------------------
        logger.info("[main] Step 1 - Load raw data")
        if not train_path.exists():
            make_dummy_ames_like_csv(train_path)

        data         = load_dataset(train_path, test_path if test_path.exists() else None)
        df_train_raw = data["train"].df

        # ------------------------------------------------------------------
        # 2) Clean
        # ------------------------------------------------------------------
        logger.info("[main] Step 2 - Clean training data")
        clean_train = clean_housing_data(
            df_train_raw,
            target_col=target_column,
            drop_cols=cfg["cleaning"]["drop_cols"],
            require_target=True,
        )
        X_all = clean_train.X
        y_all = clean_train.y

        if y_all is None:
            raise ValueError(
                "Cleaning returned y=None for training data. "
                "Check target_column in config.yaml."
            )

        # Save processed clean CSV (includes target for traceability)
        logger.info("[main] Saving processed clean CSV")
        df_clean_materialized = X_all.copy()
        df_clean_materialized[target_column] = y_all.values
        df_clean_materialized.to_csv(clean_out_path, index=False)

        # ------------------------------------------------------------------
        # 3) Validate (fail fast)
        # ------------------------------------------------------------------
        logger.info("[main] Step 3 - Validate features dataframe")
        required_cols = list(dict.fromkeys(
            cfg["features"]["quantile_bin"]
            + cfg["features"]["categorical_onehot"]
            + cfg["features"]["numeric_passthrough"]
        ))
        validate_dataframe(X_all, required_cols)

        # ------------------------------------------------------------------
        # 4) Train/test split
        # ------------------------------------------------------------------
        logger.info("[main] Step 4 - Train/test split")
        stratify = y_all if problem_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all,
                test_size=cfg["train"]["test_size"],
                random_state=cfg["train"]["random_state"],
                stratify=stratify,
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all,
                test_size=cfg["train"]["test_size"],
                random_state=cfg["train"]["random_state"],
            )

        # ------------------------------------------------------------------
        # 5) Fail-fast feature config checks (post-split)
        # ------------------------------------------------------------------
        fail_fast_feature_checks(
            X_train,
            quantile_bin=cfg["features"]["quantile_bin"],
            categorical_onehot=cfg["features"]["categorical_onehot"],
            numeric_passthrough=cfg["features"]["numeric_passthrough"],
        )

        # ------------------------------------------------------------------
        # 6) Build feature preprocessor
        # ------------------------------------------------------------------
        logger.info("[main] Step 6 - Build feature preprocessor")
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=cfg["features"]["quantile_bin"],
            categorical_onehot_cols=cfg["features"]["categorical_onehot"],
            numeric_passthrough_cols=cfg["features"]["numeric_passthrough"],
            n_bins=int(cfg["features"]["n_bins"]),
        )

        # ------------------------------------------------------------------
        # 7) Train model
        # ------------------------------------------------------------------
        logger.info("[main] Step 7 - Train model")
        model = train_model(X_train, y_train, preprocessor, problem_type)

        # ------------------------------------------------------------------
        # 8) Evaluate on held-out test set
        # ------------------------------------------------------------------
        logger.info("[main] Step 8 - Evaluate on held-out test")
        if problem_type == "regression":
            # train.py trains y in log-space -> predictions are log-space
            y_pred_log   = model.predict(X_test)
            y_pred_price = np.expm1(y_pred_log)
            y_true_price = y_test.astype(float).values

            metrics = evaluate_regression(
                y_true_price, y_pred_price, compute_rmsle=True
            )
            logger.info("[main] Regression metrics (price-scale): %s", metrics)

            # Log every metric to W&B
            wandb.log({f"metrics/{k}": v for k, v in metrics.items()})

        else:
            logger.info("[main] Classification: evaluation not wired yet.")

        # ------------------------------------------------------------------
        # 9) Save model artifact + log to W&B
        # ------------------------------------------------------------------
        logger.info("[main] Step 9 - Save model artifact")
        artifact_payload = {
            "pipeline": model,
            "metadata": {
                "problem_type":     problem_type,
                "target_transform": "log1p" if problem_type == "regression" else "none",
                "wandb_run_id":     run.id,
                "version":          "1.0.0",
            },
        }
        joblib.dump(artifact_payload, model_out_path)
        logger.info("[main] Saved model artifact → %s", model_out_path)

        # Log artifact to W&B so it can be versioned and promoted to prod
        wandb_artifact = wandb.Artifact(
            name=wandb_artifact_name,
            type="model",
            description="Lasso regression pipeline for house price prediction (Ames, Iowa)",
            metadata=artifact_payload["metadata"],
        )
        wandb_artifact.add_file(str(model_out_path))
        run.log_artifact(wandb_artifact)
        logger.info("[main] Artifact logged to W&B | name=%s", wandb_artifact_name)
        logger.info(
            "[main] ACTION REQUIRED: W&B → %s → Artifacts → %s "
            "→ latest version → Add Alias → type 'prod'",
            wandb_project, wandb_artifact_name,
        )

        # ------------------------------------------------------------------
        # 10) Inference on unseen data + save predictions
        # ------------------------------------------------------------------
        logger.info("[main] Step 10 - Inference + save predictions")
        infer_input_path = Path(cfg["data"]["inference"]["input_path"])

        if infer_input_path.exists():
            logger.info("[main] Loading inference data from: %s", infer_input_path)
            infer_result = load_csv(infer_input_path)
            clean_infer  = clean_housing_data(
                infer_result.df,
                target_col=target_column,
                drop_cols=[],   # keep Id so run_inference can include it in output
                require_target=False,
            )
            X_infer = clean_infer.X
            logger.info("[main] Inference rows: %d", len(X_infer))
        else:
            logger.info(
                "[main] No inference file at %s; falling back to X_test split",
                infer_input_path,
            )
            X_infer = X_test.copy()

        preds_df = run_inference(
            input_df=X_infer,
            artifact=artifact_payload,
            id_col=id_column,
            pred_col=target_column,
        )

        ensure_parent_dir(str(preds_out_path))
        preds_df.to_csv(preds_out_path, index=False)
        logger.info("[main] Saved predictions → %s", preds_out_path)

        logger.info("[main] Pipeline finished successfully")

    except Exception:
        logger.exception("[main] Pipeline failed")
        wandb.finish(exit_code=1)
        raise

    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()