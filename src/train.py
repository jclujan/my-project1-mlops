"""
Module: Model Training
----------------------

Role: Train model and return fitted artifact.
Input: X_train (DataFrame), y_train (Series), preprocessor (ColumnTransformer), problem_type (str).
Output: Fitted model object (GridSearchCV wrapping a Pipeline, or a Pipeline).

Educational Goal:
- Centralize training for reuse and reproducibility.
- Fit ONLY on training split.
- Keep preprocessing inside the Pipeline to avoid leakage.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression

logger = logging.getLogger(__name__)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    problem_type: str,
):
    """
    Inputs:
    - X_train: Training features
    - y_train: Training target
    - preprocessor: ColumnTransformer recipe (NOT fitted)
    - problem_type: "regression" or "classification"

    Outputs:
    - model: Fitted scikit-learn object (GridSearchCV wrapping a Pipeline, or a Pipeline)

    Notes for this project (README alignment):
    - For regression, the target is modeled in log-space (log1p).
      This means the model learns log(price), not price directly.
      Downstream evaluation/inference must invert the transform with expm1.
    """
    logger.info("Training model pipeline")

    X_fit = X_train.copy()
    y_fit = y_train.copy()

    # --------------------------------------------------------
    # PROJECT SPECIFIC MODEL LOGIC (TRAIN) - TEAM EDIT HERE
    # --------------------------------------------------------
    # Why this exists:
    # - This block is where we align training with README / notebook decisions.
    # - Keep the rest of the function stable so main.py / evaluate.py / infer.py
    #   can rely on a consistent interface.
    #
    # What belongs here:
    # 1) Model choice (Lasso, RF, XGBoost, etc.)
    # 2) CV strategy (KFold/StratifiedKFold, folds, random_state)
    # 3) Hyperparameter tuning (GridSearchCV, RandomizedSearchCV, none)
    # 4) Scoring metric
    #
    # What must NOT happen here:
    # - No preprocessing outside the pipeline
    # - No fitting the preprocessor here
    #
    # README alignment reminders:
    # - Regression target is trained in log-space (log1p).
    # - Feature preprocessing steps (missing handling, rare grouping, log features,
    #   multicollinearity drop, one-hot) belong in features.py / preprocessing recipe,
    #   not here.
    # --------------------------------------------------------

    if problem_type == "regression":
        # 1) Target in log-space (README)
        # log1p is safer than log because it can handle y=0.
        y_fit = np.log1p(y_fit.astype(float))

        # 2) Estimator (README mentions Lasso in your notebook name)
        estimator = Lasso(max_iter=20000, random_state=42)

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        # 3) Hyperparameter tuning
        param_grid = {
            "model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        }

        # 4) Cross validation setup
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # 5) Scoring metric
        # Important: this RMSE is on log-space target.
        model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        model.fit(X_fit, y_fit)
        return model

    if problem_type == "classification":
        estimator = LogisticRegression(max_iter=500)

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        param_grid = {"model__C": [0.5, 1.0, 2.0]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        model = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        model.fit(X_fit, y_fit)
        return model

    raise ValueError("problem_type must be 'regression' or 'classification'")