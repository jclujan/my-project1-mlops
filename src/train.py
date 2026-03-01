"""
Module: Model Training
----------------------

Role: Split data, train model, and save the artifact.
Input: pandas.DataFrame (Processed).
Output: Serialized model file (e.g., .pkl) in `models/`.

----------------------
Educational Goal:
- Why this module exists in an MLOps system: Centralize model training so it is reusable, testable, and consistent across runs.
- Responsibility (separation of concerns): Train (fit) a scikit-learn Pipeline on the training split only and return the fitted model object.
- Pipeline contract (inputs and outputs): Inputs are X_train, y_train, a preprocessing recipe, and the problem type; output is a fitted Pipeline ready for evaluation and inference.
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
    """
    Inputs:
    - X_train: Training features
    - y_train: Training target
    - preprocessor: ColumnTransformer recipe
    - problem_type: "regression" or "classification"
    Outputs:
    - model: Fitted scikit-learn object (Pipeline or GridSearchCV wrapping a Pipeline)
    Why this contract matters for reliable ML delivery:
    - Training must be isolated and reproducible. Putting preprocessing inside the CV pipeline prevents leakage and aligns with standard protocol.
    """
    print("[train.train_model] Training model pipeline")  # TODO: replace with logging later

    X_fit = X_train.copy()
    y_fit = y_train.copy()

    if problem_type == "regression":
        # Generic outlier removal to address "very expensive mansions are rare"
        # This is intentionally simple and configurable later.
        upper_q = float(np.nanquantile(y_fit, 0.99))
        keep_mask = y_fit <= upper_q
        X_fit = X_fit.loc[keep_mask]
        y_fit = y_fit.loc[keep_mask]

    if problem_type == "classification":
        estimator = LogisticRegression(max_iter=500)
    else:
        # Baseline estimator will be swapped by CV below
        estimator = Ridge()

    base_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    # --------------------------------------------------------
    # PROJECT SPECIFIC MODEL LOGIC (TRAIN)
    # --------------------------------------------------------
    # Purpose:
    # - This is where we decide "how we train" for THIS project:
    #   model choice + tuning strategy + validation + metric.
    # - The rest of train.py should stay stable so main.py / evaluate.py / infer.py
    #   can rely on a consistent training interface.
    #
    # What belongs here :
    # 1) Estimator (the model)
    #    Examples: LogisticRegression, RandomForest, XGBoost, Lasso/ElasticNet, etc.
    #
    # 2) Hyperparameter tuning approach
    #    - No search: directly fit the pipeline
    #    - GridSearchCV: exhaustive, slower but systematic
    #    - RandomizedSearchCV: faster exploration, good default for many params
    #
    # 3) Cross-validation setup
    #    - Number of folds (cv=5, cv=10)
    #    - Shuffle + random_state (when appropriate)
    #    - StratifiedKFold for classification, KFold for regression
    #
    # 4) Scoring metric (must match the problem)
    #    - Regression: neg_root_mean_squared_error, neg_mean_absolute_error, r2
    #    - Classification: f1, roc_auc, accuracy, precision, recall
    #
    # What does NOT belong here (to avoid leakage and broken MLOps behavior):
    # - Do NOT preprocess outside the Pipeline (no scaler.fit(X) in advance).
    # - Do NOT fit the preprocessor on the full dataset before CV.
    # - Do NOT manually transform X before GridSearchCV.
    #
    # Why the "FULL pipeline CV" rule matters:
    # - During CV, each fold must learn preprocessing ONLY from its training split.
    # - This prevents leakage and matches what we will do in production inference.
    #
    # Placeholder reminder (remove once team selects final logic):
    print("Warning: Using baseline training setup (GridSearchCV + Pipeline).")  # TODO: replace with logging later
    # --------------------------------------------------------
    # END PROJECT SPECIFIC MODEL LOGIC (TRAIN)
    # --------------------------------------------------------

    # Fix leakage in cross validation by tuning the whole Pipeline, not pre scaled data.
    if problem_type == "regression":
        # Use Lasso + GridSearchCV so scaling and encoding are refit per fold.
        base_pipeline.set_params(model=Lasso(max_iter=20000))

        # Keep it small and stable; we can change this later.
        param_grid = {"model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        model = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        model.fit(X_fit, y_fit)
        return model

    if problem_type == "classification":
        # Optional: tune C similarly, also without leakage.
        param_grid = {"model__C": [0.5, 1.0, 2.0]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        model = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        model.fit(X_fit, y_fit)
        return model

    # Fallback, should not be reached, but keeps the function robust.
    base_pipeline.fit(X_fit, y_fit)
    return base_pipeline