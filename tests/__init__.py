"""
Test suite for the MLOps house price prediction pipeline.

Each test file maps to one src/ module:

    test_load_data.py   → src/load_data.py   (CSV loading, error handling)
    test_clean_data.py  → src/clean_data.py  (deduplication, column standardisation,
                                              target extraction)
    test_validate.py    → src/validate.py    (domain constraint checks, fail-fast)
    test_features.py    → src/features.py    (ColumnTransformer construction,
                                              fit/transform smoke test)
    test_train.py       → src/train.py       (regression + classification pipelines,
                                              GridSearchCV, invalid problem_type)
    test_evaluate.py    → src/evaluate.py    (RMSE/MAE/R²/RMSLE, edge cases)
    test_infer.py       → src/infer.py       (artifact loading, inference output shape,
                                              empty DataFrame guard)

Run the full suite from the project root:

    pytest

Run a single file:

    pytest tests/test_train.py

Run only fast tests (exclude slow/integration):

    pytest -m "not slow and not integration"

Check coverage for a specific module:

    pytest tests/test_train.py --cov=src/train --cov-report=term-missing
"""
