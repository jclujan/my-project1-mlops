# tests/test_main_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

import src.main as m
import src.utils as u


@dataclass
class _Dataset:
    df: pd.DataFrame


class _FakeModel:
    def predict(self, X: pd.DataFrame):
        return [0.0] * len(X)


def test__load_config_raises_if_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        u.load_config(tmp_path / "missing.yaml")


def test__fail_fast_feature_checks_missing_cols_raises():
    X = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        u.fail_fast_feature_checks(
            X,
            quantile_bin=["c"],
            categorical_onehot=[],
            numeric_passthrough=[],
        )


def test__fail_fast_feature_checks_non_numeric_quantile_bin_raises():
    X = pd.DataFrame({"q": ["x"], "n": [1]})
    with pytest.raises(ValueError):
        u.fail_fast_feature_checks(
            X,
            quantile_bin=["q"],
            categorical_onehot=[],
            numeric_passthrough=["n"],
        )


def test_main_happy_path_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train_path = tmp_path / "data" / "raw" / "train.csv"
    test_path = tmp_path / "data" / "raw" / "test.csv"
    clean_out_path = tmp_path / "data" / "processed" / "clean.csv"
    model_out_path = tmp_path / "models" / "model.joblib"
    preds_out_path = tmp_path / "reports" / "predictions.csv"

    cfg = {
        "data": {
            "raw": {"train_path": str(train_path), "test_path": str(test_path)},
            "processed": {"clean_path": str(clean_out_path)},
            "inference": {
                "input_path": str(tmp_path / "data" / "inference" / "input.csv"),
                "output_path": str(tmp_path / "data" / "inference" / "output.csv"),
            },
        },
        "output": {
            "model_path": str(model_out_path),
            "predictions_path": str(preds_out_path),
        },
        "pipeline": {
            "problem_type": "regression",
            "target_column": "SalePrice",
            "id_column": "Id",
        },
        "cleaning": {"drop_cols": []},
        "features": {
            "quantile_bin": ["LotArea"],
            "categorical_onehot": ["Neighborhood"],
            "numeric_passthrough": ["OverallQual", "YearBuilt", "GrLivArea"],
            "n_bins": 3,
        },
        "train": {"test_size": 0.25, "random_state": 0},
    }

    def fake_load_config(*_args, **_kwargs):
        return cfg

    def fake_load_dataset(train_p: Path, test_p: Path | None):
        df_train = pd.read_csv(train_p)
        out = {"train": _Dataset(df=df_train)}
        if test_p is not None:
            df_test = df_train.drop(columns=["SalePrice"]).copy()
            df_test.to_csv(test_p, index=False)
            out["test"] = _Dataset(df=pd.read_csv(test_p))
        return out

    @dataclass
    class _CleanResult:
        X: pd.DataFrame
        y: pd.Series | None

    def fake_clean_housing_data(df: pd.DataFrame, *, target_col: str, drop_cols, require_target: bool):
        df2 = df.drop(columns=list(drop_cols), errors="ignore").copy()
        if require_target:
            y = df2[target_col].copy()
            X = df2.drop(columns=[target_col])
            return _CleanResult(X=X, y=y)
        X = df2.drop(columns=[target_col], errors="ignore")
        return _CleanResult(X=X, y=None)

    def fake_validate_dataframe(X: pd.DataFrame, required_cols):
        missing = [c for c in required_cols if c not in X.columns]
        if missing:
            raise ValueError

    def fake_get_feature_preprocessor(*_args, **_kwargs):
        return object()

    def fake_train_model(X_train, y_train, preprocessor, problem_type):
        return _FakeModel()

    def fake_evaluate_regression(y_true, y_pred, compute_rmsle: bool):
        return {"rmse": 0.0, "r2": 1.0}

    def fake_run_inference(*, input_df: pd.DataFrame, artifact: dict, id_col: str, pred_col: str):
        return pd.DataFrame(
            {id_col: input_df[id_col].values, pred_col: [123.0] * len(input_df)}
        )

    def fake_ensure_parent_dir(path_str: str):
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    def fake_joblib_dump(obj, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"artifact")

    monkeypatch.setattr(m, "load_config", fake_load_config)
    monkeypatch.setattr(m, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(m, "clean_housing_data", fake_clean_housing_data)
    monkeypatch.setattr(m, "validate_dataframe", fake_validate_dataframe)
    monkeypatch.setattr(m, "get_feature_preprocessor", fake_get_feature_preprocessor)
    monkeypatch.setattr(m, "train_model", fake_train_model)
    monkeypatch.setattr(m, "evaluate_regression", fake_evaluate_regression)
    monkeypatch.setattr(m, "run_inference", fake_run_inference)
    monkeypatch.setattr(m, "ensure_parent_dir", fake_ensure_parent_dir)
    monkeypatch.setattr(m.joblib, "dump", fake_joblib_dump)

    m.main()

    assert clean_out_path.exists()
    assert model_out_path.exists()
    assert preds_out_path.exists()

    df_clean = pd.read_csv(clean_out_path)
    assert "SalePrice" in df_clean.columns

    df_preds = pd.read_csv(preds_out_path)
    assert list(df_preds.columns) == ["Id", "SalePrice"]
    assert len(df_preds) > 0