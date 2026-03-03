import os
import tempfile
import pandas as pd
import numpy as np
import joblib
import pytest

from src.infer import load_artifact, run_inference


# ------------------------
# Helper: dummy pipeline
# ------------------------

class DummyModel:
    def predict(self, X):
        # Return constant prediction
        return np.ones(len(X))


def create_dummy_artifact(tmp_path):
    pipeline = DummyModel()
    artifact = {
        "pipeline": pipeline,
        "metadata": {"target_transform": None},
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact, model_path)
    return model_path


# ------------------------
# Tests
# ------------------------

def test_load_artifact_success(tmp_path):
    model_path = create_dummy_artifact(tmp_path)
    artifact = load_artifact(model_path)
    assert "pipeline" in artifact
    assert "metadata" in artifact


def test_load_artifact_invalid_structure(tmp_path):
    bad_path = tmp_path / "bad.joblib"
    joblib.dump("not_a_dict", bad_path)

    with pytest.raises(TypeError):
        load_artifact(bad_path)


def test_run_inference_preserves_index(tmp_path):
    model_path = create_dummy_artifact(tmp_path)
    artifact = load_artifact(model_path)

    df = pd.DataFrame({"A": [1, 2, 3]}, index=[10, 20, 30])
    out = run_inference(df, artifact, id_col="Id", pred_col="Pred")

    assert list(out.index) == [10, 20, 30]


def test_run_inference_output_shape(tmp_path):
    model_path = create_dummy_artifact(tmp_path)
    artifact = load_artifact(model_path)

    df = pd.DataFrame({"A": [1, 2, 3]})
    out = run_inference(df, artifact)

    assert len(out) == 3
    assert "SalePrice" in out.columns


def test_run_inference_empty_dataframe(tmp_path):
    model_path = create_dummy_artifact(tmp_path)
    artifact = load_artifact(model_path)

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        run_inference(df, artifact)