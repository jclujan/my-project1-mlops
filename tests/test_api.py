"""
Tests for src/api.py — covers /health and /predict endpoints.
Uses FastAPI TestClient with a mocked model pipeline so no real
model file or W&B connection is needed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_RECORD = {
    "Id": 1,
    "LotArea": 8450,
    "GrLivArea": 1710,
    "Neighborhood": "CollgCr",
    "OverallQual": 7,
    "YearBuilt": 2003,
}

VALID_PAYLOAD = {"records": [VALID_RECORD]}


def _make_mock_pipeline():
    """Returns a sklearn-like pipeline mock that predicts log-space values."""
    mock = MagicMock()
    mock.predict.return_value = np.array([np.log1p(200_000.0)])
    return mock


# ---------------------------------------------------------------------------
# Fixtures: control model loading via patch (NOT app.state)
# ---------------------------------------------------------------------------

@pytest.fixture
def client_with_model():
    mock_pipeline = _make_mock_pipeline()
    with patch("src.api._load_from_local", return_value=(mock_pipeline, "test-v1")):
        with TestClient(app) as client:
            yield client


@pytest.fixture
def client_without_model():
    with patch("src.api._load_from_local", return_value=(None, "missing")):
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client_with_model):
    r = client_with_model.get("/health")
    assert r.status_code == 200


def test_health_response_schema(client_with_model):
    r = client_with_model.get("/health")
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "model_version" in body


def test_health_returns_200_when_model_missing(client_without_model):
    r = client_without_model.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is False


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------

def test_predict_valid_payload_returns_200(client_with_model):
    r = client_with_model.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200


def test_predict_response_contains_sale_price(client_with_model):
    r = client_with_model.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert "SalePrice" in body["predictions"][0]
    assert body["predictions"][0]["SalePrice"] > 0


def test_predict_echoes_id(client_with_model):
    r = client_with_model.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert body["predictions"][0]["Id"] == 1


def test_predict_multiple_records(client_with_model):
    payload = {
        "records": [
            VALID_RECORD,
            {**VALID_RECORD, "Id": 2, "LotArea": 9000},
        ]
    }

    # override mock for multiple outputs
    mock = MagicMock()
    mock.predict.return_value = np.array(
        [np.log1p(200_000.0), np.log1p(220_000.0)]
    )

    with patch("src.api._load_from_local", return_value=(mock, "test-v1")):
        with TestClient(app) as client:
            r = client.post("/predict", json=payload)

    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_returns_model_version(client_with_model):
    r = client_with_model.post("/predict", json=VALID_PAYLOAD)
    assert r.json()["model_version"] == "test-v1"


# ---------------------------------------------------------------------------
# /predict — validation errors (Pydantic → 422)
# ---------------------------------------------------------------------------

def test_predict_missing_required_field_returns_422(client_with_model):
    bad = {"records": [{"Id": 1, "LotArea": 8450}]}
    r = client_with_model.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_extra_field_returns_422(client_with_model):
    bad_record = {**VALID_RECORD, "UnknownField": 99}
    r = client_with_model.post("/predict", json={"records": [bad_record]})
    assert r.status_code == 422


def test_predict_wrong_type_returns_422(client_with_model):
    bad = {
        "records": [
            {
                "Id": 1,
                "LotArea": "not-a-number",
                "GrLivArea": 1710,
                "Neighborhood": "CollgCr",
                "OverallQual": 7,
                "YearBuilt": 2003,
            }
        ]
    }
    r = client_with_model.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_empty_records_returns_error(client_with_model):
    r = client_with_model.post("/predict", json={"records": []})
    assert r.status_code in (422, 500)


# ---------------------------------------------------------------------------
# /predict — model not loaded → 503
# ---------------------------------------------------------------------------

def test_predict_returns_503_when_model_missing(client_without_model):
    r = client_without_model.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503