"""
Tests for src/api.py — covers /health and /predict endpoints.
Uses FastAPI TestClient with a mocked model pipeline so no real
model file or W&B connection is needed.
"""

from unittest.mock import MagicMock

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
    # predict returns log1p(200_000) ≈ 12.2 — infer.py will expm1 it back
    mock.predict.return_value = np.array([np.log1p(200_000.0)])
    return mock


# ---------------------------------------------------------------------------
# Fixture: inject mock model into app.state before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inject_mock_model():
    """Patch app.state so tests never need a real model file."""
    app.state.model_pipeline = _make_mock_pipeline()
    app.state.model_version = "test-v0.0.1"
    yield
    # Cleanup
    app.state.model_pipeline = None
    app.state.model_version = "unloaded"


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_200():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200


def test_health_response_schema():
    with TestClient(app) as client:
        r = client.get("/health")
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "model_version" in body


def test_health_returns_503_when_model_missing():
    with TestClient(app) as client:
        app.state.model_pipeline = None
        r = client.get("/health")
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------

def test_predict_valid_payload_returns_200():
    with TestClient(app) as client:
        r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200


def test_predict_response_contains_sale_price():
    with TestClient(app) as client:
        r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert "SalePrice" in body["predictions"][0]
    assert body["predictions"][0]["SalePrice"] > 0


def test_predict_echoes_id():
    with TestClient(app) as client:
        r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert body["predictions"][0]["Id"] == 1


def test_predict_multiple_records():
    payload = {"records": [VALID_RECORD, {**VALID_RECORD, "Id": 2, "LotArea": 9000}]}
    mock = _make_mock_pipeline()
    mock.predict.return_value = np.array(
        [np.log1p(200_000.0), np.log1p(220_000.0)]
    )
    app.state.model_pipeline = mock

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_returns_model_version():
    with TestClient(app) as client:
        app.state.model_version = "test-v0.0.1"
        r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.json()["model_version"] == "test-v0.0.1"


# ---------------------------------------------------------------------------
# /predict — validation errors (Pydantic rejects bad input → 422)
# ---------------------------------------------------------------------------

def test_predict_missing_required_field_returns_422():
    bad = {"records": [{"Id": 1, "LotArea": 8450}]}  # missing GrLivArea etc.
    with TestClient(app) as client:
        r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_extra_field_returns_422():
    bad_record = {**VALID_RECORD, "UnknownField": 99}
    with TestClient(app) as client:
        r = client.post("/predict", json={"records": [bad_record]})
    assert r.status_code == 422


def test_predict_wrong_type_returns_422():
    bad = {"records": [{"Id": 1, "LotArea": "not-a-number",
                        "GrLivArea": 1710, "Neighborhood": "CollgCr",
                        "OverallQual": 7, "YearBuilt": 2003}]}
    with TestClient(app) as client:
        r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_empty_records_returns_error():
    with TestClient(app) as client:
        r = client.post("/predict", json={"records": []})
    # empty list: DataCleanError (not ValueError) → 500, or Pydantic → 422
    assert r.status_code in (422, 500)


# ---------------------------------------------------------------------------
# /predict — model not loaded → 503
# ---------------------------------------------------------------------------

def test_predict_returns_503_when_model_missing():
    with TestClient(app) as client:
        app.state.model_pipeline = None
        r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 503
