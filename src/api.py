"""
Educational Goal:
- Why this module exists in an MLOps system: Turn the trained ML pipeline
  into a callable web service so external systems can request predictions
  without touching training code.
- Responsibility (separation of concerns): HTTP routing, schema validation,
  startup model loading, and response formatting only. No ML logic lives here.
- Pipeline contract (inputs and outputs): Accepts JSON payloads with house
  features, returns predicted SalePrice. Reuses clean_housing_data(),
  validate_dataframe(), and run_inference() from existing src/ modules.

What this module owns:
- Pydantic request/response schemas (the strict data contract)
- FastAPI app and endpoint definitions
- Model loading at startup (local file or W&B artifact)

What this module does NOT own:
- Any ML logic — no new transformations, no new models
- Training or evaluation code
- Direct calls to scikit-learn

Key principle: api.py only does HTTP + schema validation + routing.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.clean_data import clean_housing_data
from src.infer import run_inference
from src.validate import validate_dataframe

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config() -> dict:
    """
    Inputs:
    - None (reads from the fixed config.yaml path at repo root)

    Outputs:
    - Parsed configuration dictionary

    Why this contract matters for reliable ML delivery:
    - Ensures the API reads the same runtime settings as the training pipeline.
    - W&B project and artifact names stay in one place: config.yaml.
    """
    with _CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------------
# 1) Pydantic Schemas (The Strict API Contract)
# -------------------------------------------------------------------

class HouseRecord(BaseModel):
    """
    One row of house features sent by the caller.

    Mirrors config.yaml features exactly:
        quantile_bin:        LotArea, GrLivArea
        categorical_onehot:  Neighborhood
        numeric_passthrough: OverallQual, YearBuilt

    Id is optional — if provided it is echoed back in the response
    so callers can match predictions to their own records.

    extra="forbid" means unexpected fields are rejected before our
    ML code ever runs — same principle as the opioid repo contract.
    """
    model_config = ConfigDict(extra="forbid")

    Id:           Optional[int] = None
    LotArea:      float
    GrLivArea:    float
    Neighborhood: str
    OverallQual:  int
    YearBuilt:    int


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[HouseRecord]


class PredictionItem(BaseModel):
    Id:        Optional[int]
    SalePrice: float


class PredictResponse(BaseModel):
    model_version: str
    predictions:   list[PredictionItem]


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str


# -------------------------------------------------------------------
# 2) Model Loading Helpers
# -------------------------------------------------------------------

def _load_from_local(model_path: Path):
    """
    Inputs:
    - model_path: Path to the local model.joblib file

    Outputs:
    - (pipeline, version_string) tuple, or (None, "missing") if not found

    Why this contract matters for reliable ML delivery:
    - Fail-fast on startup prevents silent 500 errors at prediction time.
    - Supports both plain pipelines and the dict artifact saved by main.py.
    """
    if not model_path.exists():
        logger.error("[api] Model file not found at %s", model_path)
        return None, "missing"

    artifact = joblib.load(model_path)

    # main.py saves {"pipeline": ..., "metadata": {...}}
    if isinstance(artifact, dict) and "pipeline" in artifact:
        version = artifact.get("metadata", {}).get("version", model_path.name)
        return artifact["pipeline"], version

    # Fallback: plain pipeline saved directly
    return artifact, model_path.name


def _load_from_wandb(wandb_project: str, wandb_artifact_name: str):
    """
    Inputs:
    - wandb_project: W&B project name (read from config.yaml)
    - wandb_artifact_name: artifact name to fetch (read from config.yaml)

    Outputs:
    - (pipeline, artifact_path_string) tuple

    Why this contract matters for reliable ML delivery:
    - Production inference always uses the promoted 'prod' artifact from
      the W&B registry, never an unmanaged local file.
    - Alias 'prod' is set manually in W&B after a human reviews the run,
      which enforces a deliberate promotion gate before deployment.
    """
    import wandb  # lazy import — keeps tests fast when W&B is not needed

    api_key = os.getenv("WANDB_API_KEY")
    entity  = os.getenv("WANDB_ENTITY")
    alias   = os.getenv("WANDB_MODEL_ALIAS", "prod")

    if not api_key or not entity:
        raise ValueError(
            "MODEL_SOURCE=wandb requires WANDB_API_KEY and WANDB_ENTITY in .env"
        )

    wandb.login(key=api_key, relogin=True)
    api           = wandb.Api()
    artifact_path = f"{entity}/{wandb_project}/{wandb_artifact_name}:{alias}"
    artifact      = api.artifact(artifact_path)
    artifact_dir  = artifact.download()

    model_file = Path(artifact_dir) / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(
            f"model.joblib not found inside downloaded artifact at {artifact_dir}"
        )

    saved    = joblib.load(model_file)
    pipeline = saved["pipeline"] if isinstance(saved, dict) and "pipeline" in saved else saved

    logger.info("[api] Model loaded from W&B: %s", artifact_path)
    return pipeline, artifact_path


# -------------------------------------------------------------------
# 3) Lifespan: Load shared resources once at API startup
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inputs:
    - None (reads MODEL_SOURCE from .env, paths from config.yaml)

    Outputs:
    - Sets app.state.model_pipeline and app.state.model_version

    Why this contract matters for reliable ML delivery:
    - Loading the model once at startup avoids reloading on every request.
    - Separating local vs W&B loading keeps the serving environment
      identical whether running on a laptop or on Render.
    """
    cfg          = _load_config()
    model_source = os.getenv("MODEL_SOURCE", "local").lower()
    project_root = Path(__file__).resolve().parents[1]

    wandb_project       = cfg["wandb"]["project"]
    wandb_artifact_name = cfg["wandb"]["model_artifact_name"]

    try:
        if model_source == "wandb":
            logger.info("[api] MODEL_SOURCE=wandb — fetching from W&B registry")
            pipeline, version = _load_from_wandb(wandb_project, wandb_artifact_name)
        else:
            logger.info("[api] MODEL_SOURCE=local — loading from disk")
            model_path        = project_root / cfg["output"]["model_path"]
            pipeline, version = _load_from_local(model_path)

        app.state.model_pipeline = pipeline
        app.state.model_version  = version

        if pipeline is None:
            logger.error("[api] Model not loaded — /predict will return 503")
        else:
            logger.info("[api] Startup complete | model_version=%s", version)

    except Exception as e:
        logger.exception("[api] Startup failed: %s", str(e))
        app.state.model_pipeline = None
        app.state.model_version  = "startup_error"

    yield

    logger.info("[api] Shutdown complete")


# -------------------------------------------------------------------
# 4) App + Endpoints
# -------------------------------------------------------------------

app = FastAPI(
    title="Smart Residential Price Estimation API",
    description=(
        "Predicts house sale prices for the Ames, Iowa dataset "
        "using a Lasso regression pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", summary="Show basic API instructions")
def root() -> dict:
    return {"message": "Use /docs, /health, or /predict"}


@app.get("/health", response_model=HealthResponse, summary="Check API health")
def health() -> HealthResponse:
    """
    Inputs:
    - None

    Outputs:
    - HealthResponse with status, model_loaded flag, and model_version

    Why this contract matters for reliable ML delivery:
    - Render and Docker use this endpoint to verify the service is ready
      before routing real traffic to it.
    """
    model_loaded  = getattr(app.state, "model_pipeline", None) is not None
    model_version = getattr(app.state, "model_version", "unloaded")

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_version=model_version,
    )


@app.post("/predict", response_model=PredictResponse, summary="Generate a house price prediction")
def predict(req: PredictRequest) -> PredictResponse:
    """
    Inputs:
    - req: PredictRequest with a list of HouseRecord objects

    Outputs:
    - PredictResponse with model_version and a SalePrice per record

    Why this contract matters for reliable ML delivery:
    - Reusing clean_housing_data(), validate_dataframe(), and run_inference()
      guarantees the API applies identical transformations to what was used
      during training — preventing training-serving skew.
    - No new ML logic in this file means bugs in predictions trace back
      to the shared pipeline modules, not to the API layer.
    """
    pipeline      = getattr(app.state, "model_pipeline", None)
    model_version = getattr(app.state, "model_version", "unloaded")

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check startup logs and MODEL_SOURCE in .env.",
        )

    try:
        records_dicts = [r.model_dump() for r in req.records]
        df_raw        = pd.DataFrame(records_dicts)

        # Preserve Id before cleaning removes it
        ids = df_raw["Id"].tolist() if "Id" in df_raw.columns else [None] * len(df_raw)

        # Clean — reuses existing module, same logic as training
        clean_result = clean_housing_data(
            df_raw,
            target_col="SalePrice",   # not present at inference time → y=None, that is fine
            drop_cols=["Id"],
            require_target=False,
        )
        X = clean_result.X

        # Validate — reuses existing module, same logic as training
        required_cols = ["LotArea", "GrLivArea", "Neighborhood", "OverallQual", "YearBuilt"]
        validate_dataframe(X, required_columns=required_cols)

        # Infer — reuses existing module, same logic as training
        artifact_payload = {
            "pipeline": pipeline,
            "metadata": {"target_transform": "log1p"},
        }
        preds_df = run_inference(
            input_df=X,
            artifact=artifact_payload,
            id_col="Id",
            pred_col="SalePrice",
        )

        predictions = [
            PredictionItem(
                Id=ids[i],
                SalePrice=float(preds_df["SalePrice"].iloc[i]),
            )
            for i in range(len(preds_df))
        ]

        return PredictResponse(
            model_version=model_version,
            predictions=predictions,
        )

    except ValueError as exc:
        logger.warning("[api] Validation error in /predict: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[api] Unexpected error in /predict")
        raise HTTPException(status_code=500, detail="Internal server error") from exc