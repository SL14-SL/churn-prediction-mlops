import os
import traceback
import time
import io
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Security, Depends, Response, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi_swagger_ui_theme import setup_swagger_ui_theme

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.status import HTTP_403_FORBIDDEN

from src.api.schema import (
    PredictionRequest, 
    PredictionResponse, 
    PrioritizeRequest, 
    CampaignSimulationRequest,
    ServingRollbackRequest,
)

from src.configs.loader import load_config, get_path

from src.monitoring.prediction_logger import log_prediction
from src.monitoring.data_quality import (
    initialize_data_quality_reference_cache, 
    build_reference_category_cache, 
)
from src.monitoring.config import get_serving_settings, get_data_quality_settings
from src.monitoring.serving import (
    normalize_path,
    observe_request,
    set_serving_readiness,
    should_ignore_path,
)
from src.data.features.build_features import build_features

from src.inference.pipeline import (
    validate_prediction_input, 
    align_features_for_model,
    predict_and_decide,
)
from src.inference.adapters import request_to_dataframe
from src.inference.explain import explain_single_prediction
from src.inference.model_manager import reload_serving_model as reload_model_state, load_serving_bundle_for_release
from src.inference.serving_bundle import ServingBundle, validate_serving_bundle

from src.inference.releases.repository import activate_release_pointer, load_active_release_id

from src.api.services import (
    run_prediction_pipeline,
    attach_customer_ids,
    prioritize_results,
    compute_business_kpis,
    simulate_campaign,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- 1. Load configuration and paths ---
CFG = load_config()
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = CFG["model"]["registry_name"]
MODELS_PATH = Path(get_path("models"))

active_serving_bundle: (
    ServingBundle | None
) = None

dq_reference_categories: dict[
    str,
    set[str],
] = {}

def require_active_serving_bundle() -> ServingBundle:
    """
    Return the currently active serving bundle.

    A local reference ensures one request uses one consistent bundle,
    even if another request reloads the model concurrently.
    """
    bundle = active_serving_bundle

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="No complete serving bundle is active.",
        )

    return bundle

def activate_serving_bundle(
    bundle: ServingBundle,
) -> dict:
    """
    Atomically activate an already validated serving bundle.
    """
    global active_serving_bundle

    validate_serving_bundle(bundle)

    active_serving_bundle = bundle
    set_serving_readiness(True)

    return {
        "release_id": bundle.release_id,
        "model_name": bundle.model_name,
        "serving_alias": (
            bundle.serving_alias
        ),
        "model_version": (
            bundle.model_version
        ),
        "model_run_id": (
            bundle.model_run_id
        ),
        "model_uri": bundle.model_uri,
        "decision_threshold": (
            bundle.decision_threshold
        ),
    }

# API Key Security Configuration
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validates the API Key from the request header."""
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate API Key",
    )


def reload_serving_model() -> dict:
    """
    Atomically replace the active serving bundle.

    The previous bundle remains active if loading or validation of the
    replacement fails.
    """
    new_bundle = reload_model_state(
        model_name=MODEL_NAME,
        cfg=CFG,
    )

    return activate_serving_bundle(
        new_bundle
    )

    

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown.
    Loads the ML model from registry and initializes data quality caches.
    """
    global active_serving_bundle
    global dq_reference_categories

    try:
        if os.getenv("SMOKE_TEST") == "1":
            logger.info("Smoke test mode enabled. Skipping model and data quality startup loading.")
            yield
            return

        # --- Initialize Data Quality Cache ---
        ref_df = initialize_data_quality_reference_cache()
        dq_reference_categories = build_reference_category_cache(
            ref_df,
            categorical_reference_features=get_data_quality_settings().get(
                "categorical_reference_features", []
            ),
        )

        # --- Load Model from MLflow ---
        try:
            reload_serving_model()

            logger.info(
                "Model loaded: %s (version=%s)",
                active_serving_bundle.model_name,
                active_serving_bundle.model_version,
            )
        except Exception as model_err:
            logger.error(
                "Failed to load model from registry: "
                f"{model_err}"
            )
            active_serving_bundle = None
            set_serving_readiness(False)

        yield

    finally:
        set_serving_readiness(False)
        logger.info(
            "Shutdown: Cleaning up resources."
        )
        

app = FastAPI(
    title="Churn Prediction API", 
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    )

SERVING_CFG = get_serving_settings()


setup_swagger_ui_theme(
    app,
    docs_path="/docs",
    title="Churn Prediction API Docs",
)

# --- Middleware for Monitoring & Prometheus ---
@app.middleware("http")
async def serving_monitoring_middleware(request: Request, call_next):
    if not SERVING_CFG.get("enabled", True):
        return await call_next(request)

    raw_path = request.url.path
    if should_ignore_path(raw_path, SERVING_CFG.get("ignored_paths")):
        return await call_next(request)

    method = request.method
    path = normalize_path(raw_path, SERVING_CFG.get("track_paths"))
    start = time.perf_counter()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        observe_request(
            method=method,
            path=path,
            status_code=status_code,
            latency_seconds=time.perf_counter() - start,
        )

@app.get("/metrics", include_in_schema=False)
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/reload-model")
def reload_model(api_key: str = Depends(get_api_key)):
    """
    Reload the current champion model from MLflow.

    Used after a new champion model version has been promoted.
    """
    try:
        result = reload_serving_model()
    except Exception as e:
        logger.error(f"Model reload failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

    return {
        "status": "reloaded",
        **result,
    }

@app.post(
    "/admin/rollback-serving-release"
)
def rollback_serving_release(
    payload: ServingRollbackRequest,
    api_key: str = Depends(
        get_api_key
    ),
):
    """
    Validate and atomically activate a previously published release.
    """
    previous_bundle = (
        require_active_serving_bundle()
    )
    previous_release_id = (
        previous_bundle.release_id
    )

    stored_release_id = (
        load_active_release_id(
            models_path=MODELS_PATH,
        )
    )

    if (
        stored_release_id
        != previous_release_id
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "In-memory serving release does "
                "not match the active pointer."
            ),
        )

    if (
        payload.release_id
        == previous_release_id
    ):
        return {
            "status": "unchanged",
            "release_id": (
                previous_release_id
            ),
            "previous_release_id": (
                previous_release_id
            ),
        }

    pointer_changed = False

    try:
        # Load and validate everything before changing either active state.
        candidate_bundle = (
            load_serving_bundle_for_release(
                release_id=(
                    payload.release_id
                ),
                model_name=MODEL_NAME,
                cfg=CFG,
                models_path=MODELS_PATH,
            )
        )

        activate_release_pointer(
            models_path=MODELS_PATH,
            release_id=(
                payload.release_id
            ),
            operation="rollback",
            previous_release_id=(
                previous_release_id
            ),
        )
        pointer_changed = True

        result = activate_serving_bundle(
            candidate_bundle
        )

    except Exception as error:
        if pointer_changed:
            try:
                activate_release_pointer(
                    models_path=MODELS_PATH,
                    release_id=(
                        previous_release_id
                    ),
                    operation=(
                        "rollback_reverted"
                    ),
                    previous_release_id=(
                        payload.release_id
                    ),
                )
            except Exception:
                logger.exception(
                    "CRITICAL: rollback pointer "
                    "could not be restored | "
                    "expected_release_id=%s",
                    previous_release_id,
                )

        logger.exception(
            "Serving release rollback failed | "
            "target_release_id=%s | "
            "previous_release_id=%s",
            payload.release_id,
            previous_release_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Serving release rollback failed. "
                f"Reason: {error}"
            ),
        ) from error

    logger.warning(
        "Serving release rollback completed | "
        "previous_release_id=%s | "
        "active_release_id=%s",
        previous_release_id,
        candidate_bundle.release_id,
    )

    return {
        "status": "rolled_back",
        "previous_release_id": (
            previous_release_id
        ),
        **result,
    }

@app.get("/health")
def health(response: Response):
    bundle = active_serving_bundle
    is_healthy = bundle is not None

    if not is_healthy:
        response.status_code = 503

    return {
        "status": (
            "online"
            if is_healthy
            else "degraded"
        ),
        "model_name": (
            bundle.model_name
            if bundle
            else MODEL_NAME
        ),
        "serving_alias": (
            bundle.serving_alias
            if bundle
            else None
        ),
        "model_version": (
            bundle.model_version
            if bundle
            else None
        ),
    }

@app.get("/livez")
def livez():
    """
    Liveness probe.

    Returns 200 if the API process is running.
    Does not check whether a model is loaded.
    """
    return {
        "status": "alive",
        "service": CFG.get("project_name", "churn-prediction-api"),
        "environment": CFG.get("environment", "unknown"),
    }

@app.get("/readyz")
def readyz():
    """
    Return 200 only when one complete serving bundle is active.
    """
    bundle = require_active_serving_bundle()

    return {
        "status": "ready",
        "serving_bundle_loaded": True,
        "release_id": bundle.release_id,
        "model_name": bundle.model_name,
        "model_type": bundle.model_type,
        "serving_alias": bundle.serving_alias,
        "model_version": bundle.model_version,
        "model_run_id": bundle.model_run_id,
        "model_uri": bundle.model_uri,
        "decision_threshold": (
            bundle.decision_threshold
        ),
        "feature_schema_loaded": bool(
            bundle.feature_schema.get(
                "columns"
            )
        ),
        "decision_threshold_loaded": True,
    }

@app.post("/explain", dependencies=[Depends(get_api_key)])
def explain(payload: PredictionRequest, top_n: int = 5):
    """
    Return churn prediction with top feature-level explanation.

    Intended for debugging, demos, and customer-level model interpretation.
    """
    bundle = require_active_serving_bundle()

    if len(payload.inputs) != 1:
        raise HTTPException(
            status_code=400,
            detail="Explain endpoint currently supports exactly one input row.",
        )

    try:
        input_df = request_to_dataframe(payload.inputs)
        validated_df = validate_prediction_input(input_df)
        processed_df = build_features(validated_df, config=TRAIN_CFG)

        final_df = align_features_for_model(
            processed_df=processed_df,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
        )

        prediction = predict_and_decide(
            input_df=final_df,
            model=bundle.model,
        )[0]

        explanation = explain_single_prediction(
            final_df=final_df,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            top_n=top_n,
        )

        return {
            "status": "success",
            "prediction": prediction,
            "top_reasons": explanation,
            "metadata": {
                "model_name": MODEL_NAME,
                "serving_alias": bundle.serving_alias,
                "model_version": bundle.model_version,
            },
        }

    except Exception as e:
        logger.error(f"Explanation failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", dependencies=[Depends(get_api_key)], response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    bundle = require_active_serving_bundle()

    try:
        output = run_prediction_pipeline(
            payload=payload,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            dq_reference_categories=dq_reference_categories,
            decision_threshold=bundle.decision_threshold,
        )

        results = attach_customer_ids(
            payload.inputs,
            output["results"],
        )

        for features, result in zip(payload.inputs, results):
            log_prediction(
                features,
                result["churn_probability"],
                model_alias=bundle.serving_alias,
                model_version=bundle.model_version,
                model_run_id=bundle.model_run_id,
                request_id=output["request_id"],
                environment=output["environment"],
                action=result["action"],
                expected_value=result["expected_value"],
                customer_value=result.get("customer_value"),
            )

        return {
            "predictions": results,
            "status": "success",
            "metadata": {
                "rows": len(results),
                "model_name": MODEL_NAME,
                "serving_alias": bundle.serving_alias,
                "request_id": output["request_id"],
                "timing_ms": output["timings"],
                "data_quality": output["dq_summary"],
                "release_id": bundle.release_id,
                "model_version": bundle.model_version,
                "model_run_id": bundle.model_run_id,
            },
        }

    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/prioritize", dependencies=[Depends(get_api_key)], response_model=PredictionResponse)
def prioritize(payload: PrioritizeRequest):
    bundle = require_active_serving_bundle()

    try:
        output = run_prediction_pipeline(
            payload=payload,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            dq_reference_categories=dq_reference_categories,
            decision_threshold=bundle.decision_threshold,
        )

        enriched = attach_customer_ids(payload.inputs, output["results"])

        prioritized = prioritize_results(
            enriched,
            top_n=payload.top_n,
            min_expected_value=payload.min_expected_value,
        )        

        business_kpis = compute_business_kpis(prioritized)

        return {
            "predictions": prioritized,
            "status": "success",
            "metadata": {
                "rows": len(prioritized),
                "total_input_rows": len(payload.inputs),
                "top_n": payload.top_n,
                "business_kpis": business_kpis,
                "min_expected_value": payload.min_expected_value,
                "model_name": MODEL_NAME,
                "serving_alias": bundle.serving_alias,
                "request_id": output["request_id"],
                "timing_ms": output["timings"],
                "data_quality": output["dq_summary"],
            },
        }

    except Exception as e:
        logger.exception("Prioritization failed")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/prioritize/export", dependencies=[Depends(get_api_key)])
def export_prioritized(payload: PrioritizeRequest):
    """
    Export prioritized customers as CSV.

    Uses the same pipeline as /prioritize but returns a downloadable CSV file.
    """
    bundle = require_active_serving_bundle()

    try:
        # 1. Run shared pipeline
        output = run_prediction_pipeline(
            payload=payload,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            dq_reference_categories=dq_reference_categories,
            decision_threshold=bundle.decision_threshold,
        )

        # 2. Attach IDs + prioritize
        enriched = attach_customer_ids(payload.inputs, output["results"])
        prioritized = prioritize_results(
            enriched,
            top_n=payload.top_n,
            min_expected_value=payload.min_expected_value,
        )

        # 3. Convert to DataFrame
        df = pd.DataFrame(prioritized)

        # Optional: nicer column order
        preferred_cols = [
            "customer_id",
            "churn_probability",
            "customer_value",
            "action",
            "expected_value",
        ]

        cols = [c for c in preferred_cols if c in df.columns]
        df = df[cols]

        # 4. Convert to CSV
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        # 5. Filename
        date_str = datetime.utcnow().strftime("%y%m%d")

        filename = f"{date_str}_prioritized_customers.csv"

        # 6. Return as file download
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )

    except Exception as e:
        logger.exception("CSV export failed")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/campaign/simulate", dependencies=[Depends(get_api_key)])
def simulate_retention_campaign(payload: CampaignSimulationRequest):
    """
    Simulate a retention campaign based on prioritized churn decisions.

    Returns campaign-level business impact metrics:
    - total expected value
    - action counts
    - targeted customers
    - actionable customers
    """
    bundle = require_active_serving_bundle()

    try:
        output = run_prediction_pipeline(
            payload=payload,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            dq_reference_categories=dq_reference_categories,
            decision_threshold=bundle.decision_threshold,
        )

        enriched = attach_customer_ids(payload.inputs, output["results"])

        prioritized = prioritize_results(
            enriched,
            top_n=payload.top_n,
            min_expected_value=payload.min_expected_value,
        )

        simulation = simulate_campaign(prioritized)

        return {
            "status": "success",
            "campaign": {
                "name": payload.campaign_name or "retention_campaign",
                "top_n": payload.top_n,
                "min_expected_value": payload.min_expected_value,
                **simulation,
            },
            "metadata": {
                "total_input_rows": len(payload.inputs),
                "selected_rows": len(prioritized),
                "model_name": MODEL_NAME,
                "serving_alias": bundle.serving_alias,
                "request_id": output["request_id"],
                "timing_ms": output["timings"],
                "data_quality": output["dq_summary"],
            },
        }

    except Exception as e:
        logger.exception("Campaign simulation failed")
        raise HTTPException(status_code=500, detail=str(e))