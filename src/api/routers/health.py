from fastapi import (
    APIRouter,
    Response,
)
from fastapi.responses import (
    PlainTextResponse,
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from src.api.serving_state import (
    CFG,
    MODEL_NAME,
    get_active_serving_bundle,
    require_active_serving_bundle,
)


router = APIRouter(
    tags=["health"],
)


@router.get(
    "/metrics",
    include_in_schema=False,
)
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics."""
    return PlainTextResponse(
        generate_latest().decode(
            "utf-8"
        ),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/health")
def health(
    response: Response,
) -> dict:
    """Return compatibility health information for the active serving bundle."""
    bundle = (
        get_active_serving_bundle()
    )
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


@router.get("/livez")
def livez() -> dict:
    """
    Return process liveness without requiring a loaded model.

    Returns:
        Service identity and environment metadata.
    """
    return {
        "status": "alive",
        "service": CFG.get(
            "project_name",
            "churn-prediction-api",
        ),
        "environment": CFG.get(
            "environment",
            "unknown",
        ),
    }


@router.get("/readyz")
def readyz() -> dict:
    """
    Return readiness and lineage for the active serving bundle.

    Raises:
        HTTPException: If no complete serving bundle is active.
    """
    bundle = (
        require_active_serving_bundle()
    )

    return {
        "status": "ready",
        "serving_bundle_loaded": True,
        "release_id": bundle.release_id,
        "model_name": bundle.model_name,
        "model_type": bundle.model_type,
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
        "feature_schema_loaded": bool(
            bundle.feature_schema.get(
                "columns"
            )
        ),
        "decision_threshold_loaded": True,
    }