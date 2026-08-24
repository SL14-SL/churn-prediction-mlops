import json
import os
import socket
from typing import Any

import mlflow
from mlflow import MlflowClient

from src.inference.router import load_registry_model
from src.inference.schema import load_feature_schema
from src.inference.serving_bundle import ServingBundle, validate_serving_bundle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_tracking_uri(cfg: dict) -> str:
    """
    Determine the MLflow tracking URI based on environment.

    Priority:
    1. MLFLOW_TRACKING_URI env var
    2. Docker service hostname
    3. config fallback
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri is not None:
        return tracking_uri

    is_docker = os.path.exists("/.dockerenv")

    if is_docker:
        try:
            mlflow_ip = socket.gethostbyname("mlflow")
            return f"http://{mlflow_ip}:5000"
        except Exception:
            return "http://mlflow:5000"

    return cfg.get("mlflow_tracking_uri", "http://localhost:5000")


def load_feature_schema_from_mlflow(
    *,
    run_id: str,
    fallback_to_local: bool = True,
) -> dict[str, Any]:
    """
    Load the feature schema from the MLflow run artifacts.

    The schema is logged during training under:
    feature_schema/feature_schema.json

    Falls back to the local models/feature_schema.json for local development.
    """
    client = MlflowClient()

    try:
        local_path = client.download_artifacts(
            run_id=run_id,
            path="feature_schema/feature_schema.json",
        )

        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as exc:
        logger.warning(
            "Could not load feature schema from MLflow artifacts "
            "(run_id=%s): %s",
            run_id,
            exc,
        )

        if fallback_to_local:
            logger.warning("Falling back to local feature schema.")
            return load_feature_schema()

        raise


def reload_serving_model(
    *,
    model_name: str,
    cfg: dict,
) -> ServingBundle:
    """
    Load and validate the complete serving state.

    The caller receives a bundle only after the model, feature schema,
    decision threshold, and registry metadata have been loaded and
    validated successfully.
    """
    mlflow.set_tracking_uri(
        resolve_tracking_uri(cfg)
    )

    (
        model,
        model_type,
        serving_alias,
        model_uri,
        decision_threshold,
    ) = load_registry_model(model_name)

    if (
        not serving_alias
        or serving_alias == "unknown"
    ):
        raise RuntimeError(
            "Cannot load a serving bundle because no valid "
            f"serving alias was resolved for model '{model_name}'."
        )

    client = MlflowClient()
    version = client.get_model_version_by_alias(
        model_name,
        serving_alias,
    )

    model_version = str(
        version.version
    )
    model_run_id = version.run_id

    feature_schema = (
        load_feature_schema_from_mlflow(
            run_id=model_run_id,
            fallback_to_local=not bool(
                os.getenv("K_SERVICE")
            ),
        )
    )

    bundle = ServingBundle(
        model=model,
        model_name=model_name,
        model_type=model_type,
        decision_threshold=float(
            decision_threshold
        ),
        feature_schema=feature_schema,
        serving_alias=serving_alias,
        model_uri=model_uri,
        model_version=model_version,
        model_run_id=model_run_id,
    )

    validate_serving_bundle(bundle)

    logger.info(
        "Serving bundle loaded: %s "
        "(alias=%s, version=%s, run_id=%s)",
        bundle.model_name,
        bundle.serving_alias,
        bundle.model_version,
        bundle.model_run_id,
    )

    return bundle