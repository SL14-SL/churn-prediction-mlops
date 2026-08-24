import os
import socket

import mlflow

from src.inference.model_loader import (
    load_model_by_type,
)
from src.inference.releases.manifest import (
    resolve_release_artifact_uri,
)
from src.inference.releases.repository import (
    load_active_serving_manifest,
    load_serving_manifest,
)
from src.inference.releases.storage import (
    load_json,
)
from src.inference.serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def resolve_tracking_uri(
    cfg: dict,
) -> str:
    """
    Determine the MLflow tracking URI.

    Priority:
    1. MLFLOW_TRACKING_URI
    2. Docker MLflow service
    3. Configuration fallback
    """
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if tracking_uri is not None:
        return tracking_uri

    is_docker = os.path.exists(
        "/.dockerenv"
    )

    if is_docker:
        try:
            mlflow_ip = (
                socket.gethostbyname(
                    "mlflow"
                )
            )
            return (
                f"http://{mlflow_ip}:5000"
            )
        except Exception:
            return "http://mlflow:5000"

    tracking = cfg.get(
        "tracking",
        {},
    )

    if isinstance(tracking, dict):
        configured_uri = tracking.get(
            "mlflow_tracking_uri"
        )

        if configured_uri:
            return str(configured_uri)

    return str(
        cfg.get(
            "mlflow_tracking_uri",
            "http://localhost:5000",
        )
    )


def load_serving_bundle_for_release(
    *,
    release_id: str,
    model_name: str,
    cfg: dict,
    models_path: str,
) -> ServingBundle:
    """
    Load and validate one concrete churn serving release.

    This function does not change the active release pointer.
    """
    mlflow.set_tracking_uri(
        resolve_tracking_uri(cfg)
    )

    manifest, release_root = (
        load_serving_manifest(
            models_path=models_path,
            release_id=release_id,
        )
    )

    if manifest.model_name != model_name:
        raise ValueError(
            "Serving manifest model name does "
            "not match configuration: "
            f"{manifest.model_name} != "
            f"{model_name}"
        )

    feature_schema_uri = (
        resolve_release_artifact_uri(
            release_root=release_root,
            reference=(
                manifest.feature_schema
            ),
        )
    )

    feature_schema = load_json(
        feature_schema_uri
    )

    model = load_model_by_type(
        manifest.model_uri,
        manifest.model_type,
    )

    bundle = ServingBundle(
        release_id=manifest.release_id,
        manifest=manifest,
        model=model,
        model_name=manifest.model_name,
        model_type=manifest.model_type,
        decision_threshold=(
            manifest.decision_threshold
        ),
        feature_schema=feature_schema,
        serving_alias="champion",
        model_uri=manifest.model_uri,
        model_version=(
            manifest.model_version
        ),
        model_run_id=(
            manifest.model_run_id
        ),
    )

    validate_serving_bundle(
        bundle
    )

    logger.info(
        "Serving bundle loaded: "
        "release_id=%s model=%s "
        "version=%s run_id=%s",
        bundle.release_id,
        bundle.model_name,
        bundle.model_version,
        bundle.model_run_id,
    )

    return bundle


def reload_serving_model(
    *,
    model_name: str,
    cfg: dict,
) -> ServingBundle:
    """
    Load the currently active versioned serving release.
    """
    paths = cfg.get(
        "paths",
        {},
    )

    if not isinstance(paths, dict):
        raise ValueError(
            "Configuration has no valid paths section."
        )

    models_path = paths.get(
        "models"
    )

    if not models_path:
        raise ValueError(
            "Configuration has no models path."
        )

    manifest, _ = (
        load_active_serving_manifest(
            models_path=str(
                models_path
            ),
        )
    )

    return load_serving_bundle_for_release(
        release_id=manifest.release_id,
        model_name=model_name,
        cfg=cfg,
        models_path=str(
            models_path
        ),
    )