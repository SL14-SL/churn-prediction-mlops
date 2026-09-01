from pathlib import Path

from fastapi import HTTPException

from src.configs.loader import (
    get_path,
    load_config,
)
from src.inference.model_manager import (
    reload_serving_model as load_current_serving_bundle,
)
from src.inference.serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
)
from src.monitoring.serving import (
    set_serving_readiness,
)


CFG = load_config()
TRAIN_CFG = load_config(
    "training.yaml"
)

MODEL_NAME = CFG["model"]["registry_name"]
MODELS_PATH = Path(
    get_path("models")
)

_active_serving_bundle: (
    ServingBundle | None
) = None

_dq_reference_categories: dict[
    str,
    set[str],
] = {}


def get_active_serving_bundle() -> ServingBundle | None:
    """Return the active serving bundle, if one is loaded."""
    return _active_serving_bundle


def require_active_serving_bundle() -> ServingBundle:
    """
    Return the currently active serving bundle.

    A local reference ensures that one request uses one consistent bundle,
    even if another request reloads the model concurrently.

    Raises:
        HTTPException: If no complete serving bundle is active.
    """
    bundle = _active_serving_bundle

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No complete serving bundle "
                "is active."
            ),
        )

    return bundle


def activate_serving_bundle(
    bundle: ServingBundle,
) -> dict:
    """
    Validate and atomically activate a serving bundle.

    Returns:
        Identity and decision metadata for the activated bundle.
    """
    global _active_serving_bundle

    validate_serving_bundle(
        bundle
    )

    _active_serving_bundle = bundle
    set_serving_readiness(
        True
    )

    return {
        "release_id": (
            bundle.release_id
        ),
        "model_name": (
            bundle.model_name
        ),
        "serving_alias": (
            bundle.serving_alias
        ),
        "model_version": (
            bundle.model_version
        ),
        "model_run_id": (
            bundle.model_run_id
        ),
        "model_uri": (
            bundle.model_uri
        ),
        "decision_threshold": (
            bundle.decision_threshold
        ),
    }


def clear_serving_bundle() -> None:
    """Clear the active bundle and mark serving as not ready."""
    global _active_serving_bundle

    _active_serving_bundle = None
    set_serving_readiness(
        False
    )


def reload_serving_model() -> dict:
    """
    Load, validate and activate the configured serving release.

    The previous serving bundle remains active when loading or validation of
    its replacement fails.

    Returns:
        Identity and decision metadata for the activated bundle.
    """
    new_bundle = (
        load_current_serving_bundle(
            model_name=MODEL_NAME,
            cfg=CFG,
        )
    )

    return activate_serving_bundle(
        new_bundle
    )


def get_data_quality_reference_categories() -> dict[
    str,
    set[str],
]:
    """Return the cached categorical reference values for runtime checks."""
    return _dq_reference_categories


def set_data_quality_reference_categories(
    categories: dict[str, set[str]],
) -> None:
    """Replace the cached categorical reference values."""
    global _dq_reference_categories

    _dq_reference_categories = categories