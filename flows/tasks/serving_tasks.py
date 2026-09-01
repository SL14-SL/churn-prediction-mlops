import os
import hashlib
import json
import requests

from mlflow.tracking import (
    MlflowClient,
)
from prefect import (
    get_run_logger,
    task,
)

from src.configs.loader import (
    get_path,
    load_config,
)
from src.deployment.verification import (
    verify_prediction_probe,
    verify_serving_release,
)
from src.inference.releases.repository import (
    load_active_release_id,
    load_release_prediction_probe,
    load_serving_release_manifest,
)

from src.deployment.prediction_probe import build_prediction_probe
from src.inference.releases.publisher import publish_serving_release


ENV_CFG = load_config()
MODEL_NAME = ENV_CFG["model"]["registry_name"]


def resolve_api_base_url() -> str:
    """Resolve the API base URL from a configured prediction endpoint."""
    api_url = (
        ENV_CFG.get(
            "api",
            {},
        ).get(
            "url",
            "http://api:8080/predict",
        )
    )

    if api_url.endswith(
        "/predict"
    ):
        return api_url.removesuffix(
            "/predict"
        )

    return api_url.rstrip("/")


def require_api_key() -> str:
    """
    Return the configured API key.

    Raises:
        RuntimeError: If no API key is available.
    """
    api_key = os.getenv(
        "API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "API_KEY environment variable "
            "is not set."
        )

    return api_key


@task(name="Resolve Previous Serving Release")
def task_resolve_previous_release() -> (
    str | None
):
    """
    Resolve the active release before publishing a replacement.
    """
    p_logger = get_run_logger()

    try:
        release_id = (
            load_active_release_id(
                models_path=get_path(
                    "models"
                ),
            )
        )
    except FileNotFoundError:
        p_logger.info(
            "No previous serving release "
            "exists. This is expected "
            "during bootstrap."
        )
        return None

    p_logger.info(
        "Previous serving release "
        "resolved | release_id=%s",
        release_id,
    )

    return release_id


@task(name="Refresh API Serving State")
def task_refresh_api() -> dict:
    """
    Reload the active release into the API process.
    """
    p_logger = get_run_logger()

    reload_url = (
        f"{resolve_api_base_url()}"
        "/admin/reload-model"
    )

    p_logger.info(
        "Refreshing API serving state via %s",
        reload_url,
    )

    response = requests.post(
        reload_url,
        headers={
            "X-API-KEY": require_api_key(),
        },
        timeout=300,
    )
    response.raise_for_status()

    result = response.json()

    p_logger.info(
        "API serving state reloaded | "
        "release_id=%s model_version=%s",
        result.get("release_id"),
        result.get("model_version"),
    )

    return result


@task(name="Verify Serving Release")
def task_verify_serving_release(
    *,
    expected_release_id: str,
) -> dict:
    """
    Verify readiness, lineage, and semantic prediction behavior.
    """
    p_logger = get_run_logger()
    api_base_url = (
        resolve_api_base_url()
    )
    models_path = get_path(
        "models"
    )

    readiness_result = (
        verify_serving_release(
            api_base_url=api_base_url,
            expected_release_id=(
                expected_release_id
            ),
        )
    )

    manifest = (
        load_serving_release_manifest(
            models_path=models_path,
            release_id=(
                expected_release_id
            ),
        )
    )

    if (
        str(
            readiness_result.model_version
        )
        != str(
            manifest.model_version
        )
    ):
        raise RuntimeError(
            "Ready endpoint model version "
            "does not match release manifest | "
            "ready="
            f"{readiness_result.model_version} | "
            "manifest="
            f"{manifest.model_version}"
        )

    if (
        readiness_result.model_run_id
        != manifest.model_run_id
    ):
        raise RuntimeError(
            "Ready endpoint model run ID "
            "does not match release manifest | "
            "ready="
            f"{readiness_result.model_run_id} | "
            "manifest="
            f"{manifest.model_run_id}"
        )

    prediction_probe_payload = (
        load_release_prediction_probe(
            models_path=models_path,
            release_id=(
                expected_release_id
            ),
        )
    )

    if prediction_probe_payload is None:
        raise RuntimeError(
            "Serving release has no "
            "prediction probe."
        )

    probe_result = (
        verify_prediction_probe(
            api_base_url=api_base_url,
            api_key=require_api_key(),
            prediction_probe_payload=(
                prediction_probe_payload
            ),
            expected_release_id=(
                manifest.release_id
            ),
            expected_model_version=(
                manifest.model_version
            ),
            expected_model_run_id=(
                manifest.model_run_id
            ),
        )
    )

    p_logger.info(
        "Serving release verified | "
        "release_id=%s model_version=%s",
        probe_result.release_id,
        probe_result.model_version,
    )

    return {
        "release_id": (
            readiness_result.release_id
        ),
        "model_version": (
            readiness_result.model_version
        ),
        "model_run_id": (
            readiness_result.model_run_id
        ),
        "readiness_attempts": (
            readiness_result.attempts
        ),
        "prediction_probe_status": (
            "verified"
        ),
        "prediction_probe_attempts": (
            probe_result.attempts
        ),
        "probe_probabilities": list(
            probe_result.probabilities
        ),
    }


@task(name="Rollback Serving Release")
def task_rollback_serving_release(
    *,
    previous_release_id: str,
) -> dict:
    """
    Roll back the API release and restore the MLflow Champion alias.
    """
    p_logger = get_run_logger()
    api_base_url = (
        resolve_api_base_url()
    )

    p_logger.warning(
        "Starting automatic rollback | "
        "target_release_id=%s",
        previous_release_id,
    )

    response = requests.post(
        (
            f"{api_base_url}"
            "/admin/rollback-serving-release"
        ),
        json={
            "release_id": (
                previous_release_id
            ),
        },
        headers={
            "X-API-KEY": require_api_key(),
        },
        timeout=300,
    )
    response.raise_for_status()

    api_result = response.json()

    previous_manifest = (
        load_serving_release_manifest(
            models_path=get_path(
                "models"
            ),
            release_id=(
                previous_release_id
            ),
        )
    )

    client = MlflowClient()

    client.set_registered_model_alias(
        name=(
            previous_manifest.model_name
        ),
        alias="champion",
        version=str(
            previous_manifest.model_version
        ),
    )

    p_logger.warning(
        "Automatic rollback completed | "
        "release_id=%s model_version=%s",
        previous_release_id,
        previous_manifest.model_version,
    )

    return {
        "release_id": (
            previous_release_id
        ),
        "model_name": (
            previous_manifest.model_name
        ),
        "model_version": str(
            previous_manifest.model_version
        ),
        "model_run_id": (
            previous_manifest.model_run_id
        ),
        "api_result": api_result,
    }


def build_config_hash(
    dataset_manifest: dict,
) -> str | None:
    """
    Build a deterministic hash of the effective training configuration.
    """
    effective_config = (
        dataset_manifest.get(
            "effective_config"
        )
    )

    if effective_config is None:
        return None

    serialized = json.dumps(
        effective_config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


@task(name="Publish Serving Release")
def task_publish_serving_release(
    *,
    registration_result: dict,
    dataset_manifest: dict,
):
    """
    Publish and activate the complete churn serving release.
    """
    
    if not registration_result.get(
        "promoted",
        False,
    ):
        raise ValueError(
            "Cannot publish a serving release "
            "for a non-promoted model."
        )
    
    p_logger = get_run_logger()
    
    models_path = get_path(
        "models"
    )
    validated_path = get_path(
        "validated_data"
    )

    feature_schema_source = (
        f"{models_path}/feature_schema.json"
    )
    validated_data_path = (
        f"{validated_path}/train.parquet"
    )

    prediction_probe = (
        build_prediction_probe(
            validated_data_path=(
                validated_data_path
            ),
        )
    )

    manifest = publish_serving_release(
        models_path=models_path,
        model_name=MODEL_NAME,
        model_version=(
            registration_result[
                "model_version"
            ]
        ),
        model_run_id=(
            registration_result[
                "model_run_id"
            ]
        ),
        model_type=(
            registration_result[
                "model_type"
            ]
        ),
        decision_threshold=(
            registration_result[
                "decision_threshold"
            ]
        ),
        dataset_version=(
            dataset_manifest.get(
                "dataset_version"
            )
        ),
        config_hash=build_config_hash(
            dataset_manifest
        ),
        git_commit=(
            dataset_manifest.get(
                "git_commit"
            )
        ),
        feature_schema_source=(
            feature_schema_source
        ),
        prediction_probe_payload=(
            prediction_probe
        ),
    )

    p_logger.info(
        "Serving release published: "
        "release_id=%s model_version=%s",
        manifest.release_id,
        manifest.model_version,
    )

    return manifest.to_dict()
