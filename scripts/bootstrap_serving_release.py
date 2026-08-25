from __future__ import annotations

import os

import mlflow
from mlflow import MlflowClient

from flows.deployment_flow import (
    deploy_and_verify_release,
)
from src.configs.loader import (
    get_path,
    load_config,
)
from src.deployment.prediction_probe import (
    build_prediction_probe,
)
from src.inference.releases.publisher import (
    publish_serving_release,
)
from src.inference.releases.repository import (
    load_active_release_id,
)


def bootstrap_serving_release() -> dict:
    """
    Publish and deploy a serving release from the existing MLflow Champion.

    If an active release already exists, the operation is idempotently
    skipped.
    """
    cfg = load_config()
    model_name = cfg[
        "model"
    ]["registry_name"]
    models_path = get_path(
        "models"
    )

    try:
        existing_release_id = (
            load_active_release_id(
                models_path=models_path,
            )
        )

        return {
            "status": "unchanged",
            "release_id": (
                existing_release_id
            ),
        }

    except FileNotFoundError:
        pass

    tracking_uri = cfg.get(
        "tracking",
        {},
    ).get(
        "mlflow_tracking_uri"
    )

    if tracking_uri:
        mlflow.set_tracking_uri(
            str(tracking_uri)
        )

    client = MlflowClient()

    champion_version = (
        client.get_model_version_by_alias(
            model_name,
            "champion",
        )
    )

    model_version = str(
        champion_version.version
    )
    model_run_id = (
        champion_version.run_id
    )

    run = client.get_run(
        model_run_id
    )

    model_type = (
        run.data.tags.get(
            "model_type"
        )
        or run.data.params.get(
            "model_type"
        )
        or "xgboost"
    )

    decision_threshold = float(
        run.data.params.get(
            "decision_threshold",
            0.5,
        )
    )

    feature_schema_source = (
        client.download_artifacts(
            run_id=model_run_id,
            path=(
                "feature_schema/"
                "feature_schema.json"
            ),
        )
    )

    validated_data_path = (
        f"{get_path('validated_data')}/"
        "train.parquet"
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
        model_name=model_name,
        model_version=model_version,
        model_run_id=model_run_id,
        model_type=model_type,
        decision_threshold=(
            decision_threshold
        ),
        dataset_version=None,
        config_hash=None,
        git_commit=os.getenv(
            "GIT_COMMIT_SHA"
        ),
        feature_schema_source=(
            feature_schema_source
        ),
        prediction_probe_payload=(
            prediction_probe
        ),
    )

    deployment = (
        deploy_and_verify_release(
            release_id=(
                manifest.release_id
            ),
            previous_release_id=None,
        )
    )

    return {
        "status": "bootstrapped",
        "release_id": (
            manifest.release_id
        ),
        "model_name": model_name,
        "model_version": (
            model_version
        ),
        "model_run_id": model_run_id,
        "deployment": deployment,
    }


if __name__ == "__main__":
    result = (
        bootstrap_serving_release()
    )

    print(
        "BOOTSTRAP_RESULT="
        f"{result}"
    )   