from unittest.mock import (
    MagicMock,
)

import pytest

from flows.tasks import serving_tasks
from src.deployment.verification import (
    PredictionProbeResult,
    ServingVerificationResult,
)
from src.inference.serving_bundle import (
    ServingArtifactReference,
    ServingReleaseManifest,
)


def build_manifest() -> (
    ServingReleaseManifest
):
    return ServingReleaseManifest(
        schema_version=1,
        release_id="release-2",
        created_at_utc=(
            "2026-08-24T12:00:00+00:00"
        ),
        model_name=(
            "customer-churn-model-dev"
        ),
        model_version="2",
        model_run_id="run-2",
        model_uri=(
            "models:/"
            "customer-churn-model-dev/2"
        ),
        model_type="xgboost",
        decision_threshold=0.42,
        dataset_version="dataset-2",
        config_hash="config-hash",
        git_commit="abc123",
        feature_schema=(
            ServingArtifactReference(
                path="feature_schema.json",
                sha256="schema-hash",
            )
        ),
        prediction_probe=(
            ServingArtifactReference(
                path="prediction_probe.json",
                sha256="probe-hash",
            )
        ),
    )


def test_resolve_previous_release(
    monkeypatch,
):
    monkeypatch.setattr(
        serving_tasks,
        "get_path",
        MagicMock(
            return_value="models"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "load_active_release_id",
        MagicMock(
            return_value="release-1"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_run_logger",
        MagicMock(
            return_value=MagicMock()
        ),
    )

    result = (
        serving_tasks
        .task_resolve_previous_release
        .fn()
    )

    assert result == "release-1"


def test_resolve_previous_release_during_bootstrap(
    monkeypatch,
):
    monkeypatch.setattr(
        serving_tasks,
        "get_path",
        MagicMock(
            return_value="models"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "load_active_release_id",
        MagicMock(
            side_effect=FileNotFoundError
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_run_logger",
        MagicMock(
            return_value=MagicMock()
        ),
    )

    result = (
        serving_tasks
        .task_resolve_previous_release
        .fn()
    )

    assert result is None


def test_refresh_api(
    monkeypatch,
):
    response = MagicMock()
    response.json.return_value = {
        "release_id": "release-2",
        "model_version": "2",
    }

    post = MagicMock(
        return_value=response
    )

    monkeypatch.setattr(
        serving_tasks,
        "resolve_api_base_url",
        MagicMock(
            return_value="http://api:8080"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "require_api_key",
        MagicMock(
            return_value="secret"
        ),
    )
    monkeypatch.setattr(
        serving_tasks.requests,
        "post",
        post,
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_run_logger",
        MagicMock(
            return_value=MagicMock()
        ),
    )

    result = (
        serving_tasks
        .task_refresh_api
        .fn()
    )

    assert result["release_id"] == (
        "release-2"
    )

    post.assert_called_once_with(
        (
            "http://api:8080"
            "/admin/reload-model"
        ),
        headers={
            "X-API-KEY": "secret",
        },
        timeout=300,
    )


def test_verify_serving_release_task(
    monkeypatch,
):
    manifest = build_manifest()

    readiness = (
        ServingVerificationResult(
            release_id="release-2",
            model_version="2",
            model_run_id="run-2",
            attempts=1,
            readiness_payload={},
        )
    )

    probe = PredictionProbeResult(
        release_id="release-2",
        model_version="2",
        model_run_id="run-2",
        probabilities=(0.82,),
        attempts=1,
    )

    monkeypatch.setattr(
        serving_tasks,
        "resolve_api_base_url",
        MagicMock(
            return_value="http://api:8080"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_path",
        MagicMock(
            return_value="models"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "verify_serving_release",
        MagicMock(
            return_value=readiness
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "load_serving_release_manifest",
        MagicMock(
            return_value=manifest
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "load_release_prediction_probe",
        MagicMock(
            return_value={
                "inputs": [
                    {
                        "customerID": (
                            "1000-AAAAA"
                        ),
                    }
                ]
            }
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "verify_prediction_probe",
        MagicMock(
            return_value=probe
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "require_api_key",
        MagicMock(
            return_value="secret"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_run_logger",
        MagicMock(
            return_value=MagicMock()
        ),
    )

    result = (
        serving_tasks
        .task_verify_serving_release
        .fn(
            expected_release_id=(
                "release-2"
            ),
        )
    )

    assert result[
        "prediction_probe_status"
    ] == "verified"

    assert result[
        "probe_probabilities"
    ] == [0.82]


def test_verify_rejects_manifest_mismatch(
    monkeypatch,
):
    manifest = build_manifest()

    readiness = (
        ServingVerificationResult(
            release_id="release-2",
            model_version="99",
            model_run_id="run-2",
            attempts=1,
            readiness_payload={},
        )
    )

    monkeypatch.setattr(
        serving_tasks,
        "resolve_api_base_url",
        MagicMock(
            return_value="http://api:8080"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_path",
        MagicMock(
            return_value="models"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "verify_serving_release",
        MagicMock(
            return_value=readiness
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "load_serving_release_manifest",
        MagicMock(
            return_value=manifest
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_run_logger",
        MagicMock(
            return_value=MagicMock()
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="model version",
    ):
        (
            serving_tasks
            .task_verify_serving_release
            .fn(
                expected_release_id=(
                    "release-2"
                ),
            )
        )


def test_rollback_serving_release_task(
    monkeypatch,
):
    manifest = build_manifest()

    response = MagicMock()
    response.json.return_value = {
        "status": "rolled_back",
        "release_id": "release-2",
    }

    post = MagicMock(
        return_value=response
    )
    client = MagicMock()

    monkeypatch.setattr(
        serving_tasks,
        "resolve_api_base_url",
        MagicMock(
            return_value="http://api:8080"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "require_api_key",
        MagicMock(
            return_value="secret"
        ),
    )
    monkeypatch.setattr(
        serving_tasks.requests,
        "post",
        post,
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_path",
        MagicMock(
            return_value="models"
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "load_serving_release_manifest",
        MagicMock(
            return_value=manifest
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "MlflowClient",
        MagicMock(
            return_value=client
        ),
    )
    monkeypatch.setattr(
        serving_tasks,
        "get_run_logger",
        MagicMock(
            return_value=MagicMock()
        ),
    )

    result = (
        serving_tasks
        .task_rollback_serving_release
        .fn(
            previous_release_id=(
                "release-2"
            ),
        )
    )

    assert result["release_id"] == (
        "release-2"
    )

    client.set_registered_model_alias\
        .assert_called_once_with(
            name=(
                "customer-churn-model-dev"
            ),
            alias="champion",
            version="2",
        )