from unittest.mock import patch

import pytest

import src.api.serving_state as serving_state
from src.inference.serving_bundle import (
    ServingArtifactReference,
    ServingBundle,
    ServingReleaseManifest,
)


def build_test_manifest(
    *,
    release_id: str,
    model_version: str,
    model_run_id: str,
    decision_threshold: float = 0.5,
) -> ServingReleaseManifest:
    """Build an immutable serving manifest for API tests."""
    model_name = (
        "customer-churn-model-dev"
    )

    return ServingReleaseManifest(
        schema_version=1,
        release_id=release_id,
        created_at_utc=(
            "2026-08-24T12:00:00+00:00"
        ),
        model_name=model_name,
        model_version=model_version,
        model_run_id=model_run_id,
        model_uri=(
            f"models:/{model_name}/"
            f"{model_version}"
        ),
        model_type="xgboost",
        decision_threshold=(
            decision_threshold
        ),
        dataset_version="test-dataset",
        config_hash="test-config-hash",
        git_commit="test-commit",
        feature_schema=(
            ServingArtifactReference(
                path="feature_schema.json",
                sha256="test-schema-hash",
            )
        ),
        prediction_probe=None,
    )


def build_test_bundle(
    *,
    model,
    release_id: str = "test-release",
    model_version: str = "test-version",
    model_run_id: str = "test-run-id",
    decision_threshold: float = 0.5,
) -> ServingBundle:
    """Build a complete in-memory serving bundle for API tests."""
    manifest = build_test_manifest(
        release_id=release_id,
        model_version=model_version,
        model_run_id=model_run_id,
        decision_threshold=(
            decision_threshold
        ),
    )

    return ServingBundle(
        release_id=manifest.release_id,
        manifest=manifest,
        model=model,
        model_name=manifest.model_name,
        model_type=manifest.model_type,
        decision_threshold=(
            manifest.decision_threshold
        ),
        feature_schema={
            "columns": [
                "tenure",
                "monthlycharges",
            ],
            "dtypes": {
                "tenure": "float64",
                "monthlycharges": "float64",
            },
        },
        serving_alias="champion",
        model_uri=manifest.model_uri,
        model_version=(
            manifest.model_version
        ),
        model_run_id=(
            manifest.model_run_id
        ),
    )


@pytest.fixture(autouse=True)
def mock_api_dependencies(
    monkeypatch,
    mock_xgb_model,
):
    """
    Isolate API tests from MLflow, GCS and the complete feature pipeline.
    """
    monkeypatch.setenv(
        "API_KEY",
        "test-secret-key",
    )
    monkeypatch.setenv(
        "APP_ENV",
        "dev",
    )

    mocked_pipeline_output = {
        "environment": "dev",
        "results": [
            {
                "prediction_id": (
                    "test-prediction-id"
                ),
                "customer_id": (
                    "1234-ABCDE"
                ),
                "churn_probability": 0.82,
                "churn_prediction": 1,
                "action": "offer_discount",
                "expected_value": 12.3,
                "customer_value": 100.0,
            }
        ],
        "request_id": "test-request",
        "timings": {
            "total_ms": 1.0,
        },
        "dq_summary": {
            "quality_status": "ok",
            "row_count": 1,
        },
    }

    mocked_bundle = build_test_bundle(
        model=mock_xgb_model,
    )

    # API tests intentionally install an already validated bundle directly.
    # Bundle validation itself is covered by dedicated unit tests.
    monkeypatch.setattr(
        serving_state,
        "_active_serving_bundle",
        mocked_bundle,
    )
    monkeypatch.setattr(
        serving_state,
        "_dq_reference_categories",
        {},
    )

    with (
        patch(
            "src.api.routers.prediction."
            "run_prediction_pipeline",
            return_value=(
                mocked_pipeline_output
            ),
        ),
        patch(
            "src.api.routers.prediction."
            "log_prediction",
        ),
    ):
        yield

    serving_state.clear_serving_bundle()


def test_api_health_endpoint(
    api_client,
):
    response = api_client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "online"
    assert "model_name" in body
    assert body["serving_alias"] == (
        "champion"
    )
    assert body["model_version"] == (
        "test-version"
    )


def test_predict_endpoint_validation_error(
    api_client,
    api_headers,
):
    response = api_client.post(
        "/predict",
        json={
            "inputs": [],
        },
        headers=api_headers,
    )

    assert response.status_code == 422


def test_predict_endpoint_success(
    api_client,
    api_headers,
    sample_prediction_payload,
):
    response = api_client.post(
        "/predict",
        json=sample_prediction_payload,
        headers=api_headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"
    assert isinstance(
        body["predictions"],
        list,
    )
    assert len(
        body["predictions"]
    ) == 1

    prediction = (
        body["predictions"][0]
    )

    assert (
        prediction["churn_probability"]
        == 0.82
    )
    assert prediction["action"] == (
        "offer_discount"
    )
    assert (
        prediction["expected_value"]
        == 12.3
    )
    assert prediction["customer_id"] == (
        "1234-ABCDE"
    )

    metadata = body["metadata"]

    assert metadata["rows"] == 1
    assert metadata["request_id"] == (
        "test-request"
    )
    assert metadata["release_id"] == (
        "test-release"
    )
    assert metadata["model_version"] == (
        "test-version"
    )
    assert metadata["model_run_id"] == (
        "test-run-id"
    )


def test_predict_endpoint_requires_api_key(
    api_client,
    sample_prediction_payload,
):
    response = api_client.post(
        "/predict",
        json=sample_prediction_payload,
    )

    assert response.status_code == 403


def test_metrics_endpoint_exposes_custom_metrics(
    api_client,
):
    response = api_client.get(
        "/metrics"
    )

    assert response.status_code == 200
    assert (
        "api_request_count_total"
        in response.text
    )
    assert (
        "api_request_latency_seconds"
        in response.text
    )


def test_readyz_endpoint(
    api_client,
):
    response = api_client.get(
        "/readyz"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert body["serving_alias"] == (
        "champion"
    )
    assert body["model_version"] == (
        "test-version"
    )
    assert body["model_run_id"] == (
        "test-run-id"
    )
    assert body["model_uri"] == (
        "models:/"
        "customer-churn-model-dev/"
        "test-version"
    )
    assert (
        body["decision_threshold"]
        == 0.5
    )
    assert (
        body["serving_bundle_loaded"]
        is True
    )
    assert body["release_id"] == (
        "test-release"
    )
    assert (
        body["feature_schema_loaded"]
        is True
    )
    assert (
        body["decision_threshold_loaded"]
        is True
    )


def test_livez_endpoint(
    api_client,
):
    response = api_client.get(
        "/livez"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "alive"
    assert "service" in body
    assert "environment" in body


def test_failed_bundle_reload_keeps_previous_serving_state(
    monkeypatch,
    mock_xgb_model,
):
    previous_bundle = build_test_bundle(
        model=mock_xgb_model,
        release_id="previous-release",
        model_version="4",
        model_run_id="run-4",
        decision_threshold=0.5,
    )

    monkeypatch.setattr(
        serving_state,
        "_active_serving_bundle",
        previous_bundle,
    )

    with patch(
        "src.api.serving_state."
        "load_current_serving_bundle",
        side_effect=RuntimeError(
            "Replacement bundle is invalid."
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match=(
                "Replacement bundle is invalid"
            ),
        ):
            serving_state.reload_serving_model()

    assert (
        serving_state.get_active_serving_bundle()
        is previous_bundle
    )


def test_rollback_serving_release(
    api_client,
    api_headers,
    monkeypatch,
    mock_xgb_model,
):
    previous_bundle = build_test_bundle(
        model=mock_xgb_model,
        release_id="release-new",
        model_version="8",
        model_run_id="run-8",
    )

    rollback_bundle = build_test_bundle(
        model=mock_xgb_model,
        release_id="release-old",
        model_version="7",
        model_run_id="run-7",
    )

    monkeypatch.setattr(
        serving_state,
        "_active_serving_bundle",
        previous_bundle,
    )

    with (
        patch(
            "src.api.routers.admin."
            "load_active_release_id",
            return_value="release-new",
        ),
        patch(
            "src.api.routers.admin."
            "load_serving_bundle_for_release",
            return_value=rollback_bundle,
        ) as load_bundle,
        patch(
            "src.api.routers.admin."
            "activate_release_pointer",
        ) as activate_pointer,
    ):
        response = api_client.post(
            (
                "/admin/"
                "rollback-serving-release"
            ),
            json={
                "release_id": (
                    "release-old"
                ),
            },
            headers=api_headers,
        )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == (
        "rolled_back"
    )
    assert body["release_id"] == (
        "release-old"
    )
    assert body["previous_release_id"] == (
        "release-new"
    )
    assert body["model_version"] == "7"

    assert (
        serving_state.get_active_serving_bundle()
        is rollback_bundle
    )

    load_bundle.assert_called_once_with(
        release_id="release-old",
        model_name=(
            serving_state.MODEL_NAME
        ),
        cfg=serving_state.CFG,
        models_path=(
            serving_state.MODELS_PATH
        ),
    )

    activate_pointer.assert_called_once_with(
        models_path=(
            serving_state.MODELS_PATH
        ),
        release_id="release-old",
        operation="rollback",
        previous_release_id=(
            "release-new"
        ),
    )


def test_failed_rollback_keeps_previous_bundle(
    api_client,
    api_headers,
    monkeypatch,
    mock_xgb_model,
):
    previous_bundle = build_test_bundle(
        model=mock_xgb_model,
        release_id="release-new",
        model_version="8",
        model_run_id="run-8",
    )

    monkeypatch.setattr(
        serving_state,
        "_active_serving_bundle",
        previous_bundle,
    )

    with (
        patch(
            "src.api.routers.admin."
            "load_active_release_id",
            return_value="release-new",
        ),
        patch(
            "src.api.routers.admin."
            "load_serving_bundle_for_release",
            side_effect=ValueError(
                "Rollback release is invalid."
            ),
        ),
        patch(
            "src.api.routers.admin."
            "activate_release_pointer",
        ) as activate_pointer,
    ):
        response = api_client.post(
            (
                "/admin/"
                "rollback-serving-release"
            ),
            json={
                "release_id": (
                    "release-old"
                ),
            },
            headers=api_headers,
        )

    assert response.status_code == 500

    assert (
        serving_state.get_active_serving_bundle()
        is previous_bundle
    )

    activate_pointer.assert_not_called()