import pytest
from unittest.mock import patch

from src.inference.serving_bundle import ServingBundle


@pytest.fixture(autouse=True)
def mock_api_dependencies(monkeypatch, mock_xgb_model):
    """
    Mock API dependencies so integration tests do not depend on MLflow,
    startup model loading, GCS, or the full feature pipeline.
    """
    monkeypatch.setenv("API_KEY", "test-secret-key")
    monkeypatch.setenv("APP_ENV", "dev")

    mocked_pipeline_output = {
        "environment": "dev",
        "results": [
            {
                "prediction_id": "test-prediction-id",
                "customer_id": "1234-ABCDE",
                "churn_probability": 0.82,
                "churn_prediction": 1,
                "action": "offer_discount",
                "expected_value": 12.3,
                "customer_value": 100.0,
            }
        ],
        "request_id": "test-request",
        "timings": {"total_ms": 1.0},
        "dq_summary": {"quality_status": "ok", "row_count": 1},
    }

    mocked_bundle = ServingBundle(
        model=mock_xgb_model,
        model_name=(
            "customer-churn-model-dev"
        ),
        model_type="xgboost",
        decision_threshold=0.5,
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
        model_uri=(
            "models:/"
            "customer-churn-model-dev"
            "@champion"
        ),
        model_version="test-version",
        model_run_id="test-run-id",
    )

    with (
        patch(
            "src.api.app.active_serving_bundle",
            mocked_bundle,
        ),
        patch(
            "src.api.app.run_prediction_pipeline",
            return_value=mocked_pipeline_output,
        ),
        patch(
            "src.api.app.log_prediction",
        )
    ):
        yield


def test_api_health_endpoint(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "online"
    assert "model_name" in body
    assert body["serving_alias"] == "champion"
    assert body["model_version"] == "test-version"


def test_predict_endpoint_validation_error(api_client, api_headers):
    bad_payload = {"inputs": []}
    response = api_client.post("/predict", json=bad_payload, headers=api_headers)

    assert response.status_code == 422


def test_predict_endpoint_success(api_client, api_headers, sample_prediction_payload):
    response = api_client.post("/predict", json=sample_prediction_payload, headers=api_headers)

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 1

    prediction = body["predictions"][0]
    assert prediction["churn_probability"] == 0.82
    assert prediction["action"] == "offer_discount"
    assert prediction["expected_value"] == 12.3

    assert "metadata" in body
    assert body["metadata"]["rows"] == 1
    assert body["metadata"]["request_id"] == "test-request"


def test_predict_endpoint_requires_api_key(api_client, sample_prediction_payload):
    response = api_client.post("/predict", json=sample_prediction_payload)

    assert response.status_code == 403


def test_metrics_endpoint_exposes_custom_metrics(api_client):
    response = api_client.get("/metrics")

    assert response.status_code == 200
    assert "api_request_count_total" in response.text
    assert "api_request_latency_seconds" in response.text

def test_readyz_endpoint(api_client):
    response = api_client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["serving_alias"] == "champion"

def test_livez_endpoint(api_client):
    response = api_client.get("/livez")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "alive"
    assert "service" in body
    assert "environment" in body


def test_failed_bundle_reload_keeps_previous_serving_state(
    mock_xgb_model,
):
    from src.api import app as api_app

    previous_bundle = ServingBundle(
        model=mock_xgb_model,
        model_name="customer-churn-model-dev",
        model_type="xgboost",
        decision_threshold=0.5,
        feature_schema={
            "columns": ["tenure"],
            "dtypes": {
                "tenure": "float64",
            },
        },
        serving_alias="champion",
        model_uri=(
            "models:/"
            "customer-churn-model-dev"
            "@champion"
        ),
        model_version="4",
        model_run_id="run-4",
    )

    api_app.active_serving_bundle = (
        previous_bundle
    )

    with patch(
        "src.api.app.reload_model_state",
        side_effect=RuntimeError(
            "Replacement bundle is invalid."
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="Replacement bundle is invalid",
        ):
            api_app.reload_serving_model()

    assert (
        api_app.active_serving_bundle
        is previous_bundle
    )