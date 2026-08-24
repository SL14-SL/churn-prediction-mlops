from unittest.mock import (
    MagicMock,
)

import pytest
import requests

from src.deployment import verification


def response_with(
    payload: dict,
) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = (
        None
    )
    return response


def ready_payload(
    *,
    release_id: str = "release-2",
) -> dict:
    return {
        "status": "ready",
        "serving_bundle_loaded": True,
        "release_id": release_id,
        "model_version": "2",
        "model_run_id": "run-2",
        "feature_schema_loaded": True,
        "decision_threshold_loaded": True,
    }


def prediction_payload(
    *,
    release_id: str = "release-2",
    probability=0.82,
) -> dict:
    return {
        "status": "success",
        "predictions": [
            {
                "churn_probability": (
                    probability
                ),
                "action": "offer_discount",
                "expected_value": 12.3,
            }
        ],
        "metadata": {
            "release_id": release_id,
            "model_version": "2",
            "model_run_id": "run-2",
        },
    }


def test_verifies_expected_release(
    monkeypatch,
):
    monkeypatch.setattr(
        verification.requests,
        "get",
        MagicMock(
            return_value=response_with(
                ready_payload()
            )
        ),
    )

    result = (
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-2"
            ),
            attempts=1,
        )
    )

    assert result.release_id == "release-2"
    assert result.model_version == "2"
    assert result.model_run_id == "run-2"


def test_rejects_unexpected_release(
    monkeypatch,
):
    monkeypatch.setattr(
        verification.requests,
        "get",
        MagicMock(
            return_value=response_with(
                ready_payload(
                    release_id="release-1"
                )
            )
        ),
    )

    with pytest.raises(
        verification.ServingVerificationError,
        match="Unexpected active release",
    ):
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-2"
            ),
            attempts=1,
        )


def test_rejects_incomplete_bundle(
    monkeypatch,
):
    payload = ready_payload()
    payload[
        "feature_schema_loaded"
    ] = False

    monkeypatch.setattr(
        verification.requests,
        "get",
        MagicMock(
            return_value=response_with(
                payload
            )
        ),
    )

    with pytest.raises(
        verification.ServingVerificationError,
        match="incomplete components",
    ):
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-2"
            ),
            attempts=1,
        )


def test_retries_transient_readiness_failure(
    monkeypatch,
):
    mock_get = MagicMock(
        side_effect=[
            requests.ConnectionError(
                "API restarting"
            ),
            response_with(
                ready_payload()
            ),
        ]
    )

    mock_sleep = MagicMock()

    monkeypatch.setattr(
        verification.requests,
        "get",
        mock_get,
    )
    monkeypatch.setattr(
        verification.time,
        "sleep",
        mock_sleep,
    )

    result = (
        verification.verify_serving_release(
            api_base_url=(
                "http://localhost:8000"
            ),
            expected_release_id=(
                "release-2"
            ),
            attempts=2,
            delay_seconds=0.01,
        )
    )

    assert result.attempts == 2

    mock_sleep.assert_called_once_with(
        0.01
    )


def test_prediction_probe_succeeds(
    monkeypatch,
):
    post = MagicMock(
        return_value=response_with(
            prediction_payload()
        )
    )

    monkeypatch.setattr(
        verification.requests,
        "post",
        post,
    )

    result = (
        verification.verify_prediction_probe(
            api_base_url=(
                "http://api:8080"
            ),
            api_key="secret",
            prediction_probe_payload={
                "inputs": [
                    {
                        "customerID": (
                            "1000-AAAAA"
                        ),
                    }
                ]
            },
            expected_release_id=(
                "release-2"
            ),
            expected_model_version="2",
            expected_model_run_id="run-2",
        )
    )

    assert result.probabilities == (
        0.82,
    )


def test_prediction_probe_rejects_wrong_release(
    monkeypatch,
):
    monkeypatch.setattr(
        verification.requests,
        "post",
        MagicMock(
            return_value=response_with(
                prediction_payload(
                    release_id=(
                        "wrong-release"
                    )
                )
            )
        ),
    )

    with pytest.raises(
        verification.ServingVerificationError,
        match="lineage mismatch",
    ):
        verification.verify_prediction_probe(
            api_base_url=(
                "http://api:8080"
            ),
            api_key="secret",
            prediction_probe_payload={
                "inputs": [
                    {
                        "customerID": (
                            "1000-AAAAA"
                        ),
                    }
                ]
            },
            expected_release_id=(
                "release-2"
            ),
            expected_model_version="2",
            expected_model_run_id="run-2",
        )


@pytest.mark.parametrize(
    "probability",
    [
        float("nan"),
        float("inf"),
        -0.1,
        1.1,
        "not-a-number",
    ],
)
def test_prediction_probe_rejects_invalid_probability(
    monkeypatch,
    probability,
):
    monkeypatch.setattr(
        verification.requests,
        "post",
        MagicMock(
            return_value=response_with(
                prediction_payload(
                    probability=probability
                )
            )
        ),
    )

    with pytest.raises(
        verification.ServingVerificationError,
    ):
        verification.verify_prediction_probe(
            api_base_url=(
                "http://api:8080"
            ),
            api_key="secret",
            prediction_probe_payload={
                "inputs": [
                    {
                        "customerID": (
                            "1000-AAAAA"
                        ),
                    }
                ]
            },
            expected_release_id=(
                "release-2"
            ),
            expected_model_version="2",
            expected_model_run_id="run-2",
        )