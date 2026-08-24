from __future__ import annotations

import math
import time

from dataclasses import dataclass
from numbers import Real
from typing import Any

import requests


class ServingVerificationError(
    RuntimeError
):
    """
    Raised when a serving release cannot be verified.
    """


@dataclass(frozen=True)
class ServingVerificationResult:
    release_id: str
    model_version: str
    model_run_id: str
    attempts: int
    readiness_payload: dict[str, Any]


@dataclass(frozen=True)
class PredictionProbeResult:
    release_id: str
    model_version: str
    model_run_id: str
    probabilities: tuple[float, ...]
    attempts: int


def verify_serving_release(
    *,
    api_base_url: str,
    expected_release_id: str,
    attempts: int = 20,
    delay_seconds: float = 2.0,
    timeout_seconds: float = 10.0,
) -> ServingVerificationResult:
    """
    Verify that the API serves the expected complete churn release.
    """
    readiness_url = (
        f"{api_base_url.rstrip('/')}/readyz"
    )
    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            response = requests.get(
                readiness_url,
                timeout=timeout_seconds,
            )
            response.raise_for_status()

            payload = response.json()

            if payload.get("status") != "ready":
                raise ValueError(
                    "API did not report ready status."
                )

            if (
                payload.get(
                    "serving_bundle_loaded"
                )
                is not True
            ):
                raise ValueError(
                    "No complete serving bundle is active."
                )

            actual_release_id = payload.get(
                "release_id"
            )

            if (
                actual_release_id
                != expected_release_id
            ):
                raise ValueError(
                    "Unexpected active release: "
                    f"expected={expected_release_id}, "
                    f"actual={actual_release_id}."
                )

            required_components = {
                "feature_schema_loaded": (
                    payload.get(
                        "feature_schema_loaded"
                    )
                ),
                "decision_threshold_loaded": (
                    payload.get(
                        "decision_threshold_loaded"
                    )
                ),
            }

            incomplete_components = [
                name
                for name, loaded
                in required_components.items()
                if loaded is not True
            ]

            if incomplete_components:
                raise ValueError(
                    "Serving release has incomplete "
                    "components: "
                    f"{incomplete_components}."
                )

            model_version = payload.get(
                "model_version"
            )
            model_run_id = payload.get(
                "model_run_id"
            )

            if not model_version:
                raise ValueError(
                    "Ready API has no model version."
                )

            if not model_run_id:
                raise ValueError(
                    "Ready API has no model run ID."
                )

            return ServingVerificationResult(
                release_id=str(
                    actual_release_id
                ),
                model_version=str(
                    model_version
                ),
                model_run_id=str(
                    model_run_id
                ),
                attempts=attempt,
                readiness_payload=payload,
            )

        except (
            requests.RequestException,
            TypeError,
            ValueError,
        ) as error:
            last_error = error

            if attempt < attempts:
                time.sleep(
                    delay_seconds
                )

    raise ServingVerificationError(
        "Serving release verification failed "
        f"after {attempts} attempts | "
        f"expected_release_id="
        f"{expected_release_id} | "
        f"last_error={last_error}"
    ) from last_error


def verify_prediction_probe(
    *,
    api_base_url: str,
    api_key: str,
    prediction_probe_payload: dict,
    expected_release_id: str,
    expected_model_version: str,
    expected_model_run_id: str,
    attempts: int = 3,
    delay_seconds: float = 1.0,
    timeout_seconds: float = 30.0,
) -> PredictionProbeResult:
    """
    Execute one authenticated semantic churn prediction probe.
    """
    if attempts < 1:
        raise ValueError(
            "Prediction probe attempts must "
            "be at least 1."
        )

    inputs = prediction_probe_payload.get(
        "inputs"
    )

    if (
        not isinstance(inputs, list)
        or not inputs
    ):
        raise ServingVerificationError(
            "Prediction probe payload has "
            "no usable inputs."
        )

    predict_url = (
        f"{api_base_url.rstrip('/')}"
        "/predict"
    )

    last_transport_error: Exception | None = (
        None
    )

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            response = requests.post(
                predict_url,
                json=prediction_probe_payload,
                headers={
                    "X-API-KEY": api_key,
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()

        except requests.RequestException as error:
            last_transport_error = error

            if attempt < attempts:
                time.sleep(
                    delay_seconds
                )
                continue

            raise ServingVerificationError(
                "Prediction probe request failed "
                f"after {attempts} attempts | "
                f"last_error={error}"
            ) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise ServingVerificationError(
                "Prediction probe returned "
                "invalid JSON."
            ) from error

        if payload.get("status") != "success":
            raise ServingVerificationError(
                "Prediction probe response status "
                "is not success."
            )

        predictions = payload.get(
            "predictions"
        )

        if (
            not isinstance(predictions, list)
            or len(predictions) != len(inputs)
        ):
            raise ServingVerificationError(
                "Prediction probe returned an "
                "unexpected prediction count."
            )

        probabilities: list[float] = []

        for prediction in predictions:
            if not isinstance(
                prediction,
                dict,
            ):
                raise ServingVerificationError(
                    "Prediction probe returned an "
                    "invalid churn result."
                )

            probability = prediction.get(
                "churn_probability"
            )

            if (
                isinstance(probability, bool)
                or not isinstance(
                    probability,
                    Real,
                )
            ):
                raise ServingVerificationError(
                    "Prediction probe returned a "
                    "non-numeric churn probability."
                )

            numeric_probability = float(
                probability
            )

            if (
                not math.isfinite(
                    numeric_probability
                )
                or not 0.0
                <= numeric_probability
                <= 1.0
            ):
                raise ServingVerificationError(
                    "Prediction probe returned an "
                    "invalid churn probability."
                )

            probabilities.append(
                numeric_probability
            )

        metadata = payload.get(
            "metadata"
        )

        if not isinstance(metadata, dict):
            raise ServingVerificationError(
                "Prediction probe response has "
                "no metadata."
            )

        lineage_expectations = {
            "release_id": (
                expected_release_id
            ),
            "model_version": str(
                expected_model_version
            ),
            "model_run_id": (
                expected_model_run_id
            ),
        }

        for field_name, expected_value in (
            lineage_expectations.items()
        ):
            actual_value = metadata.get(
                field_name
            )

            if str(actual_value) != str(
                expected_value
            ):
                raise ServingVerificationError(
                    "Prediction probe lineage "
                    "mismatch | "
                    f"field={field_name} | "
                    f"expected={expected_value} | "
                    f"actual={actual_value}"
                )

        return PredictionProbeResult(
            release_id=(
                expected_release_id
            ),
            model_version=str(
                expected_model_version
            ),
            model_run_id=(
                expected_model_run_id
            ),
            probabilities=tuple(
                probabilities
            ),
            attempts=attempt,
        )

    raise ServingVerificationError(
        "Prediction probe failed without "
        f"a result: {last_transport_error}"
    )