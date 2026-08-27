from __future__ import annotations

import json
import time
import os
import sys
from dataclasses import asdict

import requests

from src.configs.loader import get_path
from src.deployment.verification import (
    ServingVerificationError,
    verify_prediction_probe,
    verify_serving_release,
)
from src.inference.releases.repository import (
    load_active_release_id,
    load_release_prediction_probe,
    load_serving_release_manifest,
)


def require_environment_variable(
    name: str,
) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable "
            f"'{name}' is not set."
        )

    return value


def verify_liveness(
    api_base_url: str,
    *,
    attempts: int = 12,
    delay_seconds: float = 5.0,
) -> dict:
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                f"{api_base_url}/livez",
                timeout=(10, 30),
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            last_error = error

            if attempt < attempts:
                time.sleep(delay_seconds)

    raise RuntimeError(
        "Production API did not become live after "
        f"{attempts} attempts."
    ) from last_error

def main() -> int:
    api_base_url = require_environment_variable(
        "PRODUCTION_API_URL"
    ).rstrip("/")

    api_key = require_environment_variable(
        "API_KEY"
    )

    models_path = get_path("models")

    expected_release_id = (
        load_active_release_id(
            models_path=models_path,
        )
    )

    if not expected_release_id:
        raise ServingVerificationError(
            "No active production serving "
            "release could be resolved."
        )

    print(
        "Verifying production deployment | "
        f"api_base_url={api_base_url} | "
        f"expected_release_id="
        f"{expected_release_id}"
    )

    liveness_payload = verify_liveness(
        api_base_url=api_base_url,
    )

    readiness_result = verify_serving_release(
        api_base_url=api_base_url,
        expected_release_id=(
            expected_release_id
        ),
        attempts=30,
        delay_seconds=2.0,
        timeout_seconds=100.0,
    )

    manifest = load_serving_release_manifest(
        models_path=models_path,
        release_id=expected_release_id,
    )

    if (
        str(readiness_result.model_version)
        != str(manifest.model_version)
    ):
        raise ServingVerificationError(
            "Production model version does not "
            "match the active release manifest | "
            f"ready="
            f"{readiness_result.model_version} | "
            f"manifest={manifest.model_version}"
        )

    if (
        readiness_result.model_run_id
        != manifest.model_run_id
    ):
        raise ServingVerificationError(
            "Production model run ID does not "
            "match the active release manifest | "
            f"ready="
            f"{readiness_result.model_run_id} | "
            f"manifest={manifest.model_run_id}"
        )

    prediction_probe_payload = (
        load_release_prediction_probe(
            models_path=models_path,
            release_id=expected_release_id,
        )
    )

    if prediction_probe_payload is None:
        raise ServingVerificationError(
            "The active production release has "
            "no semantic prediction probe."
        )

    prediction_result = (
        verify_prediction_probe(
            api_base_url=api_base_url,
            api_key=api_key,
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
            attempts=3,
            delay_seconds=2.0,
            timeout_seconds=30.0,
        )
    )

    result = {
        "status": "verified",
        "api_base_url": api_base_url,
        "liveness": liveness_payload,
        "readiness": asdict(
            readiness_result
        ),
        "prediction_probe": asdict(
            prediction_result
        ),
    }

    print()
    print("✅ Production deployment verified")
    print(
        "   API: "
        f"{result['api_base_url']}"
    )
    print(
        "   Release: "
        f"{result['readiness']['release_id']}"
    )
    print(
        "   Model version: "
        f"{result['readiness']['model_version']}"
    )
    print(
        "   Model run: "
        f"{result['readiness']['model_run_id']}"
    )
    print(
        "   Prediction probe: passed "
        f"after {result['prediction_probe']['attempts']} "
        "attempt(s)"
    )
    print(
        "   Churn probabilities: "
        f"{result['prediction_probe']['probabilities']}"
    )
    print

    print(
        "PRODUCTION_VERIFICATION_RESULT="
        + json.dumps(
            result,
            indent=2,
            default=str,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(
            "Production deployment verification "
            f"failed: {error}",
            file=sys.stderr,
        )
        raise