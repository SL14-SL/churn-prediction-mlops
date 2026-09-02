from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

import requests

from src.configs.loader import get_path
from src.deployment.verification import (
    verify_prediction_probe,
    verify_serving_release,
)
from src.inference.releases.repository import (
    load_active_release_id,
    load_release_prediction_probe,
    load_serving_release_manifest,
)


def require_environment_variable(name: str) -> str:
    """
    Return a required environment variable.

    Raises:
        RuntimeError: If the variable is missing or empty.
    """
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set."
        )

    return value


def reload_serving_model(
    *,
    api_base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Ask the API to reload the release referenced by the active pointer.
    """
    response = requests.post(
        f"{api_base_url}/admin/reload-model",
        headers={
            "X-API-KEY": api_key,
        },
        timeout=(10, 300),
    )
    response.raise_for_status()

    result = response.json()

    if result.get("status") != "reloaded":
        raise RuntimeError(
            "Unexpected reload response status: "
            f"{result.get('status')}"
        )

    return result


def main() -> int:
    """
    Verify reload, readiness, lineage, and prediction behavior end to end.
    """
    api_base_url = require_environment_variable(
        "API_BASE_URL"
    ).rstrip("/")
    api_key = require_environment_variable("API_KEY")
    models_path = get_path("models")

    expected_release_id = load_active_release_id(
        models_path=models_path,
    )
    manifest = load_serving_release_manifest(
        models_path=models_path,
        release_id=expected_release_id,
    )
    prediction_probe = load_release_prediction_probe(
        models_path=models_path,
        release_id=expected_release_id,
    )

    if prediction_probe is None:
        raise RuntimeError(
            "The active serving release has no prediction probe."
        )

    print(
        "Testing active serving release | "
        f"release_id={expected_release_id}"
    )

    reload_result = reload_serving_model(
        api_base_url=api_base_url,
        api_key=api_key,
    )

    if reload_result.get("release_id") != expected_release_id:
        raise RuntimeError(
            "Reloaded release does not match the active pointer | "
            f"expected={expected_release_id} | "
            f"actual={reload_result.get('release_id')}"
        )

    readiness_result = verify_serving_release(
        api_base_url=api_base_url,
        expected_release_id=expected_release_id,
        attempts=10,
        delay_seconds=1.0,
        timeout_seconds=30.0,
    )

    if str(readiness_result.model_version) != str(
        manifest.model_version
    ):
        raise RuntimeError(
            "Ready endpoint model version does not match "
            "the release manifest."
        )

    if readiness_result.model_run_id != manifest.model_run_id:
        raise RuntimeError(
            "Ready endpoint model run ID does not match "
            "the release manifest."
        )

    prediction_result = verify_prediction_probe(
        api_base_url=api_base_url,
        api_key=api_key,
        prediction_probe_payload=prediction_probe,
        expected_release_id=manifest.release_id,
        expected_model_version=manifest.model_version,
        expected_model_run_id=manifest.model_run_id,
        attempts=3,
        delay_seconds=1.0,
        timeout_seconds=30.0,
    )

    result = {
        "status": "verified",
        "reload": reload_result,
        "readiness": asdict(readiness_result),
        "prediction_probe": asdict(prediction_result),
    }

    print()
    print("✅ Active serving release verified end to end")
    print(f"   Release: {readiness_result.release_id}")
    print(f"   Model version: {readiness_result.model_version}")
    print(f"   Model run: {readiness_result.model_run_id}")
    print(
        "   Churn probabilities: "
        f"{list(prediction_result.probabilities)}"
    )
    print()
    print(
        "SERVING_E2E_RESULT="
        + json.dumps(
            result,
            indent=2,
            default=str,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())