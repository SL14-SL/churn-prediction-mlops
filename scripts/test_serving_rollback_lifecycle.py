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
    list_serving_release_manifests,
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


def rollback_release(
    *,
    api_base_url: str,
    api_key: str,
    release_id: str,
) -> dict[str, Any]:
    """
    Activate an existing serving release through the rollback endpoint.
    """
    response = requests.post(
        f"{api_base_url}/admin/rollback-serving-release",
        json={
            "release_id": release_id,
        },
        headers={
            "X-API-KEY": api_key,
        },
        timeout=(10, 300),
    )
    response.raise_for_status()

    result = response.json()

    if result.get("release_id") != release_id:
        raise RuntimeError(
            "Rollback endpoint activated an unexpected release | "
            f"expected={release_id} | "
            f"actual={result.get('release_id')}"
        )

    if result.get("status") not in {
        "rolled_back",
        "unchanged",
    }:
        raise RuntimeError(
            "Unexpected rollback response status: "
            f"{result.get('status')}"
        )

    return result


def select_rollback_release(
    *,
    models_path: str,
    active_release_id: str,
) -> str:
    """
    Select the newest non-active release containing a prediction probe.
    """
    manifests = list_serving_release_manifests(
        models_path=models_path,
    )

    for manifest in manifests:
        if manifest.release_id == active_release_id:
            continue

        prediction_probe = load_release_prediction_probe(
            models_path=models_path,
            release_id=manifest.release_id,
        )

        if prediction_probe is not None:
            return manifest.release_id

    raise RuntimeError(
        "Rollback E2E test requires at least two valid serving "
        "releases with prediction probes."
    )


def verify_release(
    *,
    api_base_url: str,
    api_key: str,
    models_path: str,
    release_id: str,
) -> dict[str, Any]:
    """
    Verify readiness, lineage, and semantic prediction for one release.
    """
    manifest = load_serving_release_manifest(
        models_path=models_path,
        release_id=release_id,
    )
    prediction_probe = load_release_prediction_probe(
        models_path=models_path,
        release_id=release_id,
    )

    if prediction_probe is None:
        raise RuntimeError(
            f"Serving release '{release_id}' has no prediction probe."
        )

    readiness_result = verify_serving_release(
        api_base_url=api_base_url,
        expected_release_id=release_id,
        attempts=10,
        delay_seconds=1.0,
        timeout_seconds=30.0,
    )

    if str(readiness_result.model_version) != str(
        manifest.model_version
    ):
        raise RuntimeError(
            "Ready endpoint model version does not match "
            f"release manifest '{release_id}'."
        )

    if readiness_result.model_run_id != manifest.model_run_id:
        raise RuntimeError(
            "Ready endpoint model run ID does not match "
            f"release manifest '{release_id}'."
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

    return {
        "readiness": asdict(readiness_result),
        "prediction_probe": asdict(prediction_result),
    }


def main() -> int:
    """
    Verify rollback to a previous release and restore the original release.
    """
    api_base_url = require_environment_variable(
        "API_BASE_URL"
    ).rstrip("/")
    api_key = require_environment_variable("API_KEY")
    models_path = get_path("models")

    original_release_id = load_active_release_id(
        models_path=models_path,
    )
    rollback_release_id = select_rollback_release(
        models_path=models_path,
        active_release_id=original_release_id,
    )

    print(
        "Testing serving rollback | "
        f"original_release_id={original_release_id} | "
        f"rollback_release_id={rollback_release_id}"
    )

    rollback_result: dict[str, Any] | None = None
    rollback_verification: dict[str, Any] | None = None
    restoration_result: dict[str, Any] | None = None
    restoration_verification: dict[str, Any] | None = None

    try:
        rollback_result = rollback_release(
            api_base_url=api_base_url,
            api_key=api_key,
            release_id=rollback_release_id,
        )

        rollback_verification = verify_release(
            api_base_url=api_base_url,
            api_key=api_key,
            models_path=models_path,
            release_id=rollback_release_id,
        )

        print(
            "✅ Rollback release verified | "
            f"release_id={rollback_release_id}"
        )

    finally:
        print(
            "Restoring original serving release | "
            f"release_id={original_release_id}"
        )

        restoration_result = rollback_release(
            api_base_url=api_base_url,
            api_key=api_key,
            release_id=original_release_id,
        )

        restoration_verification = verify_release(
            api_base_url=api_base_url,
            api_key=api_key,
            models_path=models_path,
            release_id=original_release_id,
        )

    restored_release_id = load_active_release_id(
        models_path=models_path,
    )

    if restored_release_id != original_release_id:
        raise RuntimeError(
            "Original serving release was not restored | "
            f"expected={original_release_id} | "
            f"actual={restored_release_id}"
        )

    result = {
        "status": "verified",
        "original_release_id": original_release_id,
        "rollback_release_id": rollback_release_id,
        "rollback": rollback_result,
        "rollback_verification": rollback_verification,
        "restoration": restoration_result,
        "restoration_verification": restoration_verification,
    }

    print()
    print("✅ Serving rollback lifecycle verified")
    print(f"   Tested release: {rollback_release_id}")
    print(f"   Restored release: {original_release_id}")
    print()
    print(
        "SERVING_ROLLBACK_E2E_RESULT="
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