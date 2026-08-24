from typing import Any

from src.configs.paths import join_uri
from src.storage.filesystem import (
    file_exists,
)
from src.inference.serving_bundle import (
    ServingArtifactReference,
    ServingReleaseManifest,
    validate_serving_manifest,
)
from src.inference.releases.storage import (
    sha256_uri,
)


def _parse_artifact_reference(
    payload: dict[str, Any],
    field_name: str,
) -> ServingArtifactReference:
    reference = payload.get(
        field_name
    )

    if not isinstance(reference, dict):
        raise ValueError(
            "Invalid serving artifact reference: "
            f"{field_name}"
        )

    path = reference.get("path")
    checksum = reference.get("sha256")

    if not path or not checksum:
        raise ValueError(
            "Incomplete serving artifact reference: "
            f"{field_name}"
        )

    return ServingArtifactReference(
        path=str(path),
        sha256=str(checksum),
    )


def _parse_optional_artifact_reference(
    payload: dict[str, Any],
    field_name: str,
) -> ServingArtifactReference | None:
    reference = payload.get(
        field_name
    )

    if reference is None:
        return None

    return _parse_artifact_reference(
        payload,
        field_name,
    )


def parse_serving_manifest(
    payload: dict[str, Any],
) -> ServingReleaseManifest:
    """
    Parse and validate a persisted churn serving manifest.
    """
    schema_version = int(
        payload["schema_version"]
    )

    if schema_version not in {1}:
        raise ValueError(
            "Unsupported serving manifest "
            f"schema version: {schema_version}"
        )

    manifest = ServingReleaseManifest(
        schema_version=schema_version,
        release_id=str(
            payload["release_id"]
        ),
        created_at_utc=str(
            payload["created_at_utc"]
        ),
        model_name=str(
            payload["model_name"]
        ),
        model_version=str(
            payload["model_version"]
        ),
        model_run_id=str(
            payload["model_run_id"]
        ),
        model_uri=str(
            payload["model_uri"]
        ),
        model_type=str(
            payload["model_type"]
        ),
        decision_threshold=float(
            payload["decision_threshold"]
        ),
        dataset_version=payload.get(
            "dataset_version"
        ),
        config_hash=payload.get(
            "config_hash"
        ),
        git_commit=payload.get(
            "git_commit"
        ),
        feature_schema=(
            _parse_artifact_reference(
                payload,
                "feature_schema",
            )
        ),
        prediction_probe=(
            _parse_optional_artifact_reference(
                payload,
                "prediction_probe",
            )
        ),
    )

    validate_serving_manifest(
        manifest
    )

    return manifest


def resolve_release_artifact_uri(
    *,
    release_root: str,
    reference: ServingArtifactReference,
) -> str:
    """
    Resolve and checksum-validate an artifact inside one release.
    """
    relative_path = reference.path

    if (
        relative_path.startswith("/")
        or relative_path.startswith(
            "gs://"
        )
        or ".." in relative_path.split("/")
    ):
        raise ValueError(
            "Serving artifact path must be "
            f"relative and contained: {relative_path}"
        )

    artifact_uri = join_uri(
        release_root,
        relative_path,
    )

    if not file_exists(artifact_uri):
        raise FileNotFoundError(
            "Serving artifact not found: "
            f"{artifact_uri}"
        )

    actual_checksum = sha256_uri(
        artifact_uri
    )

    if actual_checksum != reference.sha256:
        raise ValueError(
            "Serving artifact checksum mismatch: "
            f"{relative_path}"
        )

    return artifact_uri