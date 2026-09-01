from __future__ import annotations

import uuid

from datetime import (
    datetime,
    timezone,
)
from typing import Any

from src.storage.filesystem import (
    file_exists,
    remove_file,
)
from src.inference.serving_bundle import (
    ServingArtifactReference,
    ServingReleaseManifest,
    validate_serving_manifest,
)
from src.inference.releases.repository import (
    activate_release_pointer,
    load_active_release_id,
)
from src.inference.releases.storage import (
    build_release_paths,
    copy_uri,
    load_json,
    sha256_uri,
    write_json,
)


def build_release_id(
    model_version: str,
) -> str:
    """Build a unique and sortable identifier for an immutable serving release."""
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    unique_suffix = uuid.uuid4().hex[
        :8
    ]

    return (
        f"release-{timestamp}"
        f"-v{model_version}"
        f"-{unique_suffix}"
    )


def publish_serving_release(
    *,
    models_path: str,
    model_name: str,
    model_version: str,
    model_run_id: str,
    model_type: str,
    decision_threshold: float,
    dataset_version: str | None,
    config_hash: str | None,
    git_commit: str | None,
    feature_schema_source: str,
    prediction_probe_payload: dict[
        str,
        Any,
    ],
) -> ServingReleaseManifest:
    """
    Publish a complete immutable churn serving release.

    Publication protocol:
    1. Copy the feature schema into an immutable release prefix.
    2. Write the deterministic prediction probe.
    3. Verify both artifacts using SHA-256.
    4. Write and verify the release manifest.
    5. Update the active pointer last.

    If any operation before step 5 fails, the previous active release
    remains selected.
    """
    resolved_model_version = str(
        model_version
    )

    release_id = build_release_id(
        resolved_model_version
    )

    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if file_exists(paths["manifest"]):
        raise FileExistsError(
            "Serving release already exists: "
            f"{release_id}"
        )

    probe_inputs = (
        prediction_probe_payload.get(
            "inputs"
        )
    )

    if (
        not isinstance(probe_inputs, list)
        or not probe_inputs
    ):
        raise ValueError(
            "Prediction probe payload must "
            "contain non-empty inputs."
        )

    try:
        source_hash = sha256_uri(
            feature_schema_source
        )

        copy_uri(
            feature_schema_source,
            paths["feature_schema"],
        )

        feature_schema_hash = sha256_uri(
            paths["feature_schema"]
        )

        if feature_schema_hash != source_hash:
            raise ValueError(
                "Serving artifact checksum mismatch "
                "after copy: feature_schema"
            )

        feature_schema_reference = (
            ServingArtifactReference(
                path="feature_schema.json",
                sha256=feature_schema_hash,
            )
        )

        write_json(
            paths["prediction_probe"],
            prediction_probe_payload,
        )

        prediction_probe_reference = (
            ServingArtifactReference(
                path="prediction_probe.json",
                sha256=sha256_uri(
                    paths["prediction_probe"]
                ),
            )
        )

        manifest = ServingReleaseManifest(
            schema_version=1,
            release_id=release_id,
            created_at_utc=datetime.now(
                timezone.utc
            ).isoformat(),
            model_name=model_name,
            model_version=(
                resolved_model_version
            ),
            model_run_id=model_run_id,
            model_uri=(
                f"models:/{model_name}/"
                f"{resolved_model_version}"
            ),
            model_type=model_type,
            decision_threshold=float(
                decision_threshold
            ),
            dataset_version=dataset_version,
            config_hash=config_hash,
            git_commit=git_commit,
            feature_schema=(
                feature_schema_reference
            ),
            prediction_probe=(
                prediction_probe_reference
            ),
        )

        validate_serving_manifest(
            manifest
        )

        # Write the manifest only after both serving artifacts
        # have been written and verified.
        write_json(
            paths["manifest"],
            manifest.to_dict(),
        )

        stored_manifest = load_json(
            paths["manifest"]
        )

        if (
            stored_manifest.get(
                "release_id"
            )
            != release_id
        ):
            raise ValueError(
                "Stored serving manifest failed "
                "read-after-write validation."
            )

        previous_release_id = None

        if file_exists(
            paths["active_pointer"]
        ):
            previous_release_id = (
                load_active_release_id(
                    models_path=models_path,
                )
            )

        # Commit point: pointer changes only after complete publication.
        activate_release_pointer(
            models_path=models_path,
            release_id=release_id,
            operation="promotion",
            previous_release_id=(
                previous_release_id
            ),
        )

        return manifest

    except Exception:
        # Do not touch the active pointer. Remove only objects belonging
        # to the incomplete new release.
        for key in (
            "manifest",
            "feature_schema",
            "prediction_probe",
        ):
            remove_file(
                paths[key]
            )

        raise