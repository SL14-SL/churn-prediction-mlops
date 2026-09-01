from __future__ import annotations
from typing import Any

from datetime import datetime, timezone

from src.storage.filesystem import (
    file_exists,
    list_files,
)
from src.configs.paths import join_uri

from src.inference.serving_bundle import (
    ServingReleaseManifest,
)

from src.inference.releases.storage import (
    write_json, 
    load_json, 
    build_release_paths,
)

from src.inference.releases.manifest import (
    parse_serving_manifest, 
    resolve_release_artifact_uri
)
MANIFEST_FILE_NAME = "serving_manifest.json"
ACTIVE_RELEASE_FILE_NAME = (
    "active_serving_release.json"
)
RELEASES_DIRECTORY_NAME = "serving_releases"


def load_serving_manifest(
    *,
    models_path: str,
    release_id: str,
) -> tuple[
    ServingReleaseManifest,
    str,
]:
    """Load and validate the manifest of one immutable serving release."""
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if not file_exists(paths["manifest"]):
        raise FileNotFoundError(
            "Serving release manifest not found: "
            f"{paths['manifest']}"
        )

    manifest = parse_serving_manifest(
        load_json(paths["manifest"])
    )

    if manifest.release_id != release_id:
        raise ValueError(
            "Requested release ID does not "
            "match the manifest."
        )

    return (
        manifest,
        paths["release_root"],
    )

def load_release_prediction_probe(
    *,
    models_path: str,
    release_id: str,
) -> dict[str, Any] | None:
    """
    Load and checksum-validate the prediction probe of one release.

    Releases without a prediction probe return None.
    """
    manifest, release_root = (
        load_serving_manifest(
            models_path=models_path,
            release_id=release_id,
        )
    )

    if manifest.prediction_probe is None:
        return None

    probe_uri = (
        resolve_release_artifact_uri(
            release_root=release_root,
            reference=(
                manifest.prediction_probe
            ),
        )
    )

    probe_payload = load_json(
        probe_uri
    )

    inputs = probe_payload.get(
        "inputs"
    )

    if (
        not isinstance(inputs, list)
        or not inputs
    ):
        raise ValueError(
            "Release prediction probe has "
            "no usable inputs."
        )

    return probe_payload


def load_active_serving_manifest(
    *,
    models_path: str,
) -> tuple[
    ServingReleaseManifest,
    str,
]:
    """Load the manifest referenced by the active-release pointer."""
    release_id = load_active_release_id(
        models_path=models_path,
    )

    return load_serving_manifest(
        models_path=models_path,
        release_id=release_id,
    )

def load_serving_release_manifest(
    *,
    models_path: str,
    release_id: str,
) -> ServingReleaseManifest:
    """
    Load one immutable serving release manifest by release ID.
    """
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    manifest_path = paths["manifest"]

    if not file_exists(manifest_path):
        raise FileNotFoundError(
            "Serving release manifest not found: "
            f"{manifest_path}"
        )

    return parse_serving_manifest(
        load_json(manifest_path)
    )


def load_active_release_id(
    *,
    models_path: str,
) -> str:
    """
    Return the active serving-release identifier.

    Raises:
        FileNotFoundError: If no active-release pointer exists.
        ValueError: If the pointer contains no release identifier.
    """
    pointer_path = join_uri(
        models_path,
        ACTIVE_RELEASE_FILE_NAME,
    )

    if not file_exists(pointer_path):
        raise FileNotFoundError(
            "Active serving release pointer "
            f"not found: {pointer_path}"
        )

    pointer = load_json(
        pointer_path
    )

    release_id = pointer.get(
        "release_id"
    )

    if not release_id:
        raise ValueError(
            "Active serving release pointer "
            "has no release_id."
        )

    return str(release_id)


def activate_release_pointer(
    *,
    models_path: str,
    release_id: str,
    operation: str = "activation",
    previous_release_id: str | None = None,
) -> None:
    """
    Update the active release pointer.

    The caller must validate the target release before calling this function.
    """
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    if not file_exists(paths["manifest"]):
        raise FileNotFoundError(
            "Cannot activate release without manifest: "
            f"{paths['manifest']}"
        )

    write_json(
        paths["active_pointer"],
        {
            "schema_version": 1,
            "release_id": release_id,
            "previous_release_id": (
                previous_release_id
            ),
            "operation": operation,
            "updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    )

    # Confirm that storage returns the newly written pointer.
    stored_release_id = load_active_release_id(
        models_path=models_path,
    )

    if stored_release_id != release_id:
        raise RuntimeError(
            "Active release pointer verification failed: "
            f"expected={release_id}, "
            f"actual={stored_release_id}"
        )


def list_serving_release_manifests(
    *,
    models_path: str,
) -> list[ServingReleaseManifest]:
    """Load all serving-release manifests ordered from newest to oldest."""
    pattern = join_uri(
        models_path,
        RELEASES_DIRECTORY_NAME,
        "*",
        MANIFEST_FILE_NAME,
    )

    manifest_paths = list_files(
        pattern
    )

    manifests: list[
        ServingReleaseManifest
    ] = []

    for manifest_path in manifest_paths:
        manifests.append(
            parse_serving_manifest(
                load_json(manifest_path)
            )
        )

    return sorted(
        manifests,
        key=lambda manifest: (
            manifest.created_at_utc
        ),
        reverse=True,
    )

