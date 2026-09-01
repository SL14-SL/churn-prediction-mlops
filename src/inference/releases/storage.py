import hashlib
import json
import fsspec

from typing import Any, BinaryIO

from src.storage.filesystem import read_text, write_text, file_exists
from src.configs.paths import join_uri

MANIFEST_FILE_NAME = "serving_manifest.json"
ACTIVE_RELEASE_FILE_NAME = (
    "active_serving_release.json"
)
RELEASES_DIRECTORY_NAME = "serving_releases"
COPY_CHUNK_SIZE = 1024 * 1024

def _copy_stream(
    source: BinaryIO,
    target: BinaryIO,
) -> None:
    while True:
        chunk = source.read(
            COPY_CHUNK_SIZE
        )

        if not chunk:
            break

        target.write(chunk)


def copy_uri(
    source_path: str,
    target_path: str,
) -> None:
    """
    Copy a file between any fsspec-supported locations.

    Supported combinations include:
    - local to local
    - local to GCS
    - GCS to local
    - GCS to GCS
    """
    if not file_exists(source_path):
        raise FileNotFoundError(
            f"Serving release source not found: "
            f"{source_path}"
        )

    with (
        fsspec.open(source_path, "rb") as source,
        fsspec.open(target_path, "wb") as target,
    ):
        _copy_stream(
            source,
            target,
        )


def sha256_uri(path: str) -> str:
    """Calculate SHA-256 for a local or remote artifact."""
    digest = hashlib.sha256()

    with fsspec.open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(
                COPY_CHUNK_SIZE
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: str,
    payload: dict[str, Any],
) -> None:
    """Serialize a mapping as JSON to a local or GCS destination."""
    write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
    )


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON object from a local or GCS source."""
    payload = json.loads(
        read_text(path)
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object at: {path}"
        )

    return payload

def build_release_paths(
    *,
    models_path: str,
    release_id: str,
) -> dict[str, str]:
    """Build every storage location belonging to one serving release."""
    release_root = join_uri(
        models_path,
        RELEASES_DIRECTORY_NAME,
        release_id,
    )

    return {
        "release_root": release_root,
        "manifest": join_uri(
            release_root,
            MANIFEST_FILE_NAME,
        ),
        "feature_schema": join_uri(
            release_root,
            "feature_schema.json",
        ),
        "prediction_probe": join_uri(
            release_root,
            "prediction_probe.json",
        ),
        "active_pointer": join_uri(
            models_path,
            ACTIVE_RELEASE_FILE_NAME,
        ),
    }
