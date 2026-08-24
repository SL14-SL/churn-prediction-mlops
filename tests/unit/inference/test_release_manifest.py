import hashlib
import json

import pytest

from src.inference.releases.manifest import (
    parse_serving_manifest,
    resolve_release_artifact_uri,
)
from src.inference.serving_bundle import (
    ServingArtifactReference,
)


def build_manifest_payload(
    **overrides,
) -> dict:
    payload = {
        "schema_version": 1,
        "release_id": "churn-release-1",
        "created_at_utc": (
            "2026-08-24T12:00:00+00:00"
        ),
        "model_name": (
            "customer-churn-model-dev"
        ),
        "model_version": "7",
        "model_run_id": "run-7",
        "model_uri": (
            "models:/"
            "customer-churn-model-dev/7"
        ),
        "model_type": "xgboost",
        "decision_threshold": 0.42,
        "dataset_version": "dataset-1",
        "config_hash": "config-hash",
        "git_commit": "abc123",
        "feature_schema": {
            "path": "feature_schema.json",
            "sha256": "schema-hash",
        },
        "prediction_probe": {
            "path": "prediction_probe.json",
            "sha256": "probe-hash",
        },
    }

    payload.update(overrides)

    return payload


def test_parse_serving_manifest():
    manifest = parse_serving_manifest(
        build_manifest_payload()
    )

    assert manifest.schema_version == 1
    assert manifest.release_id == (
        "churn-release-1"
    )
    assert manifest.model_version == "7"
    assert manifest.decision_threshold == 0.42
    assert manifest.feature_schema.path == (
        "feature_schema.json"
    )
    assert manifest.prediction_probe is not None
    assert manifest.prediction_probe.path == (
        "prediction_probe.json"
    )


def test_parse_manifest_without_optional_probe():
    payload = build_manifest_payload(
        prediction_probe=None,
    )

    manifest = parse_serving_manifest(
        payload
    )

    assert manifest.prediction_probe is None


def test_parse_rejects_unsupported_schema_version():
    payload = build_manifest_payload(
        schema_version=99,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported serving manifest",
    ):
        parse_serving_manifest(payload)


def test_parse_rejects_missing_feature_schema():
    payload = build_manifest_payload()
    payload.pop("feature_schema")

    with pytest.raises(
        ValueError,
        match=(
            "Invalid serving artifact reference: "
            "feature_schema"
        ),
    ):
        parse_serving_manifest(payload)


def test_resolve_release_artifact_uri(
    tmp_path,
):
    artifact = (
        tmp_path
        / "feature_schema.json"
    )
    content = json.dumps(
        {
            "columns": [
                "tenure",
                "monthlycharges",
            ]
        }
    )

    artifact.write_text(
        content,
        encoding="utf-8",
    )

    checksum = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    reference = ServingArtifactReference(
        path="feature_schema.json",
        sha256=checksum,
    )

    resolved = (
        resolve_release_artifact_uri(
            release_root=str(tmp_path),
            reference=reference,
        )
    )

    assert resolved == str(artifact)


def test_modified_artifact_fails_checksum_validation(
    tmp_path,
):
    artifact = (
        tmp_path
        / "feature_schema.json"
    )
    artifact.write_text(
        '{"columns": ["tenure"]}',
        encoding="utf-8",
    )

    original_checksum = hashlib.sha256(
        b'{"columns": ["tenure"]}'
    ).hexdigest()

    artifact.write_text(
        '{"tampered": true}',
        encoding="utf-8",
    )

    reference = ServingArtifactReference(
        path="feature_schema.json",
        sha256=original_checksum,
    )

    with pytest.raises(
        ValueError,
        match="checksum mismatch",
    ):
        resolve_release_artifact_uri(
            release_root=str(tmp_path),
            reference=reference,
        )


@pytest.mark.parametrize(
    "path",
    [
        "../../secret.txt",
        "/tmp/secret.txt",
        "gs://other-bucket/secret.txt",
    ],
)
def test_artifact_cannot_escape_release_prefix(
    tmp_path,
    path,
):
    reference = ServingArtifactReference(
        path=path,
        sha256="irrelevant",
    )

    with pytest.raises(
        ValueError,
        match="relative and contained",
    ):
        resolve_release_artifact_uri(
            release_root=str(tmp_path),
            reference=reference,
        )