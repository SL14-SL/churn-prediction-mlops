import json

import pytest

from src.inference.releases.repository import (
    activate_release_pointer,
    list_serving_release_manifests,
    load_active_release_id,
    load_active_serving_manifest,
    load_release_prediction_probe,
    load_serving_manifest,
)
from src.inference.releases.storage import (
    build_release_paths,
    sha256_uri,
    write_json,
)


def create_test_release(
    *,
    models_path: str,
    release_id: str,
    model_version: str,
    created_at_utc: str,
    include_probe: bool = True,
) -> dict[str, str]:
    paths = build_release_paths(
        models_path=models_path,
        release_id=release_id,
    )

    feature_schema_path = (
        f"{paths['release_root']}/"
        "feature_schema.json"
    )

    write_json(
        feature_schema_path,
        {
            "columns": [
                "tenure",
                "monthlycharges",
            ],
            "dtypes": {
                "tenure": "float64",
                "monthlycharges": "float64",
            },
        },
    )

    prediction_probe = None

    if include_probe:
        probe_path = (
            f"{paths['release_root']}/"
            "prediction_probe.json"
        )

        write_json(
            probe_path,
            {
                "inputs": [
                    {
                        "tenure": 12,
                        "MonthlyCharges": 70.35,
                    }
                ],
                "context": {
                    "purpose": (
                        "post_deployment_verification"
                    ),
                },
            },
        )

        prediction_probe = {
            "path": "prediction_probe.json",
            "sha256": sha256_uri(
                probe_path
            ),
        }

    manifest_payload = {
        "schema_version": 1,
        "release_id": release_id,
        "created_at_utc": created_at_utc,
        "model_name": (
            "customer-churn-model-dev"
        ),
        "model_version": model_version,
        "model_run_id": (
            f"run-{model_version}"
        ),
        "model_uri": (
            "models:/"
            "customer-churn-model-dev/"
            f"{model_version}"
        ),
        "model_type": "xgboost",
        "decision_threshold": 0.42,
        "dataset_version": "dataset-1",
        "config_hash": "config-hash",
        "git_commit": "abc123",
        "feature_schema": {
            "path": "feature_schema.json",
            "sha256": sha256_uri(
                feature_schema_path
            ),
        },
        "prediction_probe": prediction_probe,
    }

    write_json(
        paths["manifest"],
        manifest_payload,
    )

    return paths


def test_load_serving_manifest(
    tmp_path,
):
    models_path = str(
        tmp_path / "models"
    )

    create_test_release(
        models_path=models_path,
        release_id="release-1",
        model_version="7",
        created_at_utc=(
            "2026-08-24T12:00:00+00:00"
        ),
    )

    manifest, release_root = (
        load_serving_manifest(
            models_path=models_path,
            release_id="release-1",
        )
    )

    assert manifest.release_id == "release-1"
    assert manifest.model_version == "7"
    assert release_root.endswith(
        "serving_releases/release-1"
    )


def test_activate_and_load_active_release(
    tmp_path,
):
    models_path = str(
        tmp_path / "models"
    )

    create_test_release(
        models_path=models_path,
        release_id="release-1",
        model_version="7",
        created_at_utc=(
            "2026-08-24T12:00:00+00:00"
        ),
    )

    activate_release_pointer(
        models_path=models_path,
        release_id="release-1",
    )

    assert load_active_release_id(
        models_path=models_path,
    ) == "release-1"

    manifest, _ = (
        load_active_serving_manifest(
            models_path=models_path,
        )
    )

    assert manifest.release_id == "release-1"


def test_activate_release_rejects_missing_manifest(
    tmp_path,
):
    models_path = str(
        tmp_path / "models"
    )

    with pytest.raises(
        FileNotFoundError,
        match="without manifest",
    ):
        activate_release_pointer(
            models_path=models_path,
            release_id="missing-release",
        )


def test_activation_records_previous_release(
    tmp_path,
):
    models_path = str(
        tmp_path / "models"
    )

    paths = create_test_release(
        models_path=models_path,
        release_id="release-2",
        model_version="8",
        created_at_utc=(
            "2026-08-24T13:00:00+00:00"
        ),
    )

    activate_release_pointer(
        models_path=models_path,
        release_id="release-2",
        operation="rollback",
        previous_release_id="release-3",
    )

    with open(
        paths["active_pointer"],
        encoding="utf-8",
    ) as file_handle:
        pointer = json.load(
            file_handle
        )

    assert pointer["release_id"] == (
        "release-2"
    )
    assert pointer["previous_release_id"] == (
        "release-3"
    )
    assert pointer["operation"] == "rollback"


def test_load_prediction_probe(
    tmp_path,
):
    models_path = str(
        tmp_path / "models"
    )

    create_test_release(
        models_path=models_path,
        release_id="release-1",
        model_version="7",
        created_at_utc=(
            "2026-08-24T12:00:00+00:00"
        ),
    )

    probe = load_release_prediction_probe(
        models_path=models_path,
        release_id="release-1",
    )

    assert probe is not None
    assert probe["inputs"] == [
        {
            "tenure": 12,
            "MonthlyCharges": 70.35,
        }
    ]


def test_release_without_probe_returns_none(
    tmp_path,
):
    models_path = str(
        tmp_path / "models"
    )

    create_test_release(
        models_path=models_path,
        release_id="release-1",
        model_version="7",
        created_at_utc=(
            "2026-08-24T12:00:00+00:00"
        ),
        include_probe=False,
    )

    assert load_release_prediction_probe(
        models_path=models_path,
        release_id="release-1",
    ) is None


def test_list_releases_newest_first(
    tmp_path,
):
    models_path = str(
        tmp_path / "models"
    )

    create_test_release(
        models_path=models_path,
        release_id="release-1",
        model_version="7",
        created_at_utc=(
            "2026-08-24T12:00:00+00:00"
        ),
    )

    create_test_release(
        models_path=models_path,
        release_id="release-2",
        model_version="8",
        created_at_utc=(
            "2026-08-24T13:00:00+00:00"
        ),
    )

    manifests = (
        list_serving_release_manifests(
            models_path=models_path,
        )
    )

    assert [
        manifest.release_id
        for manifest in manifests
    ] == [
        "release-2",
        "release-1",
    ]