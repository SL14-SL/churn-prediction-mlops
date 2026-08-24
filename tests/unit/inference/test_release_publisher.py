import json

import pytest
from pathlib import Path

from src.inference.releases.publisher import (
    publish_serving_release,
)
from src.inference.releases.repository import (
    load_active_release_id,
    load_active_serving_manifest,
    load_release_prediction_probe,
)
from src.inference.releases.storage import (
    build_release_paths,
)


def create_feature_schema(
    tmp_path,
) -> str:
    feature_schema_path = (
        tmp_path / "feature_schema.json"
    )

    feature_schema_path.write_text(
        json.dumps(
            {
                "columns": [
                    "tenure",
                    "monthlycharges",
                ],
                "dtypes": {
                    "tenure": "float64",
                    "monthlycharges": "float64",
                },
            }
        ),
        encoding="utf-8",
    )

    return str(feature_schema_path)


def build_probe() -> dict:
    return {
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
    }


def publish_test_release(
    *,
    tmp_path,
    model_version: str = "7",
):
    return publish_serving_release(
        models_path=str(
            tmp_path / "models"
        ),
        model_name=(
            "customer-churn-model-dev"
        ),
        model_version=model_version,
        model_run_id=(
            f"run-{model_version}"
        ),
        model_type="xgboost",
        decision_threshold=0.42,
        dataset_version="dataset-1",
        config_hash="config-hash",
        git_commit="abc123",
        feature_schema_source=(
            create_feature_schema(
                tmp_path
            )
        ),
        prediction_probe_payload=(
            build_probe()
        ),
    )


def test_publish_and_activate_release(
    tmp_path,
):
    manifest = publish_test_release(
        tmp_path=tmp_path
    )

    models_path = str(
        tmp_path / "models"
    )

    assert load_active_release_id(
        models_path=models_path,
    ) == manifest.release_id

    loaded, _ = (
        load_active_serving_manifest(
            models_path=models_path,
        )
    )

    assert loaded.release_id == (
        manifest.release_id
    )
    assert loaded.model_version == "7"
    assert loaded.model_uri == (
        "models:/customer-churn-model-dev/7"
    )
    assert loaded.decision_threshold == 0.42


def test_published_artifacts_exist(
    tmp_path,
):
    manifest = publish_test_release(
        tmp_path=tmp_path
    )

    paths = build_release_paths(
        models_path=str(
            tmp_path / "models"
        ),
        release_id=manifest.release_id,
    )

    assert Path(
        paths["feature_schema"]
    ).exists()

    assert Path(
        paths["prediction_probe"]
    ).exists()

    with open(
        paths["feature_schema"],
        encoding="utf-8",
    ) as file_handle:
        schema = json.load(
            file_handle
        )

    assert schema["columns"] == [
        "tenure",
        "monthlycharges",
    ]

    probe = load_release_prediction_probe(
        models_path=str(
            tmp_path / "models"
        ),
        release_id=manifest.release_id,
    )

    assert probe == build_probe()


def test_publish_rejects_empty_probe(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="non-empty inputs",
    ):
        publish_serving_release(
            models_path=str(
                tmp_path / "models"
            ),
            model_name=(
                "customer-churn-model-dev"
            ),
            model_version="7",
            model_run_id="run-7",
            model_type="xgboost",
            decision_threshold=0.42,
            dataset_version="dataset-1",
            config_hash="config-hash",
            git_commit="abc123",
            feature_schema_source=(
                create_feature_schema(
                    tmp_path
                )
            ),
            prediction_probe_payload={
                "inputs": [],
            },
        )


def test_failed_publication_keeps_active_release(
    tmp_path,
):
    first_release = publish_test_release(
        tmp_path=tmp_path,
        model_version="7",
    )

    models_path = str(
        tmp_path / "models"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        publish_serving_release(
            models_path=models_path,
            model_name=(
                "customer-churn-model-dev"
            ),
            model_version="8",
            model_run_id="run-8",
            model_type="xgboost",
            decision_threshold=0.45,
            dataset_version="dataset-2",
            config_hash="new-hash",
            git_commit="def456",
            feature_schema_source=(
                "missing-feature-schema.json"
            ),
            prediction_probe_payload=(
                build_probe()
            ),
        )

    assert load_active_release_id(
        models_path=models_path,
    ) == first_release.release_id