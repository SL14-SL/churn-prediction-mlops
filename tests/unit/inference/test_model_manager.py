from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from src.inference.model_manager import (
    load_serving_bundle_for_release,
    reload_serving_model,
)
from src.inference.serving_bundle import (
    ServingArtifactReference,
    ServingBundle,
    ServingReleaseManifest,
)


def build_manifest(
    *,
    model_name: str = (
        "customer-churn-model-dev"
    ),
) -> ServingReleaseManifest:
    return ServingReleaseManifest(
        schema_version=1,
        release_id="release-7",
        created_at_utc=(
            "2026-08-24T12:00:00+00:00"
        ),
        model_name=model_name,
        model_version="7",
        model_run_id="run-7",
        model_uri=(
            f"models:/{model_name}/7"
        ),
        model_type="xgboost",
        decision_threshold=0.42,
        dataset_version="dataset-1",
        config_hash="config-hash",
        git_commit="abc123",
        feature_schema=(
            ServingArtifactReference(
                path="feature_schema.json",
                sha256="schema-hash",
            )
        ),
        prediction_probe=None,
    )


def test_load_serving_bundle_for_release():
    manifest = build_manifest()
    model = MagicMock()

    feature_schema = {
        "columns": [
            "tenure",
            "monthlycharges",
        ],
        "dtypes": {
            "tenure": "float64",
            "monthlycharges": "float64",
        },
    }

    with (
        patch(
            "src.inference.model_manager."
            "mlflow.set_tracking_uri",
        ),
        patch(
            "src.inference.model_manager."
            "load_serving_manifest",
            return_value=(
                manifest,
                (
                    "models/serving_releases/"
                    "release-7"
                ),
            ),
        ),
        patch(
            "src.inference.model_manager."
            "resolve_release_artifact_uri",
            return_value=(
                "models/serving_releases/"
                "release-7/"
                "feature_schema.json"
            ),
        ) as resolve_artifact,
        patch(
            "src.inference.model_manager."
            "load_json",
            return_value=feature_schema,
        ),
        patch(
            "src.inference.model_manager."
            "load_model_by_type",
            return_value=model,
        ) as load_model,
    ):
        bundle = (
            load_serving_bundle_for_release(
                release_id="release-7",
                model_name=(
                    "customer-churn-model-dev"
                ),
                cfg={
                    "tracking": {
                        "mlflow_tracking_uri": (
                            "http://mlflow:5000"
                        ),
                    },
                },
                models_path="models",
            )
        )

    assert isinstance(
        bundle,
        ServingBundle,
    )
    assert bundle.release_id == (
        "release-7"
    )
    assert bundle.manifest is manifest
    assert bundle.model is model
    assert bundle.model_version == "7"
    assert bundle.model_run_id == "run-7"
    assert bundle.decision_threshold == 0.42
    assert bundle.feature_schema == (
        feature_schema
    )

    resolve_artifact.assert_called_once_with(
        release_root=(
            "models/serving_releases/"
            "release-7"
        ),
        reference=(
            manifest.feature_schema
        ),
    )

    load_model.assert_called_once_with(
        (
            "models:/"
            "customer-churn-model-dev/7"
        ),
        "xgboost",
    )


def test_load_release_rejects_wrong_model_name():
    manifest = build_manifest(
        model_name="wrong-model"
    )

    with (
        patch(
            "src.inference.model_manager."
            "mlflow.set_tracking_uri",
        ),
        patch(
            "src.inference.model_manager."
            "load_serving_manifest",
            return_value=(
                manifest,
                "models/release-7",
            ),
        ),
    ):
        with pytest.raises(
            ValueError,
            match=(
                "does not match configuration"
            ),
        ):
            load_serving_bundle_for_release(
                release_id="release-7",
                model_name=(
                    "customer-churn-model-dev"
                ),
                cfg={},
                models_path="models",
            )


def test_reload_uses_active_release():
    manifest = build_manifest()
    expected_bundle = MagicMock(
        spec=ServingBundle
    )

    cfg = {
        "paths": {
            "models": "models",
        },
        "tracking": {
            "mlflow_tracking_uri": (
                "http://mlflow:5000"
            ),
        },
    }

    with (
        patch(
            "src.inference.model_manager."
            "load_active_serving_manifest",
            return_value=(
                manifest,
                (
                    "models/serving_releases/"
                    "release-7"
                ),
            ),
        ) as load_active,
        patch(
            "src.inference.model_manager."
            "load_serving_bundle_for_release",
            return_value=expected_bundle,
        ) as load_bundle,
    ):
        result = reload_serving_model(
            model_name=(
                "customer-churn-model-dev"
            ),
            cfg=cfg,
        )

    assert result is expected_bundle

    load_active.assert_called_once_with(
        models_path="models",
    )

    load_bundle.assert_called_once_with(
        release_id="release-7",
        model_name=(
            "customer-churn-model-dev"
        ),
        cfg=cfg,
        models_path="models",
    )


def test_reload_rejects_missing_models_path():
    with pytest.raises(
        ValueError,
        match="no models path",
    ):
        reload_serving_model(
            model_name=(
                "customer-churn-model-dev"
            ),
            cfg={
                "paths": {},
            },
        )