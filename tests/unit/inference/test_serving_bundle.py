import pytest

from src.inference.serving_bundle import (
    ServingBundle,
    ServingArtifactReference,
    ServingReleaseManifest,
    validate_serving_bundle,
    validate_serving_manifest,
)


def build_valid_bundle(
    **overrides,
) -> ServingBundle:
    values = {
        "model": object(),
        "model_name": "customer-churn-model-dev",
        "model_type": "xgboost",
        "decision_threshold": 0.42,
        "feature_schema": {
            "columns": [
                "tenure",
                "monthlycharges",
            ],
            "dtypes": {
                "tenure": "float64",
                "monthlycharges": "float64",
            },
        },
        "serving_alias": "champion",
        "model_uri": (
            "models:/customer-churn-model-dev@champion"
        ),
        "model_version": "3",
        "model_run_id": "run-3",
    }

    values.update(overrides)

    return ServingBundle(**values)


def test_valid_serving_bundle_passes_validation():
    bundle = build_valid_bundle()

    validate_serving_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "model",
            None,
            "Serving bundle has no model",
        ),
        (
            "model_name",
            "",
            "Serving bundle has no model name",
        ),
        (
            "model_version",
            "",
            "Serving bundle has no model version",
        ),
        (
            "model_run_id",
            "",
            "Serving bundle has no model run ID",
        ),
    ],
)
def test_incomplete_serving_bundle_fails_validation(
    field,
    value,
    message,
):
    bundle = build_valid_bundle(
        **{field: value}
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        validate_serving_bundle(bundle)


@pytest.mark.parametrize(
    "threshold",
    [-0.01, 1.01],
)
def test_invalid_decision_threshold_fails_validation(
    threshold,
):
    bundle = build_valid_bundle(
        decision_threshold=threshold,
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_serving_bundle(bundle)


def test_empty_feature_columns_fail_validation():
    bundle = build_valid_bundle(
        feature_schema={
            "columns": [],
            "dtypes": {},
        },
    )

    with pytest.raises(
        ValueError,
        match="feature schema has no columns",
    ):
        validate_serving_bundle(bundle)


def test_duplicate_feature_columns_fail_validation():
    bundle = build_valid_bundle(
        feature_schema={
            "columns": [
                "tenure",
                "tenure",
            ],
            "dtypes": {
                "tenure": "float64",
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="duplicate columns",
    ):
        validate_serving_bundle(bundle)


def test_dtype_for_unknown_feature_fails_validation():
    bundle = build_valid_bundle(
        feature_schema={
            "columns": [
                "tenure",
            ],
            "dtypes": {
                "tenure": "float64",
                "unknown_feature": "float64",
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="dtypes for unknown columns",
    ):
        validate_serving_bundle(bundle)

def build_valid_manifest(
    **overrides,
) -> ServingReleaseManifest:
    values = {
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
        "feature_schema": (
            ServingArtifactReference(
                path="feature_schema.json",
                sha256="feature-schema-hash",
            )
        ),
        "prediction_probe": (
            ServingArtifactReference(
                path="prediction_probe.json",
                sha256="prediction-probe-hash",
            )
        ),
    }

    values.update(overrides)

    return ServingReleaseManifest(
        **values
    )


def test_valid_serving_manifest_passes_validation():
    manifest = build_valid_manifest()

    validate_serving_manifest(manifest)


def test_serving_manifest_serializes_to_dict():
    manifest = build_valid_manifest()

    payload = manifest.to_dict()

    assert payload["release_id"] == (
        "churn-release-1"
    )
    assert payload["model_version"] == "7"
    assert payload["decision_threshold"] == 0.42
    assert payload["feature_schema"] == {
        "path": "feature_schema.json",
        "sha256": "feature-schema-hash",
    }
    assert payload["prediction_probe"] == {
        "path": "prediction_probe.json",
        "sha256": "prediction-probe-hash",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "schema_version",
            0,
            "invalid schema version",
        ),
        (
            "release_id",
            "",
            "no release ID",
        ),
        (
            "model_version",
            "",
            "no model version",
        ),
        (
            "model_run_id",
            "",
            "no model run ID",
        ),
    ],
)
def test_invalid_serving_manifest_fails_validation(
    field,
    value,
    message,
):
    manifest = build_valid_manifest(
        **{field: value}
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        validate_serving_manifest(
            manifest
        )


def test_manifest_rejects_invalid_threshold():
    manifest = build_valid_manifest(
        decision_threshold=1.1,
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        validate_serving_manifest(
            manifest
        )


def test_manifest_rejects_missing_feature_schema_path():
    manifest = build_valid_manifest(
        feature_schema=(
            ServingArtifactReference(
                path="",
                sha256="feature-schema-hash",
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="feature schema has no path",
    ):
        validate_serving_manifest(
            manifest
        )