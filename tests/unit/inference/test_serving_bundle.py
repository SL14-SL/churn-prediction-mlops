import pytest

from src.inference.serving_bundle import (
    ServingBundle,
    validate_serving_bundle,
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