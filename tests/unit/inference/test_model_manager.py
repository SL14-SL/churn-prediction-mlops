from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from src.inference.serving_bundle import (
    ServingBundle,
)
from src.inference.model_manager import (
    reload_serving_model,
)


def test_reload_serving_model_returns_valid_bundle():
    model = MagicMock()
    model_version = MagicMock(
        version="7",
        run_id="run-7",
    )

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
            "resolve_tracking_uri",
            return_value="http://mlflow:5000",
        ),
        patch(
            "src.inference.model_manager."
            "mlflow.set_tracking_uri",
        ) as set_tracking_uri,
        patch(
            "src.inference.model_manager."
            "load_registry_model",
            return_value=(
                model,
                "xgboost",
                "champion",
                (
                    "models:/"
                    "customer-churn-model-dev"
                    "@champion"
                ),
                0.42,
            ),
        ),
        patch(
            "src.inference.model_manager."
            "MlflowClient",
        ) as client_class,
        patch(
            "src.inference.model_manager."
            "load_feature_schema_from_mlflow",
            return_value=feature_schema,
        ) as load_schema,
    ):
        client_class.return_value\
            .get_model_version_by_alias\
            .return_value = model_version

        bundle = reload_serving_model(
            model_name=(
                "customer-churn-model-dev"
            ),
            cfg={},
        )

    assert isinstance(
        bundle,
        ServingBundle,
    )
    assert bundle.model is model
    assert bundle.model_name == (
        "customer-churn-model-dev"
    )
    assert bundle.model_type == "xgboost"
    assert bundle.serving_alias == "champion"
    assert bundle.model_version == "7"
    assert bundle.model_run_id == "run-7"
    assert bundle.decision_threshold == 0.42
    assert bundle.feature_schema == (
        feature_schema
    )

    set_tracking_uri.assert_called_once_with(
        "http://mlflow:5000"
    )
    load_schema.assert_called_once_with(
        run_id="run-7",
        fallback_to_local=True,
    )


def test_reload_rejects_invalid_feature_schema():
    model_version = MagicMock(
        version="7",
        run_id="run-7",
    )

    with (
        patch(
            "src.inference.model_manager."
            "resolve_tracking_uri",
            return_value="http://mlflow:5000",
        ),
        patch(
            "src.inference.model_manager."
            "mlflow.set_tracking_uri",
        ),
        patch(
            "src.inference.model_manager."
            "load_registry_model",
            return_value=(
                MagicMock(),
                "xgboost",
                "champion",
                (
                    "models:/"
                    "customer-churn-model-dev"
                    "@champion"
                ),
                0.42,
            ),
        ),
        patch(
            "src.inference.model_manager."
            "MlflowClient",
        ) as client_class,
        patch(
            "src.inference.model_manager."
            "load_feature_schema_from_mlflow",
            return_value={
                "columns": [],
                "dtypes": {},
            },
        ),
    ):
        client_class.return_value\
            .get_model_version_by_alias\
            .return_value = model_version

        with pytest.raises(
            ValueError,
            match="feature schema has no columns",
        ):
            reload_serving_model(
                model_name=(
                    "customer-churn-model-dev"
                ),
                cfg={},
            )


def test_reload_rejects_invalid_threshold():
    model_version = MagicMock(
        version="7",
        run_id="run-7",
    )

    with (
        patch(
            "src.inference.model_manager."
            "resolve_tracking_uri",
            return_value="http://mlflow:5000",
        ),
        patch(
            "src.inference.model_manager."
            "mlflow.set_tracking_uri",
        ),
        patch(
            "src.inference.model_manager."
            "load_registry_model",
            return_value=(
                MagicMock(),
                "xgboost",
                "champion",
                (
                    "models:/"
                    "customer-churn-model-dev"
                    "@champion"
                ),
                1.5,
            ),
        ),
        patch(
            "src.inference.model_manager."
            "MlflowClient",
        ) as client_class,
        patch(
            "src.inference.model_manager."
            "load_feature_schema_from_mlflow",
            return_value={
                "columns": ["tenure"],
                "dtypes": {
                    "tenure": "float64",
                },
            },
        ),
    ):
        client_class.return_value\
            .get_model_version_by_alias\
            .return_value = model_version

        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            reload_serving_model(
                model_name=(
                    "customer-churn-model-dev"
                ),
                cfg={},
            )