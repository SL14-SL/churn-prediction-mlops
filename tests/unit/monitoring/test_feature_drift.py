from unittest.mock import patch

from src.monitoring.feature_drift import (
    get_feature_columns_from_training_config,
)


def test_derived_features_can_be_excluded_from_drift_monitoring():
    training_config = {
        "features": {
            "numeric_columns": [
                "tenure",
                "is_new_customer",
            ],
            "categorical_columns": [
                "contract",
                "tenure_group",
            ],
        }
    }

    monitoring_config = {
        "feature_drift": {
            "excluded_features": [
                "is_new_customer",
                "tenure_group",
            ]
        }
    }

    with patch(
        "src.monitoring.feature_drift.load_config",
        side_effect=[
            training_config,
            monitoring_config,
        ],
    ):
        (
            numeric_features,
            categorical_features,
        ) = (
            get_feature_columns_from_training_config()
        )

    assert numeric_features == [
        "tenure",
    ]
    assert categorical_features == [
        "contract",
    ]