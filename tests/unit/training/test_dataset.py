from unittest.mock import MagicMock

import pandas as pd

from src.training import dataset


def test_load_and_prepare_validation_data(
    monkeypatch,
):
    validation_frame = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
            ],
            "feature_b": [
                3.0,
                4.0,
            ],
            "churn": [
                "No",
                "Yes",
            ],
            "customerid": [
                "customer-1",
                "customer-2",
            ],
        }
    )

    monkeypatch.setattr(
        dataset,
        "get_path",
        lambda name: "data/splits",
    )
    monkeypatch.setattr(
        dataset,
        "build_drop_columns",
        MagicMock(
            return_value=[
                "churn",
                "customerid",
            ]
        ),
    )
    monkeypatch.setattr(
        dataset.pd,
        "read_parquet",
        MagicMock(
            return_value=validation_frame
        ),
    )
    monkeypatch.setattr(
        dataset,
        "TRAIN_CFG",
        {
            "data": {
                "target_column": "Churn",
            }
        },
    )

    X_val, y_val = (
        dataset
        .load_and_prepare_validation_data()
    )

    assert X_val.columns.tolist() == [
        "feature_a",
        "feature_b",
    ]

    assert y_val.tolist() == [
        0,
        1,
    ]


def test_load_recent_production_data_returns_none_when_disabled(
    monkeypatch,
):
    monkeypatch.setattr(
        dataset,
        "TRAIN_CFG",
        {
            "promotion": {
                "recent_evaluation": {
                    "enabled": False,
                }
            }
        },
    )

    result = (
        dataset
        .load_and_prepare_recent_production_data(
            reference_columns=[
                "feature_a",
            ]
        )
    )

    assert result is None


def test_load_recent_production_data_returns_none_when_file_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        dataset,
        "TRAIN_CFG",
        {
            "promotion": {
                "recent_evaluation": {
                    "enabled": True,
                    "window_size": 10,
                    "minimum_samples": 5,
                }
            }
        },
    )
    monkeypatch.setattr(
        dataset,
        "get_path",
        lambda name: "data/monitoring",
    )
    monkeypatch.setattr(
        dataset,
        "file_exists",
        MagicMock(
            return_value=False
        ),
    )

    result = (
        dataset
        .load_and_prepare_recent_production_data(
            reference_columns=[
                "feature_a",
            ]
        )
    )

    assert result is None


def test_load_recent_production_data_returns_none_with_too_few_samples(
    monkeypatch,
):
    recent_frame = pd.DataFrame(
        {
            "churn": [
                0,
                1,
            ],
            "released_simulation_day": [
                1,
                2,
            ],
        }
    )

    monkeypatch.setattr(
        dataset,
        "TRAIN_CFG",
        {
            "promotion": {
                "recent_evaluation": {
                    "enabled": True,
                    "window_size": 10,
                    "minimum_samples": 3,
                }
            }
        },
    )
    monkeypatch.setattr(
        dataset,
        "get_path",
        lambda name: "data/monitoring",
    )
    monkeypatch.setattr(
        dataset,
        "file_exists",
        MagicMock(
            return_value=True
        ),
    )
    monkeypatch.setattr(
        dataset.pd,
        "read_csv",
        MagicMock(
            return_value=recent_frame
        ),
    )

    result = (
        dataset
        .load_and_prepare_recent_production_data(
            reference_columns=[
                "feature_a",
            ]
        )
    )

    assert result is None


def test_load_recent_production_data_builds_and_aligns_features(
    monkeypatch,
):
    recent_frame = pd.DataFrame(
        {
            "tenure": [
                12,
                24,
            ],
            "monthlycharges": [
                60.0,
                80.0,
            ],
            "churn": [
                0,
                1,
            ],
            "released_simulation_day": [
                1,
                2,
            ],
            "prediction_timestamp": [
                "2026-01-01T10:00:00Z",
                "2026-01-02T10:00:00Z",
            ],
        }
    )

    transformed_frame = pd.DataFrame(
        {
            "tenure": [
                12,
                24,
            ],
            "monthlycharges": [
                60.0,
                80.0,
            ],
            "derived_feature": [
                True,
                False,
            ],
        }
    )

    monkeypatch.setattr(
        dataset,
        "TRAIN_CFG",
        {
            "promotion": {
                "recent_evaluation": {
                    "enabled": True,
                    "window_size": 2,
                    "minimum_samples": 2,
                }
            },
            "features": {
                "numeric_columns": [
                    "tenure",
                    "monthlycharges",
                ],
                "categorical_columns": [],
                "derived_columns": [
                    "derived_feature",
                ],
            },
        },
    )
    monkeypatch.setattr(
        dataset,
        "get_path",
        lambda name: "data/monitoring",
    )
    monkeypatch.setattr(
        dataset,
        "file_exists",
        MagicMock(
            return_value=True
        ),
    )
    monkeypatch.setattr(
        dataset.pd,
        "read_csv",
        MagicMock(
            return_value=recent_frame
        ),
    )

    build_features_mock = MagicMock(
        return_value=transformed_frame
    )

    monkeypatch.setattr(
        dataset,
        "build_features",
        build_features_mock,
    )

    X_recent, y_recent = (
        dataset
        .load_and_prepare_recent_production_data(
            reference_columns=[
                "tenure",
                "monthlycharges",
                "derived_feature",
                "missing_reference_feature",
            ]
        )
    )

    assert X_recent.columns.tolist() == [
        "tenure",
        "monthlycharges",
        "derived_feature",
        "missing_reference_feature",
    ]
    assert (
        X_recent[
            "missing_reference_feature"
        ].eq(False).all()
    )
    assert y_recent.tolist() == [
        0,
        1,
    ]

    raw_features = (
        build_features_mock
        .call_args.args[0]
    )

    assert raw_features.columns.tolist() == [
        "tenure",
        "monthlycharges",
    ]


def test_normalize_column_name():
    assert (
        dataset._normalize_column_name(
            "Monthly Charges (EUR)"
        )
        == "monthly_charges_eur"
    )