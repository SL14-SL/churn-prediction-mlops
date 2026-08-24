import pandas as pd
import pytest

from src.deployment.prediction_probe import (
    PROBE_COLUMNS,
    build_prediction_probe,
)


def build_customer(
    customer_id: str,
) -> dict:
    return {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": (
            "Electronic check"
        ),
        "MonthlyCharges": 70.35,
        "TotalCharges": "845.50",
        "Churn": "Yes",
    }


def test_build_prediction_probe(
    tmp_path,
):
    validated_path = (
        tmp_path / "train.parquet"
    )

    pd.DataFrame(
        [
            build_customer(
                "9000-ZZZZZ"
            ),
            build_customer(
                "1000-AAAAA"
            ),
        ]
    ).to_parquet(
        validated_path,
        index=False,
    )

    probe = build_prediction_probe(
        validated_data_path=str(
            validated_path
        ),
    )

    assert len(
        probe["inputs"]
    ) == 1

    prediction_input = (
        probe["inputs"][0]
    )

    assert prediction_input[
        "customerID"
    ] == "1000-AAAAA"

    assert set(
        prediction_input
    ) == set(PROBE_COLUMNS)

    assert "Churn" not in (
        prediction_input
    )

    assert (
        probe["context"]["purpose"]
        == "post_deployment_verification"
    )


def test_probe_is_deterministic(
    tmp_path,
):
    validated_path = (
        tmp_path / "train.parquet"
    )

    pd.DataFrame(
        [
            build_customer(
                "3000-CCCCC"
            ),
            build_customer(
                "1000-AAAAA"
            ),
            build_customer(
                "2000-BBBBB"
            ),
        ]
    ).to_parquet(
        validated_path,
        index=False,
    )

    first = build_prediction_probe(
        validated_data_path=str(
            validated_path
        ),
    )

    second = build_prediction_probe(
        validated_data_path=str(
            validated_path
        ),
    )

    assert first == second
    assert first["inputs"][0][
        "customerID"
    ] == "1000-AAAAA"


def test_probe_rejects_missing_columns(
    tmp_path,
):
    validated_path = (
        tmp_path / "train.parquet"
    )

    customer = build_customer(
        "1000-AAAAA"
    )
    customer.pop(
        "MonthlyCharges"
    )

    pd.DataFrame(
        [customer]
    ).to_parquet(
        validated_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        build_prediction_probe(
            validated_data_path=str(
                validated_path
            ),
        )


def test_probe_rejects_empty_dataset(
    tmp_path,
):
    validated_path = (
        tmp_path / "train.parquet"
    )

    pd.DataFrame(
        columns=PROBE_COLUMNS
    ).to_parquet(
        validated_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="no usable customer row",
    ):
        build_prediction_probe(
            validated_data_path=str(
                validated_path
            ),
        )