from __future__ import annotations

from typing import Any

import pandas as pd


PROBE_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def _to_json_value(
    value: Any,
) -> Any:
    """
    Convert pandas and NumPy scalar values into JSON-compatible values.
    """
    if pd.isna(value):
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    return value


def build_prediction_probe(
    *,
    validated_data_path: str,
) -> dict[str, Any]:
    """
    Build one deterministic churn prediction request from validated data.

    The customer with the lexicographically smallest customer ID is used,
    making repeated release builds reproducible for the same dataset.
    """
    source_df = pd.read_parquet(
        validated_data_path
    )

    missing_columns = [
        column
        for column in PROBE_COLUMNS
        if column not in source_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction probe source is missing "
            f"columns: {missing_columns}"
        )

    probe_df = source_df[
        PROBE_COLUMNS
    ].copy()

    probe_df["customerID"] = (
        probe_df["customerID"]
        .astype("string")
        .str.strip()
    )

    probe_df = probe_df.dropna(
        subset=["customerID"]
    )

    probe_df = probe_df.loc[
        probe_df["customerID"] != ""
    ]

    if probe_df.empty:
        raise ValueError(
            "Prediction probe source contains "
            "no usable customer row."
        )

    probe_row = (
        probe_df.sort_values(
            "customerID",
            ascending=True,
        )
        .iloc[0]
    )

    prediction_input = {
        column: _to_json_value(
            probe_row[column]
        )
        for column in PROBE_COLUMNS
    }

    return {
        "inputs": [
            prediction_input,
        ],
        "context": {
            "purpose": (
                "post_deployment_verification"
            ),
        },
    }