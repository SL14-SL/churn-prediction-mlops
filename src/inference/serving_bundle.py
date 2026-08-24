from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ServingBundle:
    """
    Complete validated state required for churn inference.

    A bundle is created completely before it replaces the currently
    active serving state.
    """

    model: Any
    model_name: str
    model_type: str

    decision_threshold: float
    feature_schema: dict[str, Any]

    serving_alias: str
    model_uri: str
    model_version: str
    model_run_id: str


def validate_serving_bundle(
    bundle: ServingBundle,
) -> None:
    """
    Raise ValueError if a churn serving bundle is incomplete or invalid.
    """
    if bundle.model is None:
        raise ValueError(
            "Serving bundle has no model."
        )

    if not bundle.model_name:
        raise ValueError(
            "Serving bundle has no model name."
        )

    if not bundle.model_type:
        raise ValueError(
            "Serving bundle has no model type."
        )

    if not bundle.serving_alias:
        raise ValueError(
            "Serving bundle has no serving alias."
        )

    if not bundle.model_uri:
        raise ValueError(
            "Serving bundle has no model URI."
        )

    if not bundle.model_version:
        raise ValueError(
            "Serving bundle has no model version."
        )

    if not bundle.model_run_id:
        raise ValueError(
            "Serving bundle has no model run ID."
        )

    if not isinstance(
        bundle.decision_threshold,
        (int, float),
    ):
        raise ValueError(
            "Serving bundle has an invalid decision threshold."
        )

    if not 0.0 <= float(
        bundle.decision_threshold
    ) <= 1.0:
        raise ValueError(
            "Serving bundle decision threshold must be between 0 and 1."
        )

    if not isinstance(
        bundle.feature_schema,
        dict,
    ):
        raise ValueError(
            "Serving bundle has an invalid feature schema."
        )

    columns = bundle.feature_schema.get(
        "columns"
    )

    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "Serving bundle feature schema has no columns."
        )

    if not all(
        isinstance(column, str) and column
        for column in columns
    ):
        raise ValueError(
            "Serving bundle feature schema contains invalid columns."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Serving bundle feature schema contains duplicate columns."
        )

    dtypes = bundle.feature_schema.get(
        "dtypes",
        {},
    )

    if not isinstance(dtypes, dict):
        raise ValueError(
            "Serving bundle feature schema has invalid dtypes."
        )

    unknown_dtype_columns = (
        set(dtypes) - set(columns)
    )

    if unknown_dtype_columns:
        raise ValueError(
            "Serving bundle feature schema contains dtypes "
            "for unknown columns: "
            f"{sorted(unknown_dtype_columns)}."
        )