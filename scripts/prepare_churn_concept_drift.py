from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from src.configs.loader import (
    file_exists,
    get_path,
)


RAW_DATA_PATH = get_path(
    "raw_data"
)
SIMULATION_FILE = (
    f"{RAW_DATA_PATH}/"
    "simulation_ground_truth.csv"
)
MANIFEST_FILE = (
    f"{RAW_DATA_PATH}/"
    "churn_concept_drift_manifest.json"
)
AUDIT_FILE = (
    f"{RAW_DATA_PATH}/"
    "churn_concept_drift_audit.csv"
)


def normalize_churn(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            }
        )
    )

    if normalized.isna().any():
        invalid_values = sorted(
            series[
                normalized.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "Invalid churn labels: "
            f"{invalid_values}"
        )

    return normalized.astype(int)


def calculate_drift_cohort(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Select customers whose conditional churn behavior changes.

    Customer features remain unchanged. Only original non-churn
    labels in this cohort may be flipped by the controlled scenario.
    """
    return (
        dataframe["Contract"]
        .eq("Month-to-month")
        & (
            dataframe["tenure"]
            >= 13
        )
        & (
            ~dataframe[
                "PaymentMethod"
            ].eq("Electronic check")
        )
    )


def target_flip_rate(
    *,
    day: int,
    drift_start_day: int,
    ramp_days: int,
    post_drift_rate: float,
) -> float:
    if day < drift_start_day:
        return 0.0

    ramp_position = (
        day
        - drift_start_day
        + 1
    )

    if ramp_position >= ramp_days:
        return post_drift_rate

    return (
        post_drift_rate
        * ramp_position
        / ramp_days
    )


def hash_values(
    values: pd.Series,
) -> str:
    payload = "|".join(
        values.astype(str)
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def validate_arguments(
    *,
    max_days: int,
    batch_size: int,
    drift_start_day: int,
    ramp_days: int,
    post_drift_rate: float,
) -> None:
    if max_days < 1:
        raise ValueError(
            "max_days must be positive."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive."
        )

    if drift_start_day < 1:
        raise ValueError(
            "drift_start_day must be positive."
        )

    if ramp_days < 1:
        raise ValueError(
            "ramp_days must be positive."
        )

    if not 0.0 <= post_drift_rate <= 1.0:
        raise ValueError(
            "post_drift_rate must be "
            "between 0 and 1."
        )


def prepare_scenario(
    *,
    max_days: int,
    batch_size: int,
    drift_start_day: int,
    ramp_days: int,
    post_drift_rate: float,
    random_state: int,
) -> dict:
    validate_arguments(
        max_days=max_days,
        batch_size=batch_size,
        drift_start_day=(
            drift_start_day
        ),
        ramp_days=ramp_days,
        post_drift_rate=(
            post_drift_rate
        ),
    )

    if not file_exists(
        SIMULATION_FILE
    ):
        raise FileNotFoundError(
            "Simulation ground truth does "
            f"not exist: {SIMULATION_FILE}"
        )

    dataframe = pd.read_csv(
        SIMULATION_FILE
    )

    required_columns = {
        "customerID",
        "Churn",
        "tenure",
        "Contract",
        "PaymentMethod",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Simulation data is missing: "
            f"{sorted(missing_columns)}"
        )

    required_rows = (
        max_days
        * batch_size
    )

    if len(dataframe) < required_rows:
        raise ValueError(
            "Not enough simulation rows | "
            f"required={required_rows} "
            f"available={len(dataframe)}"
        )

    working = (
        dataframe
        .sample(
            frac=1,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    original_churn = normalize_churn(
        working["Churn"]
    )
    drift_cohort = (
        calculate_drift_cohort(
            working
        )
    )

    selected = (
        working
        .iloc[:required_rows]
        .copy()
    )
    remaining = (
        working
        .iloc[required_rows:]
        .copy()
    )

    selected_original_churn = (
        original_churn
        .iloc[:required_rows]
        .reset_index(drop=True)
    )
    selected_drift_cohort = (
        drift_cohort
        .iloc[:required_rows]
        .reset_index(drop=True)
    )

    selected = selected.reset_index(
        drop=True
    )
    effective_churn = (
        selected_original_churn.copy()
    )

    audit_rows = []
    daily_summary = []

    for day in range(
        1,
        max_days + 1,
    ):
        start = (
            (day - 1)
            * batch_size
        )
        end = (
            start
            + batch_size
        )

        batch = selected.iloc[
            start:end
        ]
        batch_original = (
            selected_original_churn
            .iloc[start:end]
            .copy()
        )
        batch_cohort = (
            selected_drift_cohort
            .iloc[start:end]
            .copy()
        )

        flip_rate = target_flip_rate(
            day=day,
            drift_start_day=(
                drift_start_day
            ),
            ramp_days=ramp_days,
            post_drift_rate=(
                post_drift_rate
            ),
        )

        eligible = (
            batch_cohort
            & batch_original.eq(0)
        )

        day_random = random.Random(
            random_state
            + day * 10_000
        )

        applied = pd.Series(
            False,
            index=batch.index,
            dtype=bool,
        )

        for row_index in batch.index:
            local_position = (
                row_index - start
            )

            if not bool(
                eligible.iloc[
                    local_position
                ]
            ):
                continue

            if (
                day_random.random()
                < flip_rate
            ):
                effective_churn.iloc[
                    row_index
                ] = 1
                applied.loc[
                    row_index
                ] = True

        batch_effective = (
            effective_churn
            .iloc[start:end]
        )

        original_rate = float(
            batch_original.mean()
        )
        effective_rate = float(
            batch_effective.mean()
        )

        daily_summary.append(
            {
                "simulation_day": day,
                "customers": int(
                    len(batch)
                ),
                "drift_cohort_customers": int(
                    batch_cohort.sum()
                ),
                "eligible_non_churn_customers": int(
                    eligible.sum()
                ),
                "target_flip_rate": float(
                    flip_rate
                ),
                "labels_flipped": int(
                    applied.sum()
                ),
                "actual_flip_rate_among_eligible": (
                    float(
                        applied.sum()
                        / eligible.sum()
                    )
                    if eligible.sum() > 0
                    else 0.0
                ),
                "original_churn_rate": (
                    original_rate
                ),
                "effective_churn_rate": (
                    effective_rate
                ),
                "churn_rate_increase": float(
                    effective_rate
                    - original_rate
                ),
            }
        )

        for row_index in batch.index:
            local_position = (
                row_index - start
            )

            audit_rows.append(
                {
                    "customerID": (
                        batch.loc[
                            row_index,
                            "customerID",
                        ]
                    ),
                    "simulation_day": day,
                    "Churn_original": (
                        "Yes"
                        if int(
                            batch_original.iloc[
                                local_position
                            ]
                        )
                        == 1
                        else "No"
                    ),
                    "Churn_effective": (
                        "Yes"
                        if int(
                            batch_effective.iloc[
                                local_position
                            ]
                        )
                        == 1
                        else "No"
                    ),
                    "concept_drift_cohort": bool(
                        batch_cohort.iloc[
                            local_position
                        ]
                    ),
                    "concept_drift_eligible": bool(
                        eligible.iloc[
                            local_position
                        ]
                    ),
                    "concept_drift_applied": bool(
                        applied.loc[
                            row_index
                        ]
                    ),
                    "target_flip_rate": float(
                        flip_rate
                    ),
                }
            )

    selected["Churn"] = (
        effective_churn
        .map(
            {
                0: "No",
                1: "Yes",
            }
        )
    )

    ordered = pd.concat(
        [
            selected,
            remaining,
        ],
        ignore_index=True,
    )

    ordered.to_csv(
        SIMULATION_FILE,
        index=False,
    )

    audit = pd.DataFrame(
        audit_rows
    )
    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    selected_effective_churn = (
        effective_churn.astype(str)
    )

    manifest = {
        "scenario": (
            "controlled_synthetic_"
            "concept_drift"
        ),
        "synthetic_features": False,
        "synthetic_labels": True,
        "max_days": max_days,
        "batch_size": batch_size,
        "drift_start_day": (
            drift_start_day
        ),
        "ramp_days": ramp_days,
        "post_drift_flip_rate": (
            post_drift_rate
        ),
        "random_state": random_state,
        "drift_cohort_definition": {
            "contract": (
                "Month-to-month"
            ),
            "minimum_tenure": 13,
            "excluded_payment_method": (
                "Electronic check"
            ),
            "original_label": "No",
        },
        "ordered_customer_id_sha256": (
            hash_values(
                ordered["customerID"]
            )
        ),
        "selected_effective_label_sha256": (
            hash_values(
                selected_effective_churn
            )
        ),
        "available_drift_cohort_rows": int(
            drift_cohort.sum()
        ),
        "selected_drift_cohort_rows": int(
            selected_drift_cohort.sum()
        ),
        "selected_labels_flipped": int(
            audit[
                "concept_drift_applied"
            ].sum()
        ),
        "audit_file": AUDIT_FILE,
        "daily_summary": daily_summary,
    }

    manifest_path = Path(
        MANIFEST_FILE
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True,
        )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a reproducible controlled "
            "churn concept-drift scenario."
        )
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--drift-start-day",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--ramp-days",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--post-drift-rate",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = prepare_scenario(
        max_days=args.max_days,
        batch_size=args.batch_size,
        drift_start_day=(
            args.drift_start_day
        ),
        ramp_days=args.ramp_days,
        post_drift_rate=(
            args.post_drift_rate
        ),
        random_state=(
            args.random_state
        ),
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()