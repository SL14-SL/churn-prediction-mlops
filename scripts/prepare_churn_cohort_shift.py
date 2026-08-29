from __future__ import annotations

import argparse
import hashlib
import json
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
    "churn_cohort_shift_manifest.json"
)


def calculate_risk_score(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Build a cohort score from existing customer attributes.

    No feature values or churn labels are modified.
    """
    return (
        (
            dataframe["tenure"]
            <= 12
        ).astype(int)
        + dataframe["Contract"]
        .eq("Month-to-month")
        .astype(int)
        + dataframe["InternetService"]
        .eq("Fiber optic")
        .astype(int)
        + dataframe["PaymentMethod"]
        .eq("Electronic check")
        .astype(int)
        + dataframe["PaperlessBilling"]
        .eq("Yes")
        .astype(int)
    )


def target_high_risk_rate(
    *,
    day: int,
    drift_start_day: int,
    ramp_days: int,
    baseline_rate: float,
    post_drift_rate: float,
) -> float:
    if day < drift_start_day:
        return baseline_rate

    ramp_position = (
        day - drift_start_day + 1
    )

    if ramp_position >= ramp_days:
        return post_drift_rate

    progress = (
        ramp_position
        / ramp_days
    )

    return (
        baseline_rate
        + (
            post_drift_rate
            - baseline_rate
        )
        * progress
    )


def prepare_scenario(
    *,
    max_days: int,
    batch_size: int,
    drift_start_day: int,
    ramp_days: int,
    baseline_rate: float,
    post_drift_rate: float,
    random_state: int,
) -> dict:
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
        "InternetService",
        "PaymentMethod",
        "PaperlessBilling",
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

    working = dataframe.copy()
    working["_risk_score"] = (
        calculate_risk_score(
            working
        )
    )
    working["_churn_binary"] = (
        working["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
            }
        )
    )

    high_risk = (
        working[
            working["_risk_score"]
            >= 3
        ]
        .sample(
            frac=1,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    other = (
        working[
            working["_risk_score"]
            < 3
        ]
        .sample(
            frac=1,
            random_state=(
                random_state + 1
            ),
        )
        .reset_index(drop=True)
    )

    batches = []
    daily_summary = []

    high_cursor = 0
    other_cursor = 0

    for day in range(
        1,
        max_days + 1,
    ):
        high_rate = (
            target_high_risk_rate(
                day=day,
                drift_start_day=(
                    drift_start_day
                ),
                ramp_days=ramp_days,
                baseline_rate=(
                    baseline_rate
                ),
                post_drift_rate=(
                    post_drift_rate
                ),
            )
        )

        high_count = int(
            round(
                batch_size
                * high_rate
            )
        )
        other_count = (
            batch_size
            - high_count
        )

        high_batch = high_risk.iloc[
            high_cursor:
            high_cursor + high_count
        ]
        other_batch = other.iloc[
            other_cursor:
            other_cursor + other_count
        ]

        if len(high_batch) != high_count:
            raise ValueError(
                "Not enough high-risk rows "
                f"for simulation day {day}."
            )

        if len(other_batch) != other_count:
            raise ValueError(
                "Not enough other rows "
                f"for simulation day {day}."
            )

        batch = pd.concat(
            [
                high_batch,
                other_batch,
            ],
            ignore_index=True,
        ).sample(
            frac=1,
            random_state=(
                random_state + day
            ),
        )

        batches.append(
            batch
        )

        daily_summary.append(
            {
                "simulation_day": day,
                "customers": int(
                    len(batch)
                ),
                "target_high_risk_rate": (
                    high_rate
                ),
                "actual_high_risk_rate": float(
                    (
                        batch[
                            "_risk_score"
                        ]
                        >= 3
                    ).mean()
                ),
                "churn_rate": float(
                    batch[
                        "_churn_binary"
                    ].mean()
                ),
                "avg_risk_score": float(
                    batch[
                        "_risk_score"
                    ].mean()
                ),
            }
        )

        high_cursor += high_count
        other_cursor += other_count

    selected = pd.concat(
        batches,
        ignore_index=True,
    )

    selected_customer_ids = set(
        selected["customerID"]
    )

    remaining = working[
        ~working["customerID"].isin(
            selected_customer_ids
        )
    ].sample(
        frac=1,
        random_state=(
            random_state + 10_000
        ),
    )

    ordered = pd.concat(
        [
            selected,
            remaining,
        ],
        ignore_index=True,
    )

    ordered = ordered.drop(
        columns=[
            "_risk_score",
            "_churn_binary",
        ]
    )

    ordered.to_csv(
        SIMULATION_FILE,
        index=False,
    )

    ordered_id_payload = "|".join(
        ordered["customerID"].astype(str)
    )

    manifest = {
        "scenario": (
            "controlled_real_cohort_shift"
        ),
        "synthetic_features": False,
        "synthetic_labels": False,
        "max_days": max_days,
        "batch_size": batch_size,
        "drift_start_day": (
            drift_start_day
        ),
        "ramp_days": ramp_days,
        "baseline_high_risk_rate": (
            baseline_rate
        ),
        "post_drift_high_risk_rate": (
            post_drift_rate
        ),
        "high_risk_threshold": 3,
        "random_state": random_state,
        "ordered_customer_id_sha256": (
            hashlib.sha256(
                ordered_id_payload.encode(
                    "utf-8"
                )
            ).hexdigest()
        ),
        "available_high_risk_rows": int(
            len(high_risk)
        ),
        "available_other_rows": int(
            len(other)
        ),
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
    parser = argparse.ArgumentParser()

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
        "--baseline-rate",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--post-drift-rate",
        type=float,
        default=0.80,
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
        baseline_rate=(
            args.baseline_rate
        ),
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