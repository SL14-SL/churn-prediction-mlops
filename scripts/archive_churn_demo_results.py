from __future__ import annotations

import argparse
import json
from datetime import (
    datetime,
    timezone,
)
import fsspec
from pathlib import Path

import pandas as pd

from src.configs.loader import (
    file_exists,
    get_path,
)


MONITORING_PATH = get_path(
    "monitoring"
)
RAW_DATA_PATH = get_path(
    "raw_data"
)

SCENARIO_MANIFEST_FILE = (
    f"{RAW_DATA_PATH}/"
    "churn_cohort_shift_manifest.json"
)

RESULTS_ROOT = Path(
    "results/churn_retraining_comparison"
)

TABLES = {
    "performance_history": (
        f"{MONITORING_PATH}/"
        "churn_performance_history.parquet"
    ),
    "retraining_events": (
        f"{MONITORING_PATH}/"
        "retraining_event_history.parquet"
    ),
    "feature_drift_history": (
        f"{MONITORING_PATH}/"
        "feature_drift_history.parquet"
    ),
    "cumulative_ground_truth": (
        f"{MONITORING_PATH}/"
        "cumulative_ground_truth.csv"
    ),
}


def copy_table(
    *,
    source: str,
    destination: Path,
) -> int:
    if not file_exists(source):
        return 0

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if source.endswith(".parquet"):
        dataframe = pd.read_parquet(
            source
        )
        dataframe.to_parquet(
            destination,
            index=False,
        )
    elif source.endswith(".csv"):
        dataframe = pd.read_csv(
            source
        )
        dataframe.to_csv(
            destination,
            index=False,
        )
    else:
        raise ValueError(
            f"Unsupported result file: "
            f"{source}"
        )

    return len(dataframe)

def load_scenario_manifest() -> dict:
    if not file_exists(
        SCENARIO_MANIFEST_FILE
    ):
        return {}

    with fsspec.open(
        SCENARIO_MANIFEST_FILE,
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def archive_results(
    *,
    run_name: str,
    retraining_mode: str,
    scenario: str,
    max_days: int,
    batch_size: int,
    label_delay_days: int,
    start_at: str,
) -> Path:
    output_directory = (
        RESULTS_ROOT
        / run_name
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    archived_rows = {}

    for name, source in (
        TABLES.items()
    ):
        suffix = Path(source).suffix

        destination = (
            output_directory
            / f"{name}{suffix}"
        )

        archived_rows[name] = (
            copy_table(
                source=source,
                destination=destination,
            )
        )

    scenario_manifest = (
        load_scenario_manifest()
    )

    if scenario_manifest:
        scenario_manifest_path = (
            output_directory
            / "scenario_manifest.json"
        )

        with scenario_manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                scenario_manifest,
                file,
                indent=2,
                sort_keys=True,
            )

    metadata = {
        "run_name": run_name,
        "retraining_mode": (
            retraining_mode
        ),
        "scenario": scenario,
        "max_days": max_days,
        "batch_size": batch_size,
        "label_delay_days": (
            label_delay_days
        ),
        "start_at": start_at,
        "archived_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "archived_rows": archived_rows,
        "scenario_manifest": (
            scenario_manifest
        ),
        "scenario_sha256": (
            scenario_manifest.get(
                "ordered_customer_id_sha256"
            )
        ),
    }

    metadata_path = (
        output_directory
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )

    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
    )
    print(
        "Archived demo results to: "
        f"{output_directory}"
    )

    return output_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        required=True,
    )
    parser.add_argument(
        "--retraining-mode",
        choices=[
            "enabled",
            "disabled",
        ],
        required=True,
    )
    parser.add_argument(
        "--scenario",
        default="controlled_drift",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--label-delay-days",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--start-at",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    archive_results(
        run_name=args.run_name,
        retraining_mode=(
            args.retraining_mode
        ),
        scenario=args.scenario,
        max_days=args.max_days,
        batch_size=args.batch_size,
        label_delay_days=(
            args.label_delay_days
        ),
        start_at=args.start_at,
    )


if __name__ == "__main__":
    main()