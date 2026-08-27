from __future__ import annotations

import argparse
import subprocess

import fsspec
import pandas as pd

from datetime import datetime, timezone, timedelta
from src.configs.loader import file_exists, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = get_path("raw_data")
SIMULATION_FILE = f"{RAW_DATA_PATH}/simulation_ground_truth.csv"

MONITORING_PATH = get_path("monitoring")
GROUND_TRUTH_BATCH_DIR = f"{MONITORING_PATH}/ground_truth_batches"


def remaining_rows() -> int:
    if not file_exists(SIMULATION_FILE):
        return 0

    return len(pd.read_csv(SIMULATION_FILE))


def has_released_labels() -> bool:
    """
    Check whether released churn label batches exist.

    Uses fsspec so the same logic works for local paths and GCS paths.
    """
    pattern = f"{GROUND_TRUTH_BATCH_DIR}/ground_truth_churn*.csv"

    fs, fs_pattern = fsspec.core.url_to_fs(pattern)
    files = fs.glob(fs_pattern)

    return len(files) > 0


def run_command(cmd: list[str], description: str) -> None:
    logger.info("🚀 %s", description)

    subprocess.run(
        ["uv", "run", "--no-sync", *cmd],
        check=True,
    )

def parse_start_at(
    value: str,
) -> datetime:
    parsed = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(
        timezone.utc
    )

def run_demo(
    *,
    batch_size: int,
    max_days: int,
    label_delay_days: int,
    start_at: datetime,
) -> None:
    for day in range(1, max_days + 1):
        evaluated_at = (
            start_at
            + timedelta(days=day - 1)
        )
        evaluated_at_iso = (
            evaluated_at.isoformat()
        )
        remaining = remaining_rows()

        if remaining <= 0:
            logger.info("🏁 Simulation pool is empty. Demo finished.")
            break

        logger.info("=" * 80)
        logger.info("📅 Simulation Day %s | remaining_rows=%s", day, remaining)
        logger.info("=" * 80)

        # 1. Score new customers and store them as pending labels.
        run_command(
            [
                "python",
                "scripts/simulate_churn_batch.py",
                "--batch-size",
                str(batch_size),
                "--simulation-day",
                str(day),
                "--label-delay-days",
                str(label_delay_days),
            ],
            f"Scoring churn batch for day {day}",
        )

        # 2. Release labels that became available today.
        run_command(
            [
                "python",
                "scripts/release_churn_labels.py",
                "--simulation-day",
                str(day),
            ],
            f"Releasing delayed labels for day {day}",
        )

        # 3. Evaluate only if labels are available.
        if not has_released_labels():
            logger.info(
                "No released labels available yet. Skipping performance evaluation and retraining."
            )
            day += 1
            continue

        run_command(
            [
                "python",
                "scripts/run_performance_demo.py",
                "--simulation-day",
                str(day),
                "--evaluated-at",
                evaluated_at_iso,
            ],
            f"Evaluating churn performance for day {day}",
        )

        run_command(
            [
                "python",
                "-m",
                "flows.auto_retrain_flow",
                "--evaluated-at",
                evaluated_at_iso,
                "--simulation-day",
                str(day),
            ],
            (
                "Running auto-retrain decision "
                f"for day {day}"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-days", type=int, default=10)
    parser.add_argument("--label-delay-days", type=int, default=1)
    parser.add_argument(
        "--start-at",
        type=parse_start_at,
        default=datetime.now(timezone.utc),
    )
    args = parser.parse_args()

    run_demo(
        batch_size=args.batch_size,
        max_days=args.max_days,
        label_delay_days=args.label_delay_days,
        start_at=args.start_at,
    )