from __future__ import annotations

import argparse
import json
from datetime import (
    datetime,
)
from pathlib import Path

from mlflow.tracking import MlflowClient

from scripts.archive_churn_demo_results import (
    archive_results,
)
from scripts.prepare_churn_cohort_shift import (
    MANIFEST_FILE,
    prepare_scenario,
)
from scripts.reset_churn_demo_run import (
    reset_churn_demo_run,
)
from scripts.run_churn_demo import (
    parse_start_at,
    run_demo,
)
from src.configs.loader import (
    load_config,
)


def load_champion_identity() -> dict:
    client = MlflowClient()
    model_name = (
        load_config()[
            "model"
        ][
            "registry_name"
        ]
    )

    version = (
        client.get_model_version_by_alias(
            model_name,
            "champion",
        )
    )

    return {
        "model_name": model_name,
        "model_version": str(
            version.version
        ),
        "model_run_id": (
            version.run_id
        ),
    }


def load_scenario_manifest() -> dict:
    manifest_path = Path(
        MANIFEST_FILE
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Scenario manifest was not "
            f"created: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def prepare_run(
    *,
    max_days: int,
    batch_size: int,
    drift_start_day: int,
    ramp_days: int,
    baseline_rate: float,
    post_drift_rate: float,
    random_state: int,
) -> dict:
    reset_churn_demo_run()

    return prepare_scenario(
        max_days=max_days,
        batch_size=batch_size,
        drift_start_day=(
            drift_start_day
        ),
        ramp_days=ramp_days,
        baseline_rate=baseline_rate,
        post_drift_rate=(
            post_drift_rate
        ),
        random_state=random_state,
    )


def run_branch(
    *,
    experiment_name: str,
    branch_name: str,
    retraining_mode: str,
    max_days: int,
    batch_size: int,
    label_delay_days: int,
    start_at: datetime,
) -> Path:
    run_demo(
        batch_size=batch_size,
        max_days=max_days,
        label_delay_days=(
            label_delay_days
        ),
        start_at=start_at,
        retraining_mode=(
            retraining_mode
        ),
    )

    return archive_results(
        run_name=(
            f"{experiment_name}/"
            f"{branch_name}"
        ),
        retraining_mode=(
            retraining_mode
        ),
        scenario=(
            "controlled_real_"
            "cohort_shift"
        ),
        max_days=max_days,
        batch_size=batch_size,
        label_delay_days=(
            label_delay_days
        ),
        start_at=(
            start_at.isoformat()
        ),
    )


def run_experiment(
    *,
    experiment_name: str,
    max_days: int,
    batch_size: int,
    label_delay_days: int,
    start_at: datetime,
    drift_start_day: int,
    ramp_days: int,
    baseline_rate: float,
    post_drift_rate: float,
    random_state: int,
) -> dict:
    initial_champion = (
        load_champion_identity()
    )

    print(
        "Initial Champion:"
    )
    print(
        json.dumps(
            initial_champion,
            indent=2,
            sort_keys=True,
        )
    )

    static_manifest = prepare_run(
        max_days=max_days,
        batch_size=batch_size,
        drift_start_day=(
            drift_start_day
        ),
        ramp_days=ramp_days,
        baseline_rate=baseline_rate,
        post_drift_rate=(
            post_drift_rate
        ),
        random_state=random_state,
    )

    static_output = run_branch(
        experiment_name=(
            experiment_name
        ),
        branch_name=(
            "without_retraining"
        ),
        retraining_mode="disabled",
        max_days=max_days,
        batch_size=batch_size,
        label_delay_days=(
            label_delay_days
        ),
        start_at=start_at,
    )

    champion_after_static = (
        load_champion_identity()
    )

    if (
        champion_after_static
        != initial_champion
    ):
        raise RuntimeError(
            "Static branch changed the "
            "Champion unexpectedly."
        )

    adaptive_manifest = prepare_run(
        max_days=max_days,
        batch_size=batch_size,
        drift_start_day=(
            drift_start_day
        ),
        ramp_days=ramp_days,
        baseline_rate=baseline_rate,
        post_drift_rate=(
            post_drift_rate
        ),
        random_state=random_state,
    )

    static_hash = static_manifest[
        "ordered_customer_id_sha256"
    ]
    adaptive_hash = adaptive_manifest[
        "ordered_customer_id_sha256"
    ]

    if static_hash != adaptive_hash:
        raise RuntimeError(
            "Static and adaptive branches "
            "received different customer "
            "sequences."
        )

    champion_before_adaptive = (
        load_champion_identity()
    )

    if (
        champion_before_adaptive
        != initial_champion
    ):
        raise RuntimeError(
            "Adaptive branch did not start "
            "from the initial Champion."
        )

    adaptive_output = run_branch(
        experiment_name=(
            experiment_name
        ),
        branch_name="with_retraining",
        retraining_mode="enabled",
        max_days=max_days,
        batch_size=batch_size,
        label_delay_days=(
            label_delay_days
        ),
        start_at=start_at,
    )

    final_champion = (
        load_champion_identity()
    )

    result = {
        "experiment_name": (
            experiment_name
        ),
        "scenario_sha256": (
            static_hash
        ),
        "initial_champion": (
            initial_champion
        ),
        "final_champion": (
            final_champion
        ),
        "static_output": str(
            static_output
        ),
        "adaptive_output": str(
            adaptive_output
        ),
        "max_days": max_days,
        "batch_size": batch_size,
        "drift_start_day": (
            drift_start_day
        ),
        "retraining_changed_champion": (
            final_champion
            != initial_champion
        ),
    }

    summary_path = (
        Path(
            "results/"
            "churn_retraining_comparison"
        )
        / experiment_name
        / "experiment_summary.json"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            sort_keys=True,
        )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-name",
        default=(
            "controlled_cohort_shift"
        ),
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
        "--label-delay-days",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--start-at",
        type=parse_start_at,
        default=parse_start_at(
            "2026-09-01T00:00:00Z"
        ),
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
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_days = (
        2
        if args.smoke_test
        else args.max_days
    )

    experiment_name = (
        f"{args.experiment_name}_smoke"
        if args.smoke_test
        else args.experiment_name
    )

    run_experiment(
        experiment_name=(
            experiment_name
        ),
        max_days=max_days,
        batch_size=args.batch_size,
        label_delay_days=(
            args.label_delay_days
        ),
        start_at=args.start_at,
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


if __name__ == "__main__":
    main()