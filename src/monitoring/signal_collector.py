from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from typing import Any

import fsspec
import pandas as pd

from src.configs.loader import get_path

from src.storage.filesystem import file_exists
from src.configs.paths import join_uri

from src.inference.releases.repository import (
    load_active_serving_manifest,
)
from src.data.validation.validate import (
    validate_train,
)
from src.monitoring.config import (
    get_retraining_settings,
)
from src.monitoring.retraining_policy import (
    RetrainingSignals,
)
from src.monitoring.retraining_state import (
    load_retraining_state,
)
from src.monitoring.signal_evaluation import (
    evaluate_performance_degradation,
    evaluate_persistent_feature_drift,
)


def _utc_timestamp(
    value: datetime | pd.Timestamp | None = None,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(
        value or datetime.now(timezone.utc)
    )

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def _list_files(pattern: str) -> list[str]:
    """
    List local or remote files while preserving their protocol.
    """

    filesystem, fs_pattern = (
        fsspec.core.url_to_fs(pattern)
    )

    matches = filesystem.glob(fs_pattern)

    return sorted(
        filesystem.unstrip_protocol(path)
        for path in matches
    )


def _read_parquet_if_available(
    path: str,
) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()

    with fsspec.open(path, "rb") as file:
        return pd.read_parquet(file)


def _read_ground_truth_batches(
    batch_files: list[str],
    *,
    processed_batch_ids: set[str],
) -> tuple[
    int,
    str | None,
    tuple[str, ...],
    bool,
    str | None,
]:
    """
    Validate all available Ground-Truth batches.

    Rows are counted as new only when their content-based batch ID was
    not recorded by a previous successful retraining run.
    """

    if not batch_files:
        return (
            0,
            None,
            (),
            True,
            "No Ground-Truth batches available.",
        )

    total_new_rows = 0
    current_batch_ids: set[str] = set()

    for batch_path in batch_files:
        try:
            with fsspec.open(
                batch_path,
                "rb",
            ) as file:
                raw_content = file.read()

            batch_id = (
                "gt-"
                + hashlib.sha256(
                    raw_content
                ).hexdigest()[:20]
            )

            batch_df = pd.read_csv(
                io.BytesIO(raw_content)
            )

            validated_df = validate_train(
                batch_df
            )

            is_new_batch = (
                batch_id
                not in processed_batch_ids
                and batch_id
                not in current_batch_ids
            )

            if is_new_batch:
                total_new_rows += len(
                    validated_df
                )

            current_batch_ids.add(batch_id)

        except Exception as error:
            return (
                total_new_rows,
                None,
                tuple(
                    sorted(current_batch_ids)
                ),
                False,
                (
                    "Ground-Truth batch validation "
                    f"failed for {batch_path}: "
                    f"{error}"
                ),
            )

    sorted_batch_ids = tuple(
        sorted(current_batch_ids)
    )

    fingerprint_payload = "|".join(
        sorted_batch_ids
    )
    dataset_version = (
        "batch-"
        + hashlib.sha256(
            fingerprint_payload.encode(
                "utf-8"
            )
        ).hexdigest()[:16]
    )

    processed_count = len(
        current_batch_ids
        & processed_batch_ids
    )
    new_count = (
        len(current_batch_ids)
        - processed_count
    )

    return (
        total_new_rows,
        dataset_version,
        sorted_batch_ids,
        True,
        (
            f"Validated "
            f"{len(current_batch_ids)} unique "
            "Ground-Truth batches "
            f"({new_count} new, "
            f"{processed_count} already processed)."
        ),
    )


def _cooldown_active(
    state: dict[str, Any],
    *,
    evaluated_at: pd.Timestamp,
    cooldown_hours: int,
) -> bool:
    """
    Determine whether the latest successful retraining is within the cooldown.

    Returns:
        True if the cooldown period is active, otherwise False.
    """
    last_retrained_at = state.get(
        "last_retrained_at_utc"
    )

    if not last_retrained_at:
        return False

    parsed = pd.to_datetime(
        last_retrained_at,
        utc=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return False

    elapsed = evaluated_at - parsed

    return elapsed < pd.Timedelta(
        hours=cooldown_hours
    )

def _resolve_last_training_at_utc(
    state: dict[str, Any],
    *,
    models_path: str,
) -> str | None:
    """
    Resolve the latest successful training timestamp.

    Prefer the retraining state because it also represents rejected
    Candidates. Fall back to the active serving release for bootstrap.
    """

    state_timestamp = state.get(
        "last_retrained_at_utc"
    )

    if state_timestamp:
        return str(state_timestamp)

    try:
        manifest, _ = (
            load_active_serving_manifest(
                models_path=models_path
            )
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None

    return manifest.created_at_utc


def _scheduled_retraining_due(
    last_training_at_utc: str | None,
    *,
    evaluated_at: pd.Timestamp,
    interval_hours: int,
) -> tuple[bool, float | None]:
    """
    Determine whether the regular model refresh interval elapsed.

    Missing or invalid timestamps do not force training.
    """

    if not last_training_at_utc:
        return False, None

    parsed = pd.to_datetime(
        last_training_at_utc,
        utc=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return False, None

    elapsed = evaluated_at - parsed

    # Protect against incorrectly future-dated state.
    if elapsed < pd.Timedelta(0):
        return False, None

    elapsed_hours = (
        elapsed.total_seconds() / 3600.0
    )

    return (
        elapsed
        >= pd.Timedelta(
            hours=interval_hours
        ),
        elapsed_hours / 24.0,
    )


def collect_retraining_signals(
    *,
    evaluated_at: datetime
    | pd.Timestamp
    | None = None,
) -> RetrainingSignals:
    """
    Collect normalized retraining signals from local or GCS storage.

    This function performs no training and no model promotion.
    """

    settings = get_retraining_settings()
    evaluation_time = _utc_timestamp(
        evaluated_at
    )

    raw_path = get_path("raw_data")
    monitoring_path = get_path("monitoring")
    models_path = get_path("models")

    retraining_state = (
        load_retraining_state()
    )

    last_training_at_utc = (
        _resolve_last_training_at_utc(
            retraining_state,
            models_path=models_path,
        )
    )

    if evaluated_at is not None:
        simulated_retrained_at = (
            retraining_state.get(
                "simulated_retrained_at_utc"
            )
        )

        if simulated_retrained_at:
            last_training_at_utc = str(
                simulated_retrained_at
            )

    processed_batch_ids = set(
        retraining_state.get(
            "processed_batch_ids",
            [],
        )
    )

    batch_pattern = join_uri(
        raw_path,
        "new_batches",
        "train_batch_churn*.csv",
    )
    batch_files = _list_files(batch_pattern)

    (
        new_training_rows,
        dataset_version,
        batch_ids,
        data_quality_ok,
        data_quality_reason,
    ) = _read_ground_truth_batches(
        batch_files,
        processed_batch_ids=(
            processed_batch_ids
        ),
    )

    drift_path = join_uri(
        monitoring_path,
        "feature_drift_history.parquet",
    )
    drift_history = _read_parquet_if_available(
        drift_path
    )

    drift_settings = settings["drift"]
    drift_result = (
        evaluate_persistent_feature_drift(
            drift_history,
            evaluated_at=evaluation_time,
            lookback_days=drift_settings[
                "lookback_days"
            ],
            consecutive_windows=drift_settings[
                "consecutive_windows"
            ],
        )
    )

    performance_path = join_uri(
        monitoring_path,
        "churn_performance_history.parquet",
    )
    performance_history = (
        _read_parquet_if_available(
            performance_path
        )
    )

    performance_settings = settings[
        "performance"
    ]
    performance_result = (
        evaluate_performance_degradation(
            performance_history,
            consecutive_windows=(
                performance_settings[
                    "consecutive_windows"
                ]
            ),
            minimum_samples=(
                performance_settings[
                    "minimum_samples"
                ]
            ),
            min_f1=performance_settings[
                "min_f1"
            ],
            min_recall=(
                performance_settings[
                    "min_recall"
                ]
            ),
            min_roc_auc=(
                performance_settings[
                    "min_roc_auc"
                ]
            ),
            max_brier_score=(
                performance_settings[
                    "max_brier_score"
                ]
            ),
        )
    )

    effective_training_state = {
        **retraining_state,
        "last_retrained_at_utc": (
            last_training_at_utc
        ),
    }

    cooldown_active = _cooldown_active(
        effective_training_state,
        evaluated_at=evaluation_time,
        cooldown_hours=settings[
            "cooldown_hours"
        ],
    )

    (
        scheduled_retraining_due,
        days_since_last_training,
    ) = _scheduled_retraining_due(
        last_training_at_utc,
        evaluated_at=evaluation_time,
        interval_hours=settings[
            "scheduled_interval_hours"
        ],
    )

    maximum_rows = settings[
        "maximum_new_training_rows"
    ]
    budget_available = (
        new_training_rows <= maximum_rows
    )

    return RetrainingSignals(
        dataset_version=dataset_version,
        new_training_rows=new_training_rows,
        minimum_training_rows=settings[
            "minimum_new_training_rows"
        ],
        data_quality_ok=data_quality_ok,
        performance_degraded=(
            performance_result.triggered
        ),
        feature_drift_persistent=(
            drift_result.triggered
        ),
        cooldown_active=cooldown_active,
        budget_available=budget_available,
        performance_window_end=(
            performance_result.window_end
        ),
        drift_window_end=(
            drift_result.window_end
        ),
        performance_reason=(
            performance_result.reason
        ),
        drift_reason=drift_result.reason,
        data_quality_reason=(
            data_quality_reason
        ),
        batch_ids=batch_ids,
        scheduled_retraining_due=(
            scheduled_retraining_due
        ),
        days_since_last_training=(
            days_since_last_training
        ),
    )