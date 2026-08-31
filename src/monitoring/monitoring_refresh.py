from __future__ import annotations

from dataclasses import dataclass

import fsspec
import pandas as pd

from src.configs.loader import (
    get_path,
)
from src.configs.paths import (
    join_uri,
)
from src.monitoring.config import (
    get_monitoring_config,
)
from src.monitoring.feature_drift import (
    run_feature_drift_check,
)
from src.monitoring.performance import (
    compute_classification_metrics,
    save_table,
)



@dataclass(frozen=True)
class MonitoringRefreshResult:
    ground_truth_rows: int
    performance_updated: bool
    performance_rows: int
    feature_drift_updated: bool
    feature_drift_rows: int
    performance_reason: str


def _list_files(
    pattern: str,
) -> list[str]:
    filesystem, fs_pattern = (
        fsspec.core.url_to_fs(
            pattern
        )
    )

    return sorted(
        filesystem.unstrip_protocol(path)
        for path in filesystem.glob(
            fs_pattern
        )
    )


def rebuild_cumulative_ground_truth(
    batch_files: list[str],
    *,
    output_path: str,
) -> pd.DataFrame:
    """
    Rebuild cumulative labeled churn data idempotently from all batches.
    """
    if not batch_files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for batch_path in batch_files:
        with fsspec.open(
            batch_path,
            "rb",
        ) as file:
            frames.append(
                pd.read_csv(file)
            )

    cumulative = pd.concat(
        frames,
        ignore_index=True,
    )

    if "prediction_id" in cumulative.columns:
        cumulative = (
            cumulative.drop_duplicates(
                subset=["prediction_id"],
                keep="last",
            )
        )
    elif {
        "label_batch_id",
        "customerID",
    }.issubset(cumulative.columns):
        cumulative = (
            cumulative.drop_duplicates(
                subset=[
                    "label_batch_id",
                    "customerID",
                ],
                keep="last",
            )
        )
    elif {
        "label_batch_id",
        "customerid",
    }.issubset(cumulative.columns):
        cumulative = (
            cumulative.drop_duplicates(
                subset=[
                    "label_batch_id",
                    "customerid",
                ],
                keep="last",
            )
        )
    else:
        cumulative = (
            cumulative.drop_duplicates(
                keep="last"
            )
        )

    cumulative = cumulative.reset_index(
        drop=True
    )

    with fsspec.open(
        output_path,
        "w",
    ) as file:
        cumulative.to_csv(
            file,
            index=False,
        )

    return cumulative


def build_performance_history(
    cumulative: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rebuild one performance window per released label batch.
    """
    if cumulative.empty:
        return pd.DataFrame()

    required_columns = {
        "label_batch_id",
        "Churn",
        "churn_probability",
    }

    missing_columns = (
        required_columns
        - set(cumulative.columns)
    )

    if missing_columns:
        raise ValueError(
            "Labeled churn data is missing "
            f"columns: {sorted(missing_columns)}."
        )

    monitoring_cfg = (
        get_monitoring_config()
    )

    threshold = float(
        monitoring_cfg.get(
            "performance",
            {},
        ).get(
            "threshold",
            0.5,
        )
    )

    rows: list[dict] = []

    for batch_id, batch_df in (
        cumulative.groupby(
            "label_batch_id",
            sort=True,
        )
    ):
        metrics = (
            compute_classification_metrics(
                batch_df,
                y_true_col="Churn",
                y_proba_col=(
                    "churn_probability"
                ),
                threshold=threshold,
            )
        )

        if (
            "label_available_at"
            in batch_df.columns
        ):
            computed_at = (
                pd.to_datetime(
                    batch_df[
                        "label_available_at"
                    ],
                    utc=True,
                    errors="coerce",
                ).max()
            )
        else:
            computed_at = pd.NaT

        if pd.isna(computed_at):
            computed_at = (
                pd.Timestamp.now(
                    tz="UTC"
                )
            )

        rows.append(
            {
                **metrics,
                "label_batch_id": str(
                    batch_id
                ),
                "computed_at": (
                    computed_at
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("computed_at")
        .reset_index(drop=True)
    )


def refresh_monitoring_signals() -> (
    MonitoringRefreshResult
):
    """
    Refresh delayed-label performance and feature-drift evidence.
    """
    monitoring_path = get_path(
        "monitoring"
    )

    batch_pattern = join_uri(
        monitoring_path,
        "ground_truth_batches",
        "ground_truth_churn_*.csv",
    )

    batch_files = _list_files(
        batch_pattern
    )

    cumulative_path = join_uri(
        monitoring_path,
        "cumulative_ground_truth.csv",
    )

    performance_path = join_uri(
        monitoring_path,
        "churn_performance_history.parquet",
    )

    cumulative = (
        rebuild_cumulative_ground_truth(
            batch_files,
            output_path=(
                cumulative_path
            ),
        )
    )

    performance_updated = False
    performance_rows = 0
    performance_reason = (
        "No labeled churn batches available."
    )

    if not cumulative.empty:
        try:
            performance_history = (
                build_performance_history(
                    cumulative
                )
            )

            if performance_history.empty:
                performance_reason = (
                    "No usable classification "
                    "performance windows."
                )
            else:
                save_table(
                    performance_history,
                    performance_path,
                )

                performance_updated = True
                performance_rows = len(
                    performance_history
                )
                performance_reason = (
                    "Churn performance history "
                    "refreshed."
                )

        except ValueError as error:
            performance_reason = str(
                error
            )

    drift_result = (
        run_feature_drift_check()
    )

    return MonitoringRefreshResult(
        ground_truth_rows=len(
            cumulative
        ),
        performance_updated=(
            performance_updated
        ),
        performance_rows=(
            performance_rows
        ),
        feature_drift_updated=(
            not drift_result.empty
        ),
        feature_drift_rows=len(
            drift_result
        ),
        performance_reason=(
            performance_reason
        ),
    )