from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalEvaluation:
    """Normalized monitoring-signal result consumed by retraining policy."""
    triggered: bool
    window_end: str | None
    reason: str


def evaluate_persistent_feature_drift(
    history: pd.DataFrame,
    *,
    evaluated_at: pd.Timestamp,
    lookback_days: int,
    consecutive_windows: int,
) -> SignalEvaluation:
    """
    Evaluate drift only from recent complete drift checks.

    A drift signal is persistent when at least one feature is marked
    as drifted in every one of the latest N evaluation windows.
    """

    if history.empty:
        return SignalEvaluation(
            triggered=False,
            window_end=None,
            reason="No feature drift history available.",
        )
    
    required_columns = {
        "timestamp",
        "feature",
        "drift_detected",
    }

    missing_columns = (
        required_columns - set(history.columns)
    )
    if missing_columns:
        return SignalEvaluation(
            triggered=False,
            window_end=None,
            reason=(
                "Feature drift history is missing columns: "
                f"{sorted(missing_columns)}."
            ),
        )

    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    frame["drift_detected"] = (
        frame["drift_detected"]
        .fillna(False)
        .astype(bool)
    )
    frame = frame.dropna(
        subset=["timestamp", "feature"]
    )

    evaluation_time = pd.Timestamp(evaluated_at)
    if evaluation_time.tzinfo is None:
        evaluation_time = evaluation_time.tz_localize(
            "UTC"
        )
    else:
        evaluation_time = evaluation_time.tz_convert(
            "UTC"
        )

    cutoff = evaluation_time - pd.Timedelta(
        days=lookback_days
    )

    frame = frame[
        (frame["timestamp"] >= cutoff)
        & (frame["timestamp"] <= evaluation_time)
    ]

    window_timestamps = (
        frame["timestamp"]
        .drop_duplicates()
        .sort_values()
        .tail(consecutive_windows)
        .tolist()
    )

    if len(window_timestamps) < consecutive_windows:
        return SignalEvaluation(
            triggered=False,
            window_end=(
                window_timestamps[-1].isoformat()
                if window_timestamps
                else None
            ),
            reason=(
                "Not enough recent feature drift windows: "
                f"{len(window_timestamps)}/"
                f"{consecutive_windows}."
            ),
        )

    recent = frame[
        frame["timestamp"].isin(
            window_timestamps
        )
    ]

    drift_by_feature = (
        recent.groupby("feature")[
            "drift_detected"
        ]
        .agg(["sum", "count"])
    )

    persistent_features = drift_by_feature[
        (
            drift_by_feature["count"]
            == consecutive_windows
        )
        & (
            drift_by_feature["sum"]
            == consecutive_windows
        )
    ].index.tolist()

    latest_window = pd.Timestamp(
        window_timestamps[-1]
    ).isoformat()

    if not persistent_features:
        return SignalEvaluation(
            triggered=False,
            window_end=latest_window,
            reason=(
                "No feature drift persisted across "
                f"{consecutive_windows} windows."
            ),
        )

    return SignalEvaluation(
        triggered=True,
        window_end=latest_window,
        reason=(
            "Persistent feature drift detected for: "
            f"{sorted(persistent_features)}."
        ),
    )


def evaluate_performance_degradation(
    history: pd.DataFrame,
    *,
    consecutive_windows: int,
    minimum_samples: int,
    min_f1: float,
    min_recall: float,
    min_roc_auc: float,
    max_brier_score: float,
) -> SignalEvaluation:
    """
    Detect persistent churn-model degradation.

    A window is degraded when at least one monitored classification
    metric breaches its configured threshold. Retraining is triggered
    only when every one of the latest N complete windows is degraded.
    """
    if history.empty:
        return SignalEvaluation(
            triggered=False,
            window_end=None,
            reason=(
                "No performance history "
                "available."
            ),
        )

    timestamp_column = (
        "computed_at"
        if "computed_at" in history.columns
        else "timestamp"
    )

    f1_column = (
        "f1"
        if "f1" in history.columns
        else "f1_score"
    )

    required_columns = {
        timestamp_column,
        "n_samples",
        f1_column,
        "recall",
        "roc_auc",
        "brier_score",
    }

    missing_columns = (
        required_columns
        - set(history.columns)
    )

    if missing_columns:
        return SignalEvaluation(
            triggered=False,
            window_end=None,
            reason=(
                "Performance history is "
                "missing columns: "
                f"{sorted(missing_columns)}."
            ),
        )

    frame = history.copy()

    frame[timestamp_column] = (
        pd.to_datetime(
            frame[timestamp_column],
            utc=True,
            errors="coerce",
        )
    )

    numeric_columns = [
        "n_samples",
        f1_column,
        "recall",
        "roc_auc",
        "brier_score",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            timestamp_column,
            *numeric_columns,
        ]
    )

    frame = frame.loc[
        frame["n_samples"]
        >= minimum_samples
    ]

    frame = (
        frame.sort_values(
            timestamp_column
        )
        .drop_duplicates(
            subset=[timestamp_column],
            keep="last",
        )
        .tail(
            consecutive_windows
        )
    )

    if len(frame) < consecutive_windows:
        window_end = (
            frame[timestamp_column]
            .iloc[-1]
            .isoformat()
            if not frame.empty
            else None
        )

        return SignalEvaluation(
            triggered=False,
            window_end=window_end,
            reason=(
                "Not enough complete performance "
                "windows with sufficient samples: "
                f"{len(frame)}/"
                f"{consecutive_windows}."
            ),
        )

    degraded = (
        (frame[f1_column] < min_f1)
        | (
            frame["recall"]
            < min_recall
        )
        | (
            frame["roc_auc"]
            < min_roc_auc
        )
        | (
            frame["brier_score"]
            > max_brier_score
        )
    )

    latest_window = (
        frame[timestamp_column]
        .iloc[-1]
        .isoformat()
    )

    degraded_count = int(
        degraded.sum()
    )

    if not bool(
        degraded.all()
    ):
        return SignalEvaluation(
            triggered=False,
            window_end=latest_window,
            reason=(
                "Performance degradation did "
                "not persist across all recent "
                "windows: "
                f"{degraded_count}/"
                f"{consecutive_windows}."
            ),
        )

    breached_metrics: list[str] = []

    if bool(
        (
            frame[f1_column]
            < min_f1
        ).any()
    ):
        breached_metrics.append(
            "f1"
        )

    if bool(
        (
            frame["recall"]
            < min_recall
        ).any()
    ):
        breached_metrics.append(
            "recall"
        )

    if bool(
        (
            frame["roc_auc"]
            < min_roc_auc
        ).any()
    ):
        breached_metrics.append(
            "roc_auc"
        )

    if bool(
        (
            frame["brier_score"]
            > max_brier_score
        ).any()
    ):
        breached_metrics.append(
            "brier_score"
        )

    return SignalEvaluation(
        triggered=True,
        window_end=latest_window,
        reason=(
            "Persistent churn-model "
            "performance degradation "
            "detected across "
            f"{consecutive_windows} windows | "
            "breached_metrics="
            f"{sorted(breached_metrics)}."
        ),
    )