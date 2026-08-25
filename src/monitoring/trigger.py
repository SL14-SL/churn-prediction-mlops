from __future__ import annotations
from datetime import datetime

import pandas as pd

from src.configs.loader import file_exists, get_path, load_config
from src.utils.logger import get_logger

from src.monitoring.retraining_policy import RetrainingDecision, decide_retraining
from src.monitoring.signal_collector import collect_retraining_signals
logger = get_logger(__name__)

MONITORING_PATH = get_path("monitoring")
PERFORMANCE_HISTORY_FILE = f"{MONITORING_PATH}/churn_performance_history.parquet"


def load_latest_performance_row() -> pd.Series | None:
    """Load the latest churn performance monitoring row."""
    if not file_exists(PERFORMANCE_HISTORY_FILE):
        logger.info("No churn performance history found: %s", PERFORMANCE_HISTORY_FILE)
        return None

    df = pd.read_parquet(PERFORMANCE_HISTORY_FILE)

    if df.empty:
        logger.info("Churn performance history is empty.")
        return None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )
        df = df.sort_values("timestamp")

    return df.iloc[-1]


def should_retrain() -> bool:
    """
    Decide whether the latest churn monitoring result should trigger retraining.

    Uses thresholds from configs/monitoring.yaml:
    - min_f1
    - min_recall
    - min_roc_auc
    - max_brier_score
    """
    cfg = load_config("monitoring.yaml")
    thresholds = cfg.get("performance", {}).get("retrain_thresholds", {})

    min_f1 = thresholds.get("min_f1", 0.60)
    min_recall = thresholds.get("min_recall", 0.65)
    min_roc_auc = thresholds.get("min_roc_auc", 0.75)
    max_brier_score = thresholds.get("max_brier_score", 0.22)

    latest = load_latest_performance_row()

    if latest is None:
        return False

    n_samples = int(latest.get("n_samples", 0) or 0)

    if n_samples < 20:
        logger.info("Skipping retrain: not enough labeled samples (%s/20).", n_samples)
        return False

    f1 = latest.get("f1_score")
    recall = latest.get("recall")
    roc_auc = latest.get("roc_auc")
    brier = latest.get("brier_score")

    if pd.notna(f1) and float(f1) < min_f1:
        logger.warning("Retrain triggered: F1 %.3f < %.3f", f1, min_f1)
        return True

    if pd.notna(recall) and float(recall) < min_recall:
        logger.warning("Retrain triggered: Recall %.3f < %.3f", recall, min_recall)
        return True

    if pd.notna(roc_auc) and float(roc_auc) < min_roc_auc:
        logger.warning("Retrain triggered: ROC AUC %.3f < %.3f", roc_auc, min_roc_auc)
        return True

    if pd.notna(brier) and float(brier) > max_brier_score:
        logger.warning("Retrain triggered: Brier %.3f > %.3f", brier, max_brier_score)
        return True

    retrain_flag = bool(latest.get("retrain_triggered", False))

    if retrain_flag:
        logger.warning("Retrain triggered by monitoring flag.")
        return True

    logger.info("No retraining needed. Latest churn metrics are within thresholds.")
    return False

def evaluate_retraining(
    *,
    evaluated_at: datetime
    | pd.Timestamp
    | None = None,
) -> RetrainingDecision:
    """
    Collect current signals and evaluate the guarded retraining policy.
    """
    signals = (
        collect_retraining_signals(
            evaluated_at=evaluated_at
        )
    )

    decision = decide_retraining(
        signals
    )

    logger.info(
        "Retraining decision evaluated | "
        "action=%s decision_id=%s "
        "triggers=%s reasons=%s",
        decision.action.value,
        decision.decision_id,
        list(decision.trigger_types),
        list(decision.reasons),
    )

    return decision 