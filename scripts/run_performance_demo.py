from __future__ import annotations

from datetime import datetime, timezone
import argparse
import fsspec
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.configs.loader import ensure_dir, file_exists, get_path, load_config
from src.monitoring.config import get_business_settings
from src.monitoring.performance import compute_business_metrics
from src.utils.logger import get_logger


logger = get_logger(__name__)
PREDICTIONS_PATH = get_path("predictions")
RAW_DATA_PATH = get_path("raw_data")
MONITORING_PATH = get_path("monitoring")

INFERENCE_LOG_FILE = f"{PREDICTIONS_PATH}/inference_log.parquet"
GROUND_TRUTH_BATCH_DIR = f"{MONITORING_PATH}/ground_truth_batches"
PERFORMANCE_HISTORY_FILE = f"{MONITORING_PATH}/churn_performance_history.parquet"
CUMULATIVE_GT_FILE = f"{MONITORING_PATH}/cumulative_ground_truth.csv"


def list_files(directory: str, pattern: str) -> list[str]:
    """
    List files from a local directory or GCS path using the same pattern.
    """
    glob_pattern = f"{directory}/{pattern}"
    fs, fs_pattern = fsspec.core.url_to_fs(glob_pattern)

    files = fs.glob(fs_pattern)

    if directory.startswith("gs://"):
        return sorted(path if path.startswith("gs://") else f"gs://{path}" for path in files)

    return sorted(str(path) for path in files)


def load_champion_decision_threshold(default: float = 0.5) -> float:
    """Load the decision threshold from the current champion model run."""
    model_name = load_config()["model"]["registry_name"]
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        run = client.get_run(mv.run_id)
        return float(run.data.params.get("decision_threshold", default))
    except Exception as exc:
        logger.warning(
            "Could not load champion decision threshold. Falling back to %.2f. Reason: %s",
            default,
            exc,
        )
        return default


def normalize_customer_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize customer id column names to `customerid`.
    """
    df = df.copy()

    if "customerID" in df.columns and "customerid" not in df.columns:
        df = df.rename(columns={"customerID": "customerid"})

    return df


def load_predictions() -> pd.DataFrame:
    if not file_exists(INFERENCE_LOG_FILE):
        raise FileNotFoundError(f"Prediction log not found: {INFERENCE_LOG_FILE}")

    df = pd.read_parquet(INFERENCE_LOG_FILE)
    df = normalize_customer_id_column(df)

    if "prediction_id" not in df.columns:
        raise KeyError("Prediction log must contain `prediction_id`.")

    if "prediction" not in df.columns:
        raise KeyError("Prediction log must contain probability column `prediction`.")

    decision_threshold = load_champion_decision_threshold()

    df["churn_probability"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["churn_prediction"] = (df["churn_probability"] >= decision_threshold).astype(int)
    df["decision_threshold"] = decision_threshold

    if "expected_value" in df.columns:
        df["expected_value"] = pd.to_numeric(df["expected_value"], errors="coerce").fillna(0.0)
    else:
        df["expected_value"] = 0.0

    df["expected_profit"] = df["expected_value"]

    if "action" not in df.columns:
        df["action"] = "no_action"

    df["is_actioned"] = df["action"].ne("no_action")

    if "prediction_timestamp" in df.columns:
        df["prediction_timestamp"] = pd.to_datetime(
            df["prediction_timestamp"],
            errors="coerce",
            utc=True,
        )

    return df


def build_cumulative_ground_truth() -> pd.DataFrame:
    batch_files = list_files(GROUND_TRUTH_BATCH_DIR, "ground_truth_churn_*.csv")

    if not batch_files:
        raise FileNotFoundError(
            f"No churn ground truth batches found in {GROUND_TRUTH_BATCH_DIR}. "
            "Run `python scripts/release_churn_labels.py --simulation-day <day>` first."
        )

    frames = [pd.read_csv(path) for path in batch_files]
    gt = pd.concat(frames, ignore_index=True)
    gt = normalize_customer_id_column(gt)

    if "prediction_id" not in gt.columns:
        raise KeyError("Ground truth batches must contain `prediction_id`.")

    if "churn" not in gt.columns:
        raise KeyError("Ground truth batches must contain `churn`.")

    gt["churn"] = pd.to_numeric(gt["churn"], errors="coerce")
    gt = gt.dropna(subset=["prediction_id", "churn"])
    gt["churn"] = gt["churn"].astype(int)

    if "label_available_at" in gt.columns:
        gt["label_available_at"] = pd.to_datetime(
            gt["label_available_at"],
            errors="coerce",
            utc=True,
        )
        gt = gt.sort_values("label_available_at")

    gt = gt.drop_duplicates(subset=["prediction_id"], keep="last")

    ensure_dir(MONITORING_PATH)
    gt.to_csv(CUMULATIVE_GT_FILE, index=False)

    logger.info("✅ Cumulative ground truth written to: %s", CUMULATIVE_GT_FILE)
    logger.info("Rows: %s", len(gt))

    return gt


def join_predictions_with_ground_truth(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join predictions with ground truth labels using `prediction_id`.
    """
    joined = predictions.merge(
        ground_truth[["prediction_id", "churn"]],
        on="prediction_id",
        how="inner",
    )

    if joined.empty:
        raise ValueError("No matching rows between predictions and ground truth.")

    return joined

def parse_evaluated_at(
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

def compute_churn_metrics(
    joined: pd.DataFrame,
    decision_threshold: float,
    *,
    simulation_day: int | None = None,
    evaluated_at: datetime | None = None,
) -> dict:
    """
    Compute churn classification, calibration, and business metrics.
    """
    y_true = joined["churn"].astype(int)
    y_pred = joined["churn_prediction"].astype(int)
    y_prob = joined["churn_probability"].astype(float)

    window_start = None
    window_end = None

    if "prediction_timestamp" in joined.columns:
        ts = pd.to_datetime(joined["prediction_timestamp"], errors="coerce", utc=True)
        ts = ts.dropna()

        if not ts.empty:
            window_start = ts.min().isoformat()
            window_end = ts.max().isoformat()

    evaluation_time = (
        evaluated_at
        or datetime.now(timezone.utc)
    )
    metrics = {
        "timestamp": (
            evaluation_time.isoformat()
        ),
        "simulation_day": simulation_day,
        "window_start": window_start,
        "window_end": window_end,
        "n_samples": int(len(joined)),
        "decision_threshold": float(decision_threshold),
        "churn_rate": float(y_true.mean()),
        "avg_churn_probability": float(y_prob.mean()),
        "high_risk_share": float((y_prob >= decision_threshold).mean()),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_profit": float(joined["expected_profit"].sum()),
        "performance_threshold_breached": False,
        "retrain_triggered": False,
        "retraining_executed": False,
        "champion_promoted": False,
    }

    business_cfg = get_business_settings()

    business_metrics = compute_business_metrics(
        joined,
        y_true_col="churn",
        y_proba_col="churn_probability",
        action_col="action",
        customer_value=business_cfg["customer_value"],
        cost_contact=business_cfg["cost_contact"],
        cost_discount=business_cfg["cost_discount"],
        contact_uplift=business_cfg["contact_uplift"],
        discount_uplift=business_cfg["discount_uplift"],
    )

    metrics.update({f"business_{k}": v for k, v in business_metrics.items()})

    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def should_retrain(metrics: dict) -> tuple[bool, str]:
    """
    Decide whether retraining should be triggered based on churn thresholds.
    """
    if metrics["n_samples"] < 20:
        return False, "Not enough labeled samples yet."

    if metrics["f1_score"] < 0.60:
        return True, f"F1 below threshold: {metrics['f1_score']:.3f}"

    if metrics["recall"] < 0.65:
        return True, f"Recall below threshold: {metrics['recall']:.3f}"

    if pd.notna(metrics["roc_auc"]) and metrics["roc_auc"] < 0.75:
        return True, f"ROC AUC below threshold: {metrics['roc_auc']:.3f}"

    if metrics["brier_score"] > 0.22:
        return True, f"Brier score above threshold: {metrics['brier_score']:.3f}"

    return False, "Model performance is within thresholds."


def append_performance_history(metrics: dict) -> pd.DataFrame:
    ensure_dir(MONITORING_PATH)

    new_row = pd.DataFrame([metrics])

    if file_exists(PERFORMANCE_HISTORY_FILE):
        history = pd.read_parquet(PERFORMANCE_HISTORY_FILE)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history.to_parquet(PERFORMANCE_HISTORY_FILE, index=False)

    logger.info("✅ Performance history written to: %s", PERFORMANCE_HISTORY_FILE)

    return history


def main(
    *,
    simulation_day: int | None = None,
    evaluated_at: datetime | None = None,
) -> None:
    """
    Run one churn performance evaluation cycle.
    """
    logger.info("📊 Starting churn performance evaluation")

    predictions = load_predictions()
    ground_truth = build_cumulative_ground_truth()
    joined = join_predictions_with_ground_truth(predictions, ground_truth)

    decision_threshold = load_champion_decision_threshold()
    metrics = compute_churn_metrics(
        joined,
        decision_threshold,
        simulation_day=simulation_day,
        evaluated_at=evaluated_at,
    )
    retrain_needed, retrain_reason = should_retrain(metrics)
    metrics[
        "performance_threshold_breached"
    ] = retrain_needed
    metrics["retrain_triggered"] = retrain_needed
    metrics["retrain_reason"] = retrain_reason

    history = append_performance_history(metrics)

    logger.info("✅ Latest churn metrics")
    logger.info("Samples: %s", metrics["n_samples"])
    logger.info("Churn rate: %.2f%%", metrics["churn_rate"] * 100)
    logger.info("F1: %.3f", metrics["f1_score"])
    logger.info("Recall: %.3f", metrics["recall"])
    logger.info("Precision: %.3f", metrics["precision"])
    logger.info("ROC AUC: %.3f", metrics["roc_auc"])
    logger.info("Brier: %.3f", metrics["brier_score"])
    logger.info("Retrain: %s | %s", retrain_needed, retrain_reason)

    print("CHURN_PERFORMANCE_RESULT=")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"\nHistory rows: {len(history)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simulation-day",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--evaluated-at",
        type=parse_evaluated_at,
        default=None,
    )

    args = parser.parse_args()

    main(
        simulation_day=args.simulation_day,
        evaluated_at=args.evaluated_at,
    )