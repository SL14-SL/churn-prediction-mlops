from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)

from src.configs.loader import load_config
from src.training.dataset import (
    load_and_prepare_validation_data,
)
from src.training.evaluate_metrics import (
    predict_with_threshold,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

CFG = load_config()
MODEL_NAME = CFG["model"]["registry_name"]


def _generate_and_log_plots(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    run_id: str,
    threshold: float = 0.5,
) -> None:
    """
    Generate and log evaluation plots for an MLflow run.

    Currently, the function logs a confusion matrix calculated with the
    supplied decision threshold.
    """
    logger.info(
        "Generating evaluation plots | "
        "run_id=%s | threshold=%s",
        run_id,
        threshold,
    )

    predictions = predict_with_threshold(
        model,
        X_val,
        threshold,
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
    )

    with TemporaryDirectory() as temporary_directory:
        plot_path = (
            Path(temporary_directory)
            / "confusion_matrix.png"
        )

        figure, axis = plt.subplots(
            figsize=(
                8,
                6,
            )
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "No Churn",
                "Churn",
            ],
            yticklabels=[
                "No Churn",
                "Churn",
            ],
            ax=axis,
        )

        axis.set_xlabel(
            "Predicted"
        )
        axis.set_ylabel(
            "Actual"
        )
        axis.set_title(
            "Confusion Matrix"
        )

        figure.tight_layout()
        figure.savefig(
            plot_path
        )
        plt.close(
            figure
        )

        with mlflow.start_run(
            run_id=run_id,
            nested=True,
        ):
            mlflow.log_artifact(
                str(plot_path),
                "evaluation/plots",
            )

    logger.info(
        "Confusion matrix logged | "
        "run_id=%s",
        run_id,
    )


def evaluate_model(
    model_alias: str = "champion",
) -> float | None:
    """
    Evaluate one registry model on the current validation dataset.

    Args:
        model_alias: MLflow registry alias of the model to evaluate.

    Returns:
        The model F1 score, or None when the model cannot be evaluated.
    """
    X_val, y_val = (
        load_and_prepare_validation_data()
    )

    model_uri = (
        f"models:/{MODEL_NAME}"
        f"@{model_alias}"
    )

    logger.info(
        "Evaluating registry model | "
        "alias=%s | uri=%s",
        model_alias,
        model_uri,
    )

    try:
        model = mlflow.pyfunc.load_model(
            model_uri
        )

        predictions = predict_with_threshold(
            model,
            X_val,
            threshold=0.5,
        )

        score = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

    except Exception as error:
        logger.warning(
            "Registry model evaluation failed | "
            "alias=%s | reason=%s",
            model_alias,
            error,
        )
        return None

    logger.info(
        "Registry model evaluated | "
        "alias=%s | f1=%.4f",
        model_alias,
        score,
    )

    return float(
        score
    )