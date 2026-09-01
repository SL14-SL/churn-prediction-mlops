
# --- THIRD PARTY IMPORTS ---
from mlflow.tracking import MlflowClient


# --- INTERNAL CONFIG BOOTSTRAP ---
from src.configs.loader import load_config

# Load config early so environment variables (Prefect, MLflow) are set
ENV_CFG = load_config()

# --- PREFECT IMPORTS (after config bootstrap) ---
# ruff: noqa: E402
from prefect import task, get_run_logger

# --- PROJECT IMPORTS ---
# ruff: noqa: E402

from src.training.register import register_model, champion_exists
from src.training.evaluate import evaluate_model
from src.training.model_comparison import compare_models

# --- INITIALIZE CONFIGURATION ---
GCP_CFG = load_config("gcp.yaml")
TRAIN_CFG = load_config("training.yaml")


@task(name="Evaluate Current Champion")
def task_evaluate_champion():
    """
    Evaluate the current Champion for monitoring continuity.

    Returns:
        Champion evaluation metrics, or None when evaluation cannot be completed.
    """
    p_logger = get_run_logger()
    p_logger.info("Evaluating current champion for dashboard continuity.")
    try:
        champion_f1 = evaluate_model(model_alias="champion")
        print(f"Champion F1: {champion_f1}")
        return champion_f1
    except Exception as e:
        p_logger.warning(f"Could not evaluate champion: {e}")
        return None

@task(name="Evaluation & Registration")
def task_eval_and_reg(
    new_run_id: str,
) -> dict:
    """
    Compare, register, and return immutable registry metadata.

    A promoted model version is later used to build the serving release.
    """
    p_logger = get_run_logger()

    is_better, metrics = (
        compare_models(
            new_run_id
        )
    )

    metrics = metrics or {}

    if "challenger_f1" in metrics:
        print(
            "Challenger F1: "
            f"{metrics['challenger_f1']}"
        )

    if "champion_f1" in metrics:
        print(
            "Champion F1: "
            f"{metrics['champion_f1']}"
        )

    alias = (
        "champion"
        if is_better
        else "challenger"
    )

    if is_better:
        p_logger.info(
            "Challenger (run=%s) outperforms "
            "Champion. Promoting.",
            new_run_id,
        )
    else:
        p_logger.info(
            "Champion remains superior. "
            "Registering new model as Challenger."
        )

    registered_version = register_model(
        new_run_id,
        alias=alias,
    )

    return {
        "promoted": bool(is_better),
        "alias": alias,
        "model_version": str(
            registered_version.version
        ),
        "model_run_id": new_run_id,
        "model_type": TRAIN_CFG[
            "model"
        ]["type"],
        "decision_threshold": float(
            metrics.get(
                "challenger_decision_threshold",
                0.5,
            )
        ),
        "metrics": metrics,
    }

@task(name="Bootstrap Initial Champion")
def task_bootstrap_champion(
    candidate_run_id: str,
) -> dict:
    """
    Create the first Champion in an empty model registry.

    Bootstrap is rejected when a Champion already exists.
    """
    p_logger = get_run_logger()

    if champion_exists():
        raise RuntimeError(
            "Bootstrap rejected: a Champion "
            "already exists."
        )

    p_logger.info(
        "No Champion exists. Starting explicit "
        "initial bootstrap | "
        f"candidate_run_id={candidate_run_id}"
    )

    run_data = (
        MlflowClient()
        .get_run(candidate_run_id)
        .data
    )

    decision_threshold = float(
        run_data.params.get(
            "decision_threshold",
            0.5,
        )
    )

    model_type = run_data.params.get(
        "model_type",
        TRAIN_CFG["model"]["type"],
    )

    # Check again immediately before changing
    # the registry state.
    if champion_exists():
        raise RuntimeError(
            "Bootstrap aborted: a Champion "
            "was created concurrently."
        )

    registered_version = register_model(
        candidate_run_id,
        alias="champion",
    )

    p_logger.info(
        "Initial Champion created | "
        f"run_id={candidate_run_id} | "
        "model_version="
        f"{registered_version.version}"
    )

    return {
        "promoted": True,
        "alias": "champion",
        "model_version": str(
            registered_version.version
        ),
        "model_run_id": candidate_run_id,
        "model_type": model_type,
        "decision_threshold": (
            decision_threshold
        ),
        "metrics": {},
    }
