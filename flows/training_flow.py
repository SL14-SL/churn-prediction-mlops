# --- STANDARD LIBRARY IMPORTS ---
import sys
import logging
import warnings
import json

# --- THIRD PARTY IMPORTS ---
import mlflow

# --- INTERNAL CONFIG BOOTSTRAP ---
from src.configs.loader import load_config

# Load config early so environment variables (Prefect, MLflow) are set
ENV_CFG = load_config()

# --- PREFECT IMPORTS (after config bootstrap) ---
# ruff: noqa: E402
from prefect import flow, get_run_logger

# --- PROJECT IMPORTS ---
# ruff: noqa: E402

from src.training.register import champion_exists
from src.training.policy import should_refresh_api, should_skip_training, get_run_strategy

from src.utils.logger import get_logger

from flows.deployment_flow import deploy_and_verify_release
from flows.tasks.serving_tasks import (
    task_resolve_previous_release,
    task_publish_serving_release,
)
from flows.tasks.data_tasks import (
    task_check_drift,
    task_log_dataset_metadata,
    task_prepare_data,
    task_snapshot_dataset,
)
from flows.tasks.training_tasks import task_train
from flows.tasks.registry_tasks import (
    task_eval_and_reg,
    task_evaluate_champion,
    task_bootstrap_champion,
)


# --- INITIALIZE CONFIGURATION ---
GCP_CFG = load_config("gcp.yaml")
TRAIN_CFG = load_config("training.yaml")
MODEL_NAME = ENV_CFG["model"]["registry_name"]
logger = get_logger(__name__)

# --- LOGGING SETUP ---
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

tracking_uri = ENV_CFG["tracking"]["mlflow_tracking_uri"]
mlflow.set_tracking_uri(tracking_uri)
logger.info(f"Using MLflow tracking URI: {tracking_uri}")


@flow(name="End-to-End Churn Pipeline")
def training_pipeline(force_run: bool = False, bootstrap: bool = False):
        
    if bootstrap and champion_exists():
        raise RuntimeError(
            "Bootstrap rejected: a Champion "
            "already exists. Use the regular "
            "forced training flow instead."
        )

    p_logger = get_run_logger()
    p_logger.info(f"Starting Pipeline (Env: {ENV_CFG['environment']})")
    
    drift_detected = task_check_drift()

    if should_skip_training(drift_detected, force_run):
        p_logger.info("System stable. Only evaluating current champion.")
        task_evaluate_champion()
        return

    strategy = get_run_strategy(drift_detected, force_run)
    print(f"[{strategy}] mode activated.")
    
    task_prepare_data(is_drift_run=drift_detected)
    dataset_manifest = task_snapshot_dataset()
    run_id = task_train()
    task_log_dataset_metadata(run_id, dataset_manifest)

    #task_archive_logs() 

    if bootstrap:
        registration_result = (
            task_bootstrap_champion(
                candidate_run_id=run_id,
            )
        )
    else:
        registration_result = (
            task_eval_and_reg(
                run_id
            )
        )

    release_manifest = None
    deployment_result = None
    previous_release_id = None

    if should_refresh_api(
        registration_result["promoted"]
    ):
        previous_release_id = (
            task_resolve_previous_release()
        )

        p_logger.info(
            "New Champion detected. "
            "Publishing serving release | "
            "previous_release_id=%s",
            previous_release_id,
        )

        release_manifest = (
            task_publish_serving_release(
                registration_result=(
                    registration_result
                ),
                dataset_manifest=(
                    dataset_manifest
                ),
            )
        )

        release_id = release_manifest[
            "release_id"
        ]

        p_logger.info(
            "Serving release published. "
            "Starting verified deployment | "
            "release_id=%s",
            release_id,
        )

        deployment_result = (
            deploy_and_verify_release(
                release_id=release_id,
                previous_release_id=(
                    previous_release_id
                ),
            )
        )
    else:
        p_logger.info(
            "No deployment needed. "
            "Current Champion remains active."
        )
    p_logger.info("Pipeline execution finished successfully.")

    return {
        "run_id": run_id,
        "candidate_run_id": run_id,
        "champion_promoted": bool(
            registration_result[
                "promoted"
            ]
        ),
        "model_version": (
            registration_result[
                "model_version"
            ]
        ),
        "serving_release_id": (
            release_manifest[
                "release_id"
            ]
            if release_manifest
            else None
        ),
        "previous_release_id": (
            previous_release_id
        ),
        "deployment_status": (
            deployment_result[
                "deployment_status"
            ]
            if deployment_result
            else None
        ),
    }

if __name__ == "__main__":
    force = "--force" in sys.argv
    bootstrap = "--bootstrap" in sys.argv

    result = training_pipeline(
        force_run=force,
        bootstrap=bootstrap,
    )

    print(
        "TRAINING_RESULT_JSON="
        + json.dumps(result)
    )
    