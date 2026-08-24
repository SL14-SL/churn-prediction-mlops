# --- STANDARD LIBRARY IMPORTS ---
import sys
import time
import shutil
import logging
import warnings
import os
import hashlib
import json

from datetime import datetime

# --- THIRD PARTY IMPORTS ---
import requests
import mlflow

from google.cloud import storage

# --- INTERNAL CONFIG BOOTSTRAP ---
from src.configs.loader import load_config, get_path, file_exists, ensure_dir

# Load config early so environment variables (Prefect, MLflow) are set
ENV_CFG = load_config()

# --- PREFECT IMPORTS (after config bootstrap) ---
# ruff: noqa: E402
from prefect import flow, task, get_run_logger

# --- PROJECT IMPORTS ---
# ruff: noqa: E402
from src.data.raw.ingest import ingest
from src.data.features.build_features import run_feature_pipeline
from src.data.splits.split import split as split_logic
from src.data.versioning import make_dataset_version, snapshot_current_datasets, log_dataset_manifest_to_mlflow

from src.training.train import train
from src.training.register import register_model
from src.training.evaluate import compare_models, evaluate_model
from src.training.policy import should_refresh_api, should_skip_training, get_run_strategy

from src.monitoring.feature_drift import run_feature_drift_check

from src.deployment.prediction_probe import build_prediction_probe
from src.inference.releases.publisher import publish_serving_release

from src.utils.logger import get_logger

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

def build_config_hash(
    dataset_manifest: dict,
) -> str | None:
    """
    Build a deterministic hash of the effective training configuration.
    """
    effective_config = (
        dataset_manifest.get(
            "effective_config"
        )
    )

    if effective_config is None:
        return None

    serialized = json.dumps(
        effective_config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()



@task(name="Check Feature Drift")
def task_check_drift() -> bool:
    """
    Run churn feature drift monitoring and return whether drift was detected.
    """
    p_logger = get_run_logger()

    feature_drift_df = run_feature_drift_check()

    if feature_drift_df.empty:
        p_logger.info("Feature drift check returned no results.")
        print("Drift status: False")
        return False

    if "drift_detected" not in feature_drift_df.columns:
        p_logger.warning("Feature drift results do not contain drift_detected column.")
        print("Drift status: False")
        return False

    drifted_features = feature_drift_df.loc[
        feature_drift_df["drift_detected"], "feature"
    ].tolist()

    drift_detected = bool(feature_drift_df["drift_detected"].fillna(False).any())

    p_logger.info(
        "Feature drift check completed | "
        f"drift_detected={drift_detected} | "
        f"drifted_features={drifted_features}"
    )

    print(f"Drift status: {drift_detected}")
    return drift_detected

@task(name="Evaluate Current Champion")
def task_evaluate_champion():
    p_logger = get_run_logger()
    p_logger.info("Evaluating current champion for dashboard continuity.")
    try:
        champion_f1 = evaluate_model(model_alias="champion")
        print(f"Champion F1: {champion_f1}")
        return champion_f1
    except Exception as e:
        p_logger.warning(f"Could not evaluate champion: {e}")
        return None

@task(name="Data Processing & Feature State Update")
def task_prepare_data(is_drift_run: bool):
    p_logger = get_run_logger()
    p_logger.info(f"Starting data preparation (Emergency Mode: {is_drift_run})")
    ingest()
    run_feature_pipeline()
    p_logger.info("Skipping feature state snapshot (not required for this setup).")
    split_logic()

@task(name="Snapshot Dataset Version")
def task_snapshot_dataset():
    p_logger = get_run_logger()
    version_id = make_dataset_version()
    manifest = snapshot_current_datasets(version_id)
    p_logger.info(f"Dataset snapshot created: {version_id}")
    return manifest

@task(name="Log Dataset Metadata")
def task_log_dataset_metadata(run_id: str, dataset_manifest: dict):
    p_logger = get_run_logger()
    try:
        with mlflow.start_run(run_id=run_id):
            log_dataset_manifest_to_mlflow(dataset_manifest)
    except Exception as e:
        p_logger.warning(f"Could not log dataset metadata: {e}")

@task(name="Model Training")
def task_train():
    p_logger = get_run_logger()
    p_logger.info("Triggering model training task.")
    model, run_id = train()
    return run_id

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

@task(name="Publish Serving Release")
def task_publish_serving_release(
    *,
    registration_result: dict,
    dataset_manifest: dict,
):
    """
    Publish and activate the complete churn serving release.
    """
    p_logger = get_run_logger()

    if not registration_result.get(
        "promoted",
        False,
    ):
        raise ValueError(
            "Cannot publish a serving release "
            "for a non-promoted model."
        )

    models_path = get_path(
        "models"
    )
    validated_path = get_path(
        "validated_data"
    )

    feature_schema_source = (
        f"{models_path}/feature_schema.json"
    )
    validated_data_path = (
        f"{validated_path}/train.parquet"
    )

    prediction_probe = (
        build_prediction_probe(
            validated_data_path=(
                validated_data_path
            ),
        )
    )

    manifest = publish_serving_release(
        models_path=models_path,
        model_name=MODEL_NAME,
        model_version=(
            registration_result[
                "model_version"
            ]
        ),
        model_run_id=(
            registration_result[
                "model_run_id"
            ]
        ),
        model_type=(
            registration_result[
                "model_type"
            ]
        ),
        decision_threshold=(
            registration_result[
                "decision_threshold"
            ]
        ),
        dataset_version=(
            dataset_manifest.get(
                "dataset_version"
            )
        ),
        config_hash=build_config_hash(
            dataset_manifest
        ),
        git_commit=(
            dataset_manifest.get(
                "git_commit"
            )
        ),
        feature_schema_source=(
            feature_schema_source
        ),
        prediction_probe_payload=(
            prediction_probe
        ),
    )

    p_logger.info(
        "Serving release published: "
        "release_id=%s model_version=%s",
        manifest.release_id,
        manifest.model_version,
    )

    return manifest.to_dict()


@task(name="Archive Logs")
def task_archive_logs():
    """Archives logs. Handles local files and now also GCS blobs."""

    archived_count = 0
    try:
        p_logger = get_run_logger()
    except Exception:
        p_logger = logger

    PREDICTIONS_PATH = get_path("predictions")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- GCS ARCHIVING LOGIC ---
    if PREDICTIONS_PATH.startswith("gs://"):
        try:
            # Parse bucket and folder
            path_no_gs = PREDICTIONS_PATH.replace("gs://", "")
            bucket_name = path_no_gs.split("/")[0]
            source_folder = "/".join(path_no_gs.split("/")[1:])
            if source_folder and not source_folder.endswith("/"):
                source_folder += "/"
            
            archive_folder = f"{source_folder}archive/"
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=source_folder)
            
            archived_count = 0
            for blob in blobs:
                # Skip the directory placeholders and anything already in archive
                if blob.name == source_folder or "archive/" in blob.name:
                    continue
                
                filename = blob.name.split("/")[-1]
                new_blob_name = f"{archive_folder}{timestamp}_{filename}"
                
                # Move = Copy + Delete
                bucket.copy_blob(blob, bucket, new_blob_name)
                blob.delete()
                archived_count += 1
            
            p_logger.info(f"GCS: Successfully archived {archived_count} files to {archive_folder}")
        except Exception as e:
            p_logger.error(f"Failed to archive GCS logs: {e}")

    # --- LOCAL ARCHIVING LOGIC ---
    else:
        log_file = f"{PREDICTIONS_PATH}/inference_log.parquet"
        if file_exists(log_file):
            archive_dir = f"{PREDICTIONS_PATH}/archive"
            ensure_dir(archive_dir)
            target_path = f"{archive_dir}/inference_log_{timestamp}.parquet"
            shutil.move(log_file, target_path)
            p_logger.info(f"Local: Logs archived to: {target_path}")
        else:
            p_logger.info("Local: No log file found to archive.")
    
    return archived_count

@task(name="Refresh API")
def task_refresh_api() -> None:
    """
    Refresh the API model after a new champion has been promoted.

    Calls the API reload endpoint instead of restarting the container.
    """
    p_logger = get_run_logger()
    cfg = load_config()

    api_url = cfg.get("api", {}).get("url", "http://api:8080/predict")
    base_url = api_url.replace("/predict", "")
    reload_url = f"{base_url}/admin/reload-model"

    api_key = os.getenv("API_KEY")

    response = requests.post(
        reload_url,
        headers={"X-API-KEY": api_key},
        timeout=30,
    )

    response.raise_for_status()
    p_logger.info(f"API model reload successful: {response.json()}")

@task(name="Verify API Health")
def task_verify_health():
    p_logger = get_run_logger()

    api_url = ENV_CFG["api"]["url"].rsplit("/", 1)[0] + "/health"
    p_logger.info(f"Checking API health at: {api_url}")

    for i in range(20):  # 20 Versuche
        try:
            r = requests.get(api_url, timeout=10)

            if r.status_code == 200:
                p_logger.info("API is healthy.")
                return True

            p_logger.warning(f"Attempt {i+1}: status {r.status_code}")

        except requests.exceptions.RequestException as e:
            p_logger.warning(f"Attempt {i+1}: not reachable ({e})")

        time.sleep(10)

    raise Exception("API did not recover after refresh!")


@flow(name="End-to-End Churn Pipeline")
def training_pipeline(force_run: bool = False):
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

    registration_result = (
        task_eval_and_reg(
            run_id
        )
    )

    release_manifest = None

    if should_refresh_api(
        registration_result["promoted"]
    ):
        p_logger.info(
            "New Champion detected. "
            "Publishing serving release."
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

        p_logger.info(
            "Serving release published. "
            "Refreshing API."
        )

        task_refresh_api()
        task_verify_health()
    else:
        p_logger.info(
            "No API refresh needed. "
            "Current Champion remains active."
        )
    p_logger.info("Pipeline execution finished successfully.")

    return {
        "run_id": run_id,
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
    }

if __name__ == "__main__":
    force = "--force" in sys.argv
    import json

    result = training_pipeline(force_run=force)
    print("TRAINING_RESULT_JSON=" + json.dumps(result))
    