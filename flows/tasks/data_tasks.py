# --- STANDARD LIBRARY IMPORTS ---
import shutil

from datetime import datetime

# --- THIRD PARTY IMPORTS ---
import mlflow

from google.cloud import storage

# --- INTERNAL CONFIG BOOTSTRAP ---
from src.configs.loader import load_config, get_path, file_exists, ensure_dir

# Load config early so environment variables (Prefect, MLflow) are set
ENV_CFG = load_config()

# --- PREFECT IMPORTS (after config bootstrap) ---
# ruff: noqa: E402
from prefect import task, get_run_logger

# --- PROJECT IMPORTS ---
# ruff: noqa: E402

from src.utils.logger import get_logger

from src.data.raw.ingest import ingest
from src.data.features.build_features import run_feature_pipeline
from src.data.splits.split import split as split_logic
from src.data.versioning import make_dataset_version, snapshot_current_datasets, log_dataset_manifest_to_mlflow

from src.monitoring.feature_drift import run_feature_drift_check


logger = get_logger(__name__)


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


@task(name="Data Processing & Feature State Update")
def task_prepare_data(is_drift_run: bool):
    """Run ingestion, feature generation and dataset splitting for training."""
    p_logger = get_run_logger()
    p_logger.info(f"Starting data preparation (Emergency Mode: {is_drift_run})")
    ingest()
    run_feature_pipeline()
    p_logger.info("Skipping feature state snapshot (not required for this setup).")
    split_logic()

@task(name="Snapshot Dataset Version")
def task_snapshot_dataset():
    """
    Create an immutable snapshot of the prepared training datasets.

    Returns:
        The dataset manifest containing version and snapshot metadata.
    """
    p_logger = get_run_logger()
    version_id = make_dataset_version()
    manifest = snapshot_current_datasets(version_id)
    p_logger.info(f"Dataset snapshot created: {version_id}")
    return manifest

@task(name="Log Dataset Metadata")
def task_log_dataset_metadata(run_id: str, dataset_manifest: dict):
    """
    Attach dataset lineage metadata to an existing MLflow run.

    Notes:
        Logging failures are reported as warnings and do not abort the flow.
    """
    p_logger = get_run_logger()
    try:
        with mlflow.start_run(run_id=run_id):
            log_dataset_manifest_to_mlflow(dataset_manifest)
    except Exception as e:
        p_logger.warning(f"Could not log dataset metadata: {e}")


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


