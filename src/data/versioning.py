from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

import fsspec
import mlflow

from src.configs.loader import load_config, get_path, file_exists, ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

CFG = load_config()
TRAIN_CFG = load_config("training.yaml")


def make_dataset_version() -> str:
    """Creates a UTC-based version id for one pipeline run."""
    return datetime.now(timezone.utc).strftime("ds_%Y%m%d_%H%M%S")


def _join(base: str, *parts: str) -> str:
    """Joins local or gs:// paths safely."""
    if base.startswith("gs://"):
        return "/".join([base.rstrip("/"), *[p.strip("/") for p in parts]])
    return os.path.join(base, *parts)


def _copy_file(src: str, dst: str):
    """Copies one file locally or via fsspec."""
    if not src:
        return
    if not file_exists(src):
        logger.warning(f"Versioning skipped: Source not found -> {src}")
        return

    if not dst.startswith("gs://"):
        ensure_dir(os.path.dirname(dst))

    with fsspec.open(src, "rb") as fsrc, fsspec.open(dst, "wb") as fdst:
        fdst.write(fsrc.read())

def get_git_commit() -> str | None:
    """
    Returns the current git commit hash.

    Priority:
    1. GIT_COMMIT_SHA environment variable, used in Docker/CI deployments.
    2. Local git command fallback, useful during local development.
    3. None if neither source is available.
    """
    env_commit = os.getenv("GIT_COMMIT_SHA")
    if env_commit:
        return env_commit.strip()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not determine git commit: {e}")
        return None
    

def get_active_config_name() -> str:
    """
    Returns the active environment config name.
    """
    environment = CFG.get("environment", "dev")
    return f"{environment}.yaml"

def get_versioning_sources() -> dict[str, str]:
    """
    Load dataset snapshot source paths from training.yaml.
    Keeps dataset versioning reusable across different ML projects.
    """
    sources_cfg = TRAIN_CFG.get("data", {}).get("versioning_sources", {})

    if not sources_cfg:
        raise ValueError("No data.versioning_sources configured in training.yaml.")

    return {
        name: _join(get_path(spec["path_key"]), spec["file"])
        for name, spec in sources_cfg.items()
    }

def build_snapshot_paths(version_id: str, source_paths: dict[str, str]) -> dict[str, str]:
    """
    Build versioned snapshot paths for configured dataset artifacts.
    """
    base = get_path("versioning")

    paths = {
        "base": _join(base, version_id),
        "manifest": _join(base, version_id, "manifest.json"),
        "latest_manifest": _join(base, "latest_manifest.json"),
    }

    for name, src in source_paths.items():
        filename = os.path.basename(src)
        paths[name] = _join(base, version_id, name, filename)

    return paths

def snapshot_current_datasets(version_id: str) -> dict:
    """
    Copy configured dataset sources into a versioned snapshot structure.
    Fully config-driven via training.yaml.
    """
    source_paths = get_versioning_sources()
    paths = build_snapshot_paths(version_id, source_paths)

    logger.info(f"📦 Creating dataset snapshot for version: {version_id}")

    for key, src in source_paths.items():
        _copy_file(src, paths[key])

    manifest = {
        "dataset_version": version_id,
        "environment": CFG.get("environment", "dev"),
        "config_name": get_active_config_name(),
        "git_commit": get_git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": CFG.get("random_seed"),

        "target_column": TRAIN_CFG.get("data", {}).get("target_column"),
        "id_columns": TRAIN_CFG.get("data", {}).get("id_columns", []),

        "effective_config": {
            "environment_config": CFG,
            "training_config": TRAIN_CFG,
        },

        "sources": source_paths,
        "snapshots": {key: paths[key] for key in source_paths},
    }

    if not paths["manifest"].startswith("gs://"):
        ensure_dir(os.path.dirname(paths["manifest"]))

    with fsspec.open(paths["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)

    with fsspec.open(paths["latest_manifest"], "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"✅ Dataset manifest written to: {paths['manifest']}")
    return manifest

def get_latest_dataset_manifest() -> dict:
    """Loads the latest dataset manifest."""
    manifest_path = _join(get_path("versioning"), "latest_manifest.json")

    if not file_exists(manifest_path):
        raise FileNotFoundError(f"Latest manifest not found: {manifest_path}")

    with fsspec.open(manifest_path, "r") as f:
        return json.load(f)


def log_dataset_manifest_to_mlflow(manifest: dict):
    """
    Log dataset lineage, snapshot paths and effective configuration to MLflow.

    Notes:
        The function expects an active MLflow run.
    """
    dataset_version = manifest["dataset_version"]

    mlflow.log_param("dataset_version", dataset_version)
    mlflow.log_param("dataset_environment", manifest.get("environment"))
    mlflow.log_param("dataset_config_name", manifest.get("config_name"))
    mlflow.log_param("git_commit", manifest.get("git_commit"))

    if manifest.get("seed") is not None:
        mlflow.log_param("seed", manifest.get("seed"))

    snapshots = manifest.get("snapshots", {})
    for key, value in snapshots.items():
        mlflow.log_param(f"data_{key}_path", value)

    manifest_json = json.dumps(manifest, indent=2)
    mlflow.log_text(manifest_json, f"dataset_manifest/{dataset_version}.json")

    effective_config = manifest.get("effective_config")
    if effective_config is not None:
        mlflow.log_text(
            json.dumps(effective_config, indent=2, sort_keys=True),
            f"dataset_manifest/{dataset_version}_effective_config.json",
        )

    logger.info(f"🧾 Logged dataset manifest to MLflow for version: {dataset_version}")

def get_dataset_paths_from_manifest(manifest: dict) -> dict[str, str]:
    """
    Extract versioned dataset paths from a dataset manifest.
    Keeps downstream training and reproduction code independent of project-specific artifact names.
    """
    snapshots = manifest["snapshots"]

    result = {
        "train_file": snapshots["split_train"],
        "val_file": snapshots["split_val"],
    }

    for key, value in snapshots.items():
        result[f"{key}_file"] = value

    return result