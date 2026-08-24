import os
import re
from typing import Any

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

def _detect_environment() -> str:
    """
    Determine active environment.

    Priority:
    1. APP_ENV
    2. K_SERVICE -> prod
    3. dev
    """
    env = os.getenv("APP_ENV")
    if env:
        return env

    if os.getenv("K_SERVICE"):
        return "prod"

    return "dev"


def _resolve_env_placeholders(value: Any) -> Any:
    """
    Recursively resolve ${VAR} and ${VAR:-default} placeholders in YAML content.
    Leaves unresolved placeholders unchanged if no env var/default is available.
    """
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]

    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.getenv(var_name)

            if env_value is not None:
                return env_value
            if default is not None:
                return default
            return match.group(0)

        return _ENV_VAR_PATTERN.sub(replace, value)

    return value


def _override_gcs_bucket_paths(config: dict[str, Any]) -> dict[str, Any]:
    """
    Override gs:// bucket prefixes in config['paths'] when GCS_BUCKET_NAME is set.

    Example:
      gs://old-bucket/data/raw
    becomes:
      gs://new-bucket/data/raw
    """
    env_bucket = os.getenv("GCS_BUCKET_NAME")
    if not env_bucket:
        return config

    bucket_prefix = "" if env_bucket.startswith("gs://") else "gs://"
    new_base_path = f"{bucket_prefix}{env_bucket}"

    paths = config.get("paths")
    if not isinstance(paths, dict):
        return config

    for key, path in paths.items():
        if isinstance(path, str) and path.startswith("gs://"):
            parts = path.replace("gs://", "", 1).split("/", 1)
            if len(parts) > 1:
                paths[key] = f"{new_base_path}/{parts[1]}"
            else:
                paths[key] = new_base_path

    return config


def _inject_runtime_env(config: dict[str, Any]) -> None:
    """
    Push selected config values into process env for downstream libraries.
    """
    services = config.get("services", {})
    if isinstance(services, dict):
        prefect_api_url = services.get("prefect_api_url")
        if prefect_api_url:
            os.environ.setdefault("PREFECT_API_URL", str(prefect_api_url))

    tracking = config.get("tracking", {})
    mlflow_tracking_uri = None

    if isinstance(tracking, dict):
        mlflow_tracking_uri = tracking.get("mlflow_tracking_uri")

    if not mlflow_tracking_uri:
        mlflow_tracking_uri = config.get("mlflow_tracking_uri")

    if mlflow_tracking_uri and "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["MLFLOW_TRACKING_URI"] = str(mlflow_tracking_uri)

