import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST
from src.configs.loader import load_config
from src.utils.logger import get_logger



logger = get_logger(__name__)

CFG = load_config()
MODEL_NAME = CFG["model"]["registry_name"]

def register_model(run_id: str, alias: str = "champion"):
    """
    Registers a model version and assigns a registry alias.
    """
    if alias not in {"champion", "challenger"}:
        raise ValueError(f"Unsupported alias '{alias}'. Use 'champion' or 'challenger'.")

    client = MlflowClient()

    try:
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"Attempting to register model from run: {run_id}")

        # 1. Register the model
        model_version = mlflow.register_model(model_uri, MODEL_NAME)
        version = model_version.version

        # 2. Add useful Metadata as Tags (Helpful for Churn Analysis)
        run_data = client.get_run(run_id).data
        model_type = run_data.params.get("model_type", "unknown")
        
        client.set_model_version_tag(MODEL_NAME, version, "model_type", model_type)
        client.set_model_version_tag(MODEL_NAME, version, "deployed_at", str(pd.Timestamp.now()))

        logger.info(
            f"Successfully registered version {version} of '{MODEL_NAME}' ({model_type}). "
            f"Assigning alias '@{alias}'."
        )

        # 3. Assign Alias (Champion/Challenger)
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=alias,
            version=version,
        )

        logger.info(
            f"Registry update complete: Version {version} is now tagged as '@{alias}'."
        )

        return model_version

    except Exception as e:
        logger.error(f"Failed to register model or assign alias '{alias}': {str(e)}")
        raise

def champion_exists() -> bool:
    """
    Return whether the configured registered model has a Champion alias.

    A missing registered model is an expected bootstrap condition.
    Unexpected registry errors are propagated.
    """
    client = MlflowClient()

    try:
        registered_model = (
            client.get_registered_model(
                MODEL_NAME
            )
        )

    except MlflowException as error:
        missing_model_codes = {
            RESOURCE_DOES_NOT_EXIST,
            ErrorCode.Name(
                RESOURCE_DOES_NOT_EXIST
            ),
        }

        if (
            error.error_code
            in missing_model_codes
        ):
            return False

        raise

    aliases = (
        registered_model.aliases
        or {}
    )

    return "champion" in aliases

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        register_model(sys.argv[1])
    else:
        logger.warning("Please provide a valid Run ID to register a model.")