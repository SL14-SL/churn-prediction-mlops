import os
import shutil
from typing import Any

import gcsfs
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)

from src.configs.loader import (
    file_exists,
    get_path,
    load_config,
)
from src.configs.paths import join_uri
from src.data.validation.validate import (
    validate_train,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

SIMULATION_RANDOM_STATE = 42


def normalize_raw_churn_schema(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize raw Telco churn data before validation and persistence.

    ``TotalCharges`` may be inferred as either text or numeric depending on
    batch contents. Converting it consistently prevents mixed Parquet dtypes
    after combining canonical data and incremental batches.
    """
    result = df.copy()

    if "TotalCharges" in result.columns:
        result["TotalCharges"] = (
            result["TotalCharges"]
            .astype(str)
        )

    if (
        "customerid" in result.columns
        and "customerID"
        not in result.columns
    ):
        result = result.rename(
            columns={
                "customerid": (
                    "customerID"
                ),
            }
        )

    return result


def normalize_gcs_path(
    path: str,
) -> str:
    """Return a GCS path with an explicit ``gs://`` protocol."""
    if path.startswith(
        "gs://"
    ):
        return path

    return f"gs://{path}"


def list_csv_files(
    path: str,
) -> list[str]:
    """List CSV files from a local directory or GCS prefix."""
    if path.startswith(
        "gs://"
    ):
        filesystem = (
            gcsfs.GCSFileSystem()
        )

        files = filesystem.glob(
            f"{path.rstrip('/')}/*.csv"
        )

        return sorted(
            normalize_gcs_path(
                file_path
            )
            for file_path in files
        )

    if not os.path.isdir(
        path
    ):
        return []

    return sorted(
        os.path.join(
            path,
            filename,
        )
        for filename in os.listdir(
            path
        )
        if filename.endswith(
            ".csv"
        )
    )


def resolve_base_data_file(
    training_config: dict[str, Any],
) -> str:
    """
    Resolve the canonical raw training filename from configuration.

    Raises:
        KeyError: If the training feature source is not configured.
        ValueError: If the configured path is empty.
    """
    data_file = (
        training_config[
            "data"
        ][
            "feature_sources"
        ][
            "train"
        ][
            "path"
        ]
    )

    if not data_file:
        raise ValueError(
            "The canonical training data path "
            "is not configured."
        )

    return str(
        data_file
    )


def load_base_dataset(
    *,
    raw_path: str,
    training_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Load, normalize and validate the canonical Telco churn dataset.

    Args:
        raw_path: Local or GCS directory containing raw source data.
        training_config: Effective training configuration.

    Returns:
        The normalized and validated canonical dataset.

    Raises:
        FileNotFoundError: If the configured source file is unavailable.
        pandera.errors.SchemaError: If the dataset violates its schema.
    """
    data_file = resolve_base_data_file(
        training_config
    )

    source_path = join_uri(
        raw_path,
        data_file,
    )

    if not file_exists(
        source_path
    ):
        raise FileNotFoundError(
            "Canonical churn dataset "
            f"not found: {source_path}"
        )

    logger.info(
        "Loading canonical churn dataset | "
        "path=%s",
        source_path,
    )

    dataframe = pd.read_csv(
        source_path
    )

    dataframe = (
        normalize_raw_churn_schema(
            dataframe
        )
    )

    dataframe = validate_train(
        dataframe
    )

    logger.info(
        "Canonical churn dataset loaded "
        "and validated | rows=%s",
        len(dataframe),
    )

    return dataframe


def create_simulation_split(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    test_size: float,
    random_state: int = (
        SIMULATION_RANDOM_STATE
    ),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create stratified training and simulation partitions.

    Args:
        dataframe: Validated canonical churn observations.
        target_column: Binary churn target used for stratification.
        test_size: Fraction assigned to the simulation partition.
        random_state: Seed used for deterministic splitting.

    Returns:
        The base training partition and simulation Ground Truth partition.

    Raises:
        ValueError: If the split configuration, target or class distribution is
            unsuitable for a stratified split.
    """
    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between "
            "0 and 1."
        )

    if target_column not in dataframe.columns:
        raise ValueError(
            "Target column is missing from "
            f"canonical data: {target_column}"
        )

    if dataframe.empty:
        raise ValueError(
            "Canonical churn dataset is empty."
        )

    train_base, simulation_truth = (
        train_test_split(
            dataframe,
            test_size=test_size,
            random_state=random_state,
            stratify=(
                dataframe[target_column]
            ),
        )
    )

    train_base = (
        train_base
        .reset_index(drop=True)
    )
    simulation_truth = (
        simulation_truth
        .reset_index(drop=True)
    )

    logger.info(
        "Stratified simulation split completed | "
        "train_rows=%s | simulation_rows=%s",
        len(train_base),
        len(simulation_truth),
    )

    return (
        train_base,
        simulation_truth,
    )


def persist_simulation_source_if_missing(
    simulation_truth: pd.DataFrame,
    *,
    raw_path: str,
) -> None:
    """
    Persist the simulation Ground Truth when it does not yet exist.

    Existing simulation data is preserved so controlled demonstration runs
    remain reproducible across repeated ingestion executions.
    """
    simulation_path = join_uri(
        raw_path,
        "simulation_ground_truth.csv",
    )

    if file_exists(
        simulation_path
    ):
        logger.info(
            "Existing simulation Ground Truth "
            "preserved | path=%s",
            simulation_path,
        )
        return

    simulation_truth.to_csv(
        simulation_path,
        index=False,
    )

    logger.info(
        "Simulation Ground Truth created | "
        "path=%s | rows=%s",
        simulation_path,
        len(simulation_truth),
    )


def load_and_validate_batch(
    batch_path: str,
) -> pd.DataFrame:
    """
    Load, normalize and validate one incremental churn batch.

    Args:
        batch_path: Local or GCS path to the batch CSV.

    Returns:
        The normalized and validated incremental batch.
    """
    batch = pd.read_csv(
        batch_path
    )

    batch = normalize_raw_churn_schema(
        batch
    )

    return validate_train(
        batch
    )


def quarantine_local_batch(
    batch_path: str,
    *,
    quarantine_directory: str,
) -> str:
    """
    Move a rejected local batch into quarantine.

    Returns:
        The destination path of the quarantined batch.
    """
    os.makedirs(
        quarantine_directory,
        exist_ok=True,
    )

    destination_path = os.path.join(
        quarantine_directory,
        os.path.basename(
            batch_path
        ),
    )

    shutil.move(
        batch_path,
        destination_path,
    )

    logger.info(
        "Rejected batch moved to quarantine | "
        "source=%s | destination=%s",
        batch_path,
        destination_path,
    )

    return destination_path


def collect_incremental_batches(
    *,
    raw_path: str,
) -> list[pd.DataFrame]:
    """
    Collect and validate incremental churn batches.

    Invalid local batches are moved into quarantine. Invalid GCS batches remain
    at their source location and are excluded from the current ingestion run.

    Returns:
        All valid incremental batches in deterministic path order.
    """
    batch_directory = join_uri(
        raw_path,
        "new_batches",
    )
    quarantine_directory = (
        join_uri(
            raw_path,
            "quarantine",
        )
    )

    batches: list[
        pd.DataFrame
    ] = []

    for batch_path in list_csv_files(
        batch_directory
    ):
        try:
            batch = (
                load_and_validate_batch(
                    batch_path
                )
            )

        except Exception as error:
            logger.warning(
                "Incremental batch rejected | "
                "path=%s | reason=%s",
                batch_path,
                error,
            )

            if raw_path.startswith(
                "gs://"
            ):
                logger.warning(
                    "Remote batch remains at source "
                    "because GCS quarantine is not "
                    "implemented | path=%s",
                    batch_path,
                )
            else:
                quarantine_local_batch(
                    batch_path,
                    quarantine_directory=(
                        quarantine_directory
                    ),
                )

            continue

        batches.append(
            batch
        )

        logger.info(
            "Incremental batch validated | "
            "path=%s | rows=%s",
            batch_path,
            len(batch),
        )

    return batches


def merge_training_batches(
    train_base: pd.DataFrame,
    batches: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge the base training partition and valid incremental batches.

    Returns:
        The normalized and validated canonical training dataset.
    """
    if batches:
        final_train = pd.concat(
            [
                train_base,
                *batches,
            ],
            ignore_index=True,
        )

        logger.info(
            "Incremental batches integrated | "
            "batches=%s | rows=%s",
            len(batches),
            len(final_train),
        )

    else:
        final_train = (
            train_base.copy()
        )

        logger.info(
            "No incremental batches found."
        )

    final_train = (
        normalize_raw_churn_schema(
            final_train
        )
    )

    return validate_train(
        final_train
    )


def persist_validated_dataset(
    dataframe: pd.DataFrame,
    *,
    validated_path: str,
) -> str:
    """
    Persist the canonical validated training dataset as Parquet.

    Returns:
        The local or GCS path of the written Parquet artifact.
    """
    if not validated_path.startswith(
        "gs://"
    ):
        os.makedirs(
            validated_path,
            exist_ok=True,
        )

    output_path = join_uri(
        validated_path,
        "train.parquet",
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )

    logger.info(
        "Validated training dataset persisted | "
        "path=%s | rows=%s",
        output_path,
        len(dataframe),
    )

    return output_path


def ingest() -> None:
    """
    Execute the complete churn-data ingestion lifecycle.

    The lifecycle loads and validates canonical Telco data, creates a
    deterministic simulation partition, integrates valid incremental batches
    and persists the canonical training dataset for feature generation.
    """
    training_config = load_config(
        "training.yaml"
    )

    raw_path = get_path(
        "raw_data"
    )
    validated_path = get_path(
        "validated_data"
    )
    environment = os.getenv(
        "APP_ENV",
        "dev",
    )

    logger.info(
        "Starting churn ingestion | "
        "source=%s | environment=%s",
        raw_path,
        environment,
    )

    full_dataset = (
        load_base_dataset(
            raw_path=raw_path,
            training_config=(
                training_config
            ),
        )
    )

    training_section = (
        training_config.get(
            "training",
            {},
        )
    )
    data_section = (
        training_config.get(
            "data",
            {},
        )
    )

    test_size = float(
        training_section.get(
            "test_size",
            0.2,
        )
    )
    target_column = str(
        data_section[
            "target_column"
        ]
    )

    train_base, simulation_truth = (
        create_simulation_split(
            full_dataset,
            target_column=target_column,
            test_size=test_size,
            random_state=(
                SIMULATION_RANDOM_STATE
            ),
        )
    )

    persist_simulation_source_if_missing(
        simulation_truth,
        raw_path=raw_path,
    )

    batches = (
        collect_incremental_batches(
            raw_path=raw_path,
        )
    )

    final_train = (
        merge_training_batches(
            train_base,
            batches,
        )
    )

    persist_validated_dataset(
        final_train,
        validated_path=(
            validated_path
        ),
    )

    logger.info(
        "Churn ingestion completed successfully | "
        "rows=%s | output=%s",
        len(final_train),
        validated_path,
    )


if __name__ == "__main__":
    ingest()