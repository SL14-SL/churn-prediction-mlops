import os
from unittest.mock import (
    MagicMock,
)

import pandas as pd
import pytest

from src.data.raw import ingest


def build_churn_frame(
    customer_ids: list[str],
    *,
    churn: list[str] | None = None,
) -> pd.DataFrame:
    """Build a schema-compatible Telco churn dataframe."""
    row_count = len(
        customer_ids
    )

    if churn is None:
        churn = [
            (
                "Yes"
                if index % 2
                else "No"
            )
            for index in range(
                row_count
            )
        ]

    return pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": ["Male"] * row_count,
            "SeniorCitizen": [0] * row_count,
            "Partner": ["No"] * row_count,
            "Dependents": ["No"] * row_count,
            "tenure": list(
                range(
                    1,
                    row_count + 1,
                )
            ),
            "PhoneService": ["Yes"] * row_count,
            "MultipleLines": ["No"] * row_count,
            "InternetService": [
                "DSL"
            ] * row_count,
            "OnlineSecurity": ["No"] * row_count,
            "OnlineBackup": ["No"] * row_count,
            "DeviceProtection": [
                "No"
            ] * row_count,
            "TechSupport": ["No"] * row_count,
            "StreamingTV": ["No"] * row_count,
            "StreamingMovies": [
                "No"
            ] * row_count,
            "Contract": [
                "Month-to-month"
            ] * row_count,
            "PaperlessBilling": [
                "Yes"
            ] * row_count,
            "PaymentMethod": [
                "Electronic check"
            ] * row_count,
            "MonthlyCharges": [
                70.0
            ] * row_count,
            "TotalCharges": [
                str(
                    70.0
                    * (index + 1)
                )
                for index in range(
                    row_count
                )
            ],
            "Churn": churn,
        }
    )


def test_normalize_raw_churn_schema_converts_total_charges_to_string():
    frame = pd.DataFrame(
        {
            "customerID": [
                "customer-1",
            ],
            "TotalCharges": [
                123.45,
            ],
        }
    )

    result = (
        ingest
        .normalize_raw_churn_schema(
            frame
        )
    )

    assert (
        result["TotalCharges"].dtype
        == object
    )
    assert (
        result.loc[
            0,
            "TotalCharges",
        ]
        == "123.45"
    )

    # Input dataframe remains unchanged.
    assert (
        frame["TotalCharges"].dtype
        != object
        or frame.loc[
            0,
            "TotalCharges",
        ]
        == 123.45
    )


def test_normalize_raw_churn_schema_renames_customer_id():
    frame = pd.DataFrame(
        {
            "customerid": [
                "customer-1",
            ],
        }
    )

    result = (
        ingest
        .normalize_raw_churn_schema(
            frame
        )
    )

    assert "customerID" in result.columns
    assert "customerid" not in (
        result.columns
    )


def test_normalize_raw_churn_schema_preserves_existing_customer_id():
    frame = pd.DataFrame(
        {
            "customerID": [
                "canonical",
            ],
            "customerid": [
                "legacy",
            ],
        }
    )

    result = (
        ingest
        .normalize_raw_churn_schema(
            frame
        )
    )

    assert (
        result.loc[
            0,
            "customerID",
        ]
        == "canonical"
    )


def test_normalize_gcs_path_adds_protocol():
    assert (
        ingest.normalize_gcs_path(
            "bucket/path/file.csv"
        )
        == "gs://bucket/path/file.csv"
    )


def test_normalize_gcs_path_preserves_protocol():
    path = "gs://bucket/path/file.csv"

    assert (
        ingest.normalize_gcs_path(
            path
        )
        == path
    )


def test_list_csv_files_returns_sorted_local_files(
    tmp_path,
):
    batch_directory = (
        tmp_path
        / "new_batches"
    )
    batch_directory.mkdir()

    (
        batch_directory
        / "b.csv"
    ).write_text(
        "value\n2\n",
        encoding="utf-8",
    )
    (
        batch_directory
        / "a.csv"
    ).write_text(
        "value\n1\n",
        encoding="utf-8",
    )
    (
        batch_directory
        / "notes.txt"
    ).write_text(
        "ignored",
        encoding="utf-8",
    )

    files = ingest.list_csv_files(
        str(batch_directory)
    )

    assert files == [
        str(
            batch_directory
            / "a.csv"
        ),
        str(
            batch_directory
            / "b.csv"
        ),
    ]


def test_list_csv_files_returns_empty_for_missing_directory(
    tmp_path,
):
    files = ingest.list_csv_files(
        str(
            tmp_path
            / "missing"
        )
    )

    assert files == []


def test_list_csv_files_lists_gcs_objects(
    monkeypatch,
):
    filesystem = MagicMock()
    filesystem.glob.return_value = [
        "bucket/new_batches/b.csv",
        "bucket/new_batches/a.csv",
    ]

    filesystem_factory = MagicMock(
        return_value=filesystem
    )

    monkeypatch.setattr(
        ingest.gcsfs,
        "GCSFileSystem",
        filesystem_factory,
    )

    files = ingest.list_csv_files(
        "gs://bucket/new_batches"
    )

    assert files == [
        "gs://bucket/new_batches/a.csv",
        "gs://bucket/new_batches/b.csv",
    ]

    filesystem.glob.assert_called_once_with(
        "gs://bucket/new_batches/*.csv"
    )


def test_resolve_base_data_file():
    config = {
        "data": {
            "feature_sources": {
                "train": {
                    "path": (
                        "telco.csv"
                    ),
                }
            }
        }
    }

    result = (
        ingest.resolve_base_data_file(
            config
        )
    )

    assert result == "telco.csv"


def test_resolve_base_data_file_rejects_empty_path():
    config = {
        "data": {
            "feature_sources": {
                "train": {
                    "path": "",
                }
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="not configured",
    ):
        ingest.resolve_base_data_file(
            config
        )


def test_load_base_dataset_normalizes_and_validates(
    monkeypatch,
):
    raw_frame = build_churn_frame(
        [
            "customer-1",
            "customer-2",
        ]
    )
    normalized_frame = raw_frame.copy()
    validated_frame = (
        normalized_frame.copy()
    )

    monkeypatch.setattr(
        ingest,
        "file_exists",
        MagicMock(
            return_value=True
        ),
    )
    monkeypatch.setattr(
        ingest.pd,
        "read_csv",
        MagicMock(
            return_value=raw_frame
        ),
    )

    normalize_mock = MagicMock(
        return_value=normalized_frame
    )
    validate_mock = MagicMock(
        return_value=validated_frame
    )

    monkeypatch.setattr(
        ingest,
        "normalize_raw_churn_schema",
        normalize_mock,
    )
    monkeypatch.setattr(
        ingest,
        "validate_train",
        validate_mock,
    )

    result = ingest.load_base_dataset(
        raw_path="data/raw",
        training_config={
            "data": {
                "feature_sources": {
                    "train": {
                        "path": (
                            "telco.csv"
                        ),
                    }
                }
            }
        },
    )

    pd.testing.assert_frame_equal(
        result,
        validated_frame,
    )

    ingest.pd.read_csv\
        .assert_called_once_with(
            "data/raw/telco.csv"
        )

    normalize_mock.assert_called_once_with(
        raw_frame
    )
    validate_mock.assert_called_once_with(
        normalized_frame
    )


def test_load_base_dataset_rejects_missing_source(
    monkeypatch,
):
    monkeypatch.setattr(
        ingest,
        "file_exists",
        MagicMock(
            return_value=False
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        ingest.load_base_dataset(
            raw_path="data/raw",
            training_config={
                "data": {
                    "feature_sources": {
                        "train": {
                            "path": (
                                "telco.csv"
                            ),
                        }
                    }
                }
            },
        )


def test_create_simulation_split_is_deterministic_and_stratified():
    frame = build_churn_frame(
        [
            f"customer-{index}"
            for index in range(20)
        ],
        churn=[
            (
                "Yes"
                if index % 2
                else "No"
            )
            for index in range(20)
        ],
    )

    train_first, simulation_first = (
        ingest.create_simulation_split(
            frame,
            target_column="Churn",
            test_size=0.2,
            random_state=42,
        )
    )

    train_second, simulation_second = (
        ingest.create_simulation_split(
            frame,
            target_column="Churn",
            test_size=0.2,
            random_state=42,
        )
    )

    pd.testing.assert_frame_equal(
        train_first,
        train_second,
    )
    pd.testing.assert_frame_equal(
        simulation_first,
        simulation_second,
    )

    assert len(train_first) == 16
    assert len(simulation_first) == 4

    assert set(
        simulation_first["Churn"]
    ) == {
        "No",
        "Yes",
    }


@pytest.mark.parametrize(
    "test_size",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_create_simulation_split_rejects_invalid_size(
    test_size: float,
):
    frame = build_churn_frame(
        [
            "customer-1",
            "customer-2",
        ]
    )

    with pytest.raises(
        ValueError,
        match="test_size",
    ):
        ingest.create_simulation_split(
            frame,
            target_column="Churn",
            test_size=test_size,
        )


def test_create_simulation_split_rejects_missing_target():
    frame = build_churn_frame(
        [
            "customer-1",
            "customer-2",
        ]
    ).drop(
        columns=[
            "Churn",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Target column",
    ):
        ingest.create_simulation_split(
            frame,
            target_column="Churn",
            test_size=0.5,
        )


def test_create_simulation_split_rejects_empty_frame():
    frame = build_churn_frame(
        []
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        ingest.create_simulation_split(
            frame,
            target_column="Churn",
            test_size=0.5,
        )


def test_persist_simulation_source_creates_missing_file(
    tmp_path,
):
    simulation = build_churn_frame(
        [
            "customer-1",
            "customer-2",
        ]
    )

    ingest.persist_simulation_source_if_missing(
        simulation,
        raw_path=str(tmp_path),
    )

    output_path = (
        tmp_path
        / "simulation_ground_truth.csv"
    )

    assert output_path.exists()

    saved = pd.read_csv(
        output_path
    )

    assert len(saved) == 2


def test_persist_simulation_source_preserves_existing_file(
    tmp_path,
):
    output_path = (
        tmp_path
        / "simulation_ground_truth.csv"
    )

    output_path.write_text(
        "existing-content",
        encoding="utf-8",
    )

    simulation = build_churn_frame(
        [
            "customer-1",
            "customer-2",
        ]
    )

    ingest.persist_simulation_source_if_missing(
        simulation,
        raw_path=str(tmp_path),
    )

    assert output_path.read_text(
        encoding="utf-8"
    ) == "existing-content"


def test_load_and_validate_batch(
    monkeypatch,
):
    raw_batch = build_churn_frame(
        [
            "customer-3",
        ],
        churn=[
            "Yes",
        ],
    )
    normalized_batch = (
        raw_batch.copy()
    )
    validated_batch = (
        normalized_batch.copy()
    )

    monkeypatch.setattr(
        ingest.pd,
        "read_csv",
        MagicMock(
            return_value=raw_batch
        ),
    )
    monkeypatch.setattr(
        ingest,
        "normalize_raw_churn_schema",
        MagicMock(
            return_value=(
                normalized_batch
            )
        ),
    )
    monkeypatch.setattr(
        ingest,
        "validate_train",
        MagicMock(
            return_value=(
                validated_batch
            )
        ),
    )

    result = (
        ingest.load_and_validate_batch(
            "data/raw/new_batches/batch.csv"
        )
    )

    pd.testing.assert_frame_equal(
        result,
        validated_batch,
    )


def test_quarantine_local_batch_moves_file(
    tmp_path,
):
    batch_path = (
        tmp_path
        / "bad.csv"
    )
    quarantine_directory = (
        tmp_path
        / "quarantine"
    )

    batch_path.write_text(
        "invalid",
        encoding="utf-8",
    )

    destination = (
        ingest.quarantine_local_batch(
            str(batch_path),
            quarantine_directory=str(
                quarantine_directory
            ),
        )
    )

    expected_destination = (
        quarantine_directory
        / "bad.csv"
    )

    assert destination == str(
        expected_destination
    )
    assert not batch_path.exists()
    assert expected_destination.exists()


def test_collect_incremental_batches_returns_valid_batches(
    monkeypatch,
):
    valid_batch = build_churn_frame(
        [
            "customer-3",
        ],
        churn=[
            "Yes",
        ],
    )

    monkeypatch.setattr(
        ingest,
        "list_csv_files",
        MagicMock(
            return_value=[
                (
                    "data/raw/"
                    "new_batches/valid.csv"
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        ingest,
        "load_and_validate_batch",
        MagicMock(
            return_value=valid_batch
        ),
    )

    batches = (
        ingest.collect_incremental_batches(
            raw_path="data/raw",
        )
    )

    assert len(batches) == 1

    pd.testing.assert_frame_equal(
        batches[0],
        valid_batch,
    )


def test_collect_incremental_batches_quarantines_invalid_local_batch(
    monkeypatch,
):
    quarantine_mock = MagicMock()

    monkeypatch.setattr(
        ingest,
        "list_csv_files",
        MagicMock(
            return_value=[
                (
                    "data/raw/"
                    "new_batches/bad.csv"
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        ingest,
        "load_and_validate_batch",
        MagicMock(
            side_effect=ValueError(
                "Invalid batch"
            )
        ),
    )
    monkeypatch.setattr(
        ingest,
        "quarantine_local_batch",
        quarantine_mock,
    )

    batches = (
        ingest.collect_incremental_batches(
            raw_path="data/raw",
        )
    )

    assert batches == []

    quarantine_mock.assert_called_once_with(
        "data/raw/new_batches/bad.csv",
        quarantine_directory=(
            "data/raw/quarantine"
        ),
    )


def test_collect_incremental_batches_does_not_move_invalid_gcs_batch(
    monkeypatch,
):
    quarantine_mock = MagicMock()

    monkeypatch.setattr(
        ingest,
        "list_csv_files",
        MagicMock(
            return_value=[
                (
                    "gs://bucket/data/raw/"
                    "new_batches/bad.csv"
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        ingest,
        "load_and_validate_batch",
        MagicMock(
            side_effect=ValueError(
                "Invalid batch"
            )
        ),
    )
    monkeypatch.setattr(
        ingest,
        "quarantine_local_batch",
        quarantine_mock,
    )

    batches = (
        ingest.collect_incremental_batches(
            raw_path=(
                "gs://bucket/data/raw"
            ),
        )
    )

    assert batches == []
    quarantine_mock.assert_not_called()


def test_merge_training_batches_combines_and_validates(
    monkeypatch,
):
    train_base = build_churn_frame(
        [
            "customer-1",
            "customer-2",
        ]
    )
    batch = build_churn_frame(
        [
            "customer-3",
        ],
        churn=[
            "Yes",
        ],
    )

    normalized = pd.concat(
        [
            train_base,
            batch,
        ],
        ignore_index=True,
    )
    validated = normalized.copy()

    normalize_mock = MagicMock(
        return_value=normalized
    )
    validate_mock = MagicMock(
        return_value=validated
    )

    monkeypatch.setattr(
        ingest,
        "normalize_raw_churn_schema",
        normalize_mock,
    )
    monkeypatch.setattr(
        ingest,
        "validate_train",
        validate_mock,
    )

    result = (
        ingest.merge_training_batches(
            train_base,
            [
                batch,
            ],
        )
    )

    pd.testing.assert_frame_equal(
        result,
        validated,
    )

    normalize_mock.assert_called_once()
    validate_mock.assert_called_once_with(
        normalized
    )


def test_merge_training_batches_without_batches_returns_copy(
    monkeypatch,
):
    train_base = build_churn_frame(
        [
            "customer-1",
            "customer-2",
        ]
    )

    monkeypatch.setattr(
        ingest,
        "normalize_raw_churn_schema",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        ingest,
        "validate_train",
        lambda frame: frame,
    )

    result = (
        ingest.merge_training_batches(
            train_base,
            [],
        )
    )

    pd.testing.assert_frame_equal(
        result,
        train_base,
    )

    assert result is not train_base


def test_persist_validated_dataset_writes_parquet(
    tmp_path,
):
    frame = build_churn_frame(
        [
            "customer-1",
            "customer-2",
        ]
    )

    output_path = (
        ingest.persist_validated_dataset(
            frame,
            validated_path=str(
                tmp_path
            ),
        )
    )

    expected_path = (
        tmp_path
        / "train.parquet"
    )

    assert output_path == str(
        expected_path
    )
    assert expected_path.exists()

    saved = pd.read_parquet(
        expected_path
    )

    pd.testing.assert_frame_equal(
        saved,
        frame,
    )


def test_ingest_orchestrates_complete_lifecycle(
    monkeypatch,
):
    full_dataset = build_churn_frame(
        [
            "customer-1",
            "customer-2",
            "customer-3",
            "customer-4",
        ]
    )
    train_base = full_dataset.iloc[
        :2
    ].copy()
    simulation_truth = (
        full_dataset.iloc[
            2:
        ].copy()
    )
    batch = build_churn_frame(
        [
            "customer-5",
        ],
        churn=[
            "Yes",
        ],
    )
    final_train = pd.concat(
        [
            train_base,
            batch,
        ],
        ignore_index=True,
    )

    training_config = {
        "training": {
            "test_size": 0.5,
        },
        "data": {
            "target_column": "Churn",
            "feature_sources": {
                "train": {
                    "path": (
                        "telco.csv"
                    ),
                }
            },
        },
    }

    load_config_mock = MagicMock(
        return_value=training_config
    )
    load_base_mock = MagicMock(
        return_value=full_dataset
    )
    split_mock = MagicMock(
        return_value=(
            train_base,
            simulation_truth,
        )
    )
    persist_simulation_mock = (
        MagicMock()
    )
    collect_batches_mock = MagicMock(
        return_value=[
            batch,
        ]
    )
    merge_mock = MagicMock(
        return_value=final_train
    )
    persist_validated_mock = (
        MagicMock(
            return_value=(
                "data/validated/"
                "train.parquet"
            )
        )
    )

    monkeypatch.setattr(
        ingest,
        "load_config",
        load_config_mock,
    )
    monkeypatch.setattr(
        ingest,
        "get_path",
        lambda name: {
            "raw_data": "data/raw",
            "validated_data": (
                "data/validated"
            ),
        }[name],
    )
    monkeypatch.setattr(
        ingest,
        "load_base_dataset",
        load_base_mock,
    )
    monkeypatch.setattr(
        ingest,
        "create_simulation_split",
        split_mock,
    )
    monkeypatch.setattr(
        ingest,
        "persist_simulation_source_if_missing",
        persist_simulation_mock,
    )
    monkeypatch.setattr(
        ingest,
        "collect_incremental_batches",
        collect_batches_mock,
    )
    monkeypatch.setattr(
        ingest,
        "merge_training_batches",
        merge_mock,
    )
    monkeypatch.setattr(
        ingest,
        "persist_validated_dataset",
        persist_validated_mock,
    )

    monkeypatch.setenv(
        "APP_ENV",
        "dev",
    )

    ingest.ingest()

    load_config_mock.assert_called_once_with(
        "training.yaml"
    )
    load_base_mock.assert_called_once_with(
        raw_path="data/raw",
        training_config=training_config,
    )
    split_mock.assert_called_once_with(
        full_dataset,
        target_column="Churn",
        test_size=0.5,
        random_state=(
            ingest
            .SIMULATION_RANDOM_STATE
        ),
    )
    persist_simulation_mock\
        .assert_called_once_with(
            simulation_truth,
            raw_path="data/raw",
        )
    collect_batches_mock\
        .assert_called_once_with(
            raw_path="data/raw",
        )
    merge_mock.assert_called_once_with(
        train_base,
        [
            batch,
        ],
    )
    persist_validated_mock\
        .assert_called_once_with(
            final_train,
            validated_path=(
                "data/validated"
            ),
        )