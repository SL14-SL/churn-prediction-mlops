from __future__ import annotations

import argparse
from pathlib import (
    PurePosixPath,
)

import fsspec

from src.configs.loader import (
    ensure_dir,
    get_path,
)
from src.data.raw.ingest import ingest


RAW_FILES_TO_KEEP = {
    "Telco-Customer-Churn.csv",
    ".gitkeep",
}

GENERATED_FILES_TO_KEEP = {
    ".gitkeep",
}


def list_items(
    path: str,
) -> list[str]:
    filesystem, filesystem_path = (
        fsspec.core.url_to_fs(
            path
        )
    )

    if not filesystem.exists(
        filesystem_path
    ):
        return []

    return filesystem.ls(
        filesystem_path,
        detail=False,
    )


def display_path(
    item: str,
    base_path: str,
) -> str:
    if (
        base_path.startswith("gs://")
        and not item.startswith("gs://")
    ):
        return f"gs://{item}"

    return item


def remove_item(
    path: str,
) -> None:
    filesystem, filesystem_path = (
        fsspec.core.url_to_fs(
            path
        )
    )

    if filesystem.exists(
        filesystem_path
    ):
        print(
            f"Removing demo artifact: {path}"
        )
        filesystem.rm(
            filesystem_path,
            recursive=True,
        )


def clear_directory_contents(
    path: str,
    *,
    keep_names: set[str],
) -> None:
    """
    Remove directory contents while preserving explicit placeholder files.
    """
    ensure_dir(path)

    for item in list_items(path):
        item_path = display_path(
            item,
            path,
        )
        item_name = (
            PurePosixPath(
                item_path
            ).name
        )

        if item_name in keep_names:
            print(
                "Keeping demo artifact: "
                f"{item_path}"
            )
            continue

        remove_item(
            item_path
        )


def reset_churn_demo_run() -> None:
    """
    Reset mutable churn-demo state while preserving the active ML system.

    This intentionally keeps MLflow, registry aliases, serving releases,
    model files, dataset versions, and archived comparison results.
    """
    raw_path = get_path(
        "raw_data"
    )
    predictions_path = get_path(
        "predictions"
    )
    monitoring_path = get_path(
        "monitoring"
    )

    clear_directory_contents(
        raw_path,
        keep_names=RAW_FILES_TO_KEEP,
    )
    clear_directory_contents(
        predictions_path,
        keep_names=(
            GENERATED_FILES_TO_KEEP
        ),
    )
    clear_directory_contents(
        monitoring_path,
        keep_names=(
            GENERATED_FILES_TO_KEEP
        ),
    )

    print(
        "Recreating deterministic simulation "
        "ground truth..."
    )

    ingest()

    print(
        "Churn demo run reset complete."
    )
    print(
        "Preserved: MLflow, Champion, serving "
        "release, models and archived results."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reset mutable churn-demo data "
            "without resetting the active "
            "model system."
        )
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Confirm reset without an "
            "interactive prompt."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.yes:
        answer = input(
            "Reset predictions, monitoring "
            "history and simulation batches? "
            "[y/N] "
        )

        if (
            answer.strip().lower()
            != "y"
        ):
            raise SystemExit(
                "Demo run reset aborted."
            )

    reset_churn_demo_run()


if __name__ == "__main__":
    main()