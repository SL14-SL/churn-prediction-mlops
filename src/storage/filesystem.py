import glob

from pathlib import Path

def _gcs_fs():
    try:
        import gcsfs
    except ImportError as exc:
        raise RuntimeError("gcsfs is required for gs:// path operations.") from exc

    return gcsfs.GCSFileSystem()


def file_exists(path: str) -> bool:
    """
    Check existence for local paths and gs:// paths.
    """
    path = str(path)

    if path.startswith("gs://"):
        fs = _gcs_fs()
        return fs.exists(path)

    return Path(path).exists()


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Local paths are created on disk.
    gs:// paths are left untouched because bucket/prefix creation is implicit.
    """
    path = str(path)

    if path.startswith("gs://"):
        return

    Path(path).mkdir(parents=True, exist_ok=True)


def list_files(path_pattern: str) -> list[str]:
    """
    List files for local glob patterns and gs:// glob patterns.
    """
    path_pattern = str(path_pattern)

    if path_pattern.startswith("gs://"):
        fs = _gcs_fs()
        files = fs.glob(path_pattern)

        return sorted(
            f"gs://{path}" if not str(path).startswith("gs://") else str(path)
            for path in files
        )

    return sorted(glob.glob(path_pattern))


def modified_time(path: str) -> float:
    """
    Return comparable modification time for local and gs:// paths.
    """
    path = str(path)

    if path.startswith("gs://"):
        fs = _gcs_fs()
        info = fs.info(path)

        value = (
            info.get("updated")
            or info.get("mtime")
            or info.get("created")
            or info.get("timeCreated")
        )

        if value is None:
            return 0.0

        if hasattr(value, "timestamp"):
            return float(value.timestamp())

        if isinstance(value, (int, float)):
            return float(value)

        try:
            import pandas as pd

            return float(pd.Timestamp(value).timestamp())
        except Exception:
            return 0.0

    return Path(path).stat().st_mtime


def remove_file(path: str) -> None:
    """
    Remove a local or gs:// file if it exists.
    """
    path = str(path)

    if path.startswith("gs://"):
        fs = _gcs_fs()
        if fs.exists(path):
            fs.rm(path)
        return

    local_path = Path(path)
    if local_path.exists():
        local_path.unlink()


def read_text(path: str) -> str:
    """
    Read text from local or gs:// path.
    """
    path = str(path)

    if path.startswith("gs://"):
        fs = _gcs_fs()
        with fs.open(path, "r") as file:
            return file.read()

    return Path(path).read_text(encoding="utf-8")


def write_text(path: str, text: str) -> None:
    """
    Write text to local or gs:// path.
    """
    path = str(path)

    if path.startswith("gs://"):
        fs = _gcs_fs()
        with fs.open(path, "w") as file:
            file.write(text)
        return

    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(text, encoding="utf-8")