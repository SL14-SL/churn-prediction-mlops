import hashlib
import json

from src.inference.releases.storage import (
    copy_uri,
    load_json,
    sha256_uri,
    write_json,
)


def test_sha256_uri_returns_file_checksum(
    tmp_path,
):
    source = tmp_path / "artifact.txt"
    source.write_text(
        "churn-serving-artifact",
        encoding="utf-8",
    )

    expected = hashlib.sha256(
        b"churn-serving-artifact"
    ).hexdigest()

    assert sha256_uri(str(source)) == expected


def test_copy_uri_copies_local_file(
    tmp_path,
):
    source = tmp_path / "source.json"
    destination = (
        tmp_path
        / "release"
        / "artifact.json"
    )

    source.write_text(
        '{"version": 1}',
        encoding="utf-8",
    )

    copy_uri(
        str(source),
        str(destination),
    )

    assert destination.read_text(
        encoding="utf-8"
    ) == '{"version": 1}'


def test_write_and_read_json_roundtrip(
    tmp_path,
):
    destination = (
        tmp_path
        / "release"
        / "manifest.json"
    )

    payload = {
        "schema_version": 1,
        "release_id": "churn-release-1",
        "decision_threshold": 0.42,
        "columns": [
            "tenure",
            "monthlycharges",
        ],
    }

    write_json(
        str(destination),
        payload,
    )

    assert json.loads(
        destination.read_text(
            encoding="utf-8"
        )
    ) == payload

    assert load_json(
        str(destination)
    ) == payload


def test_copy_uri_creates_parent_directory(
    tmp_path,
):
    source = tmp_path / "source.txt"
    destination = (
        tmp_path
        / "nested"
        / "release"
        / "source.txt"
    )

    source.write_text(
        "artifact",
        encoding="utf-8",
    )

    copy_uri(
        str(source),
        str(destination),
    )

    assert destination.exists()