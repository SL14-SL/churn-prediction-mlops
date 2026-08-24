from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

@dataclass(frozen=True)
class ServingArtifactReference:
    """
    Reference to one immutable serving artifact.
    """

    path: str
    sha256: str


@dataclass(frozen=True)
class ServingReleaseManifest:
    """
    Persistent description of one complete churn serving release.
    """

    schema_version: int
    release_id: str
    created_at_utc: str

    model_name: str
    model_version: str
    model_run_id: str
    model_uri: str
    model_type: str

    decision_threshold: float

    dataset_version: str | None
    config_hash: str | None
    git_commit: str | None

    feature_schema: (
        ServingArtifactReference
    )

    prediction_probe: (
        ServingArtifactReference | None
    ) = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)

def validate_artifact_reference(
    reference: ServingArtifactReference,
    *,
    name: str,
) -> None:
    """
    Validate one immutable serving artifact reference.
    """
    if not isinstance(
        reference,
        ServingArtifactReference,
    ):
        raise ValueError(
            f"Serving manifest has an invalid {name} reference."
        )

    if not reference.path:
        raise ValueError(
            f"Serving manifest {name} has no path."
        )

    if not reference.sha256:
        raise ValueError(
            f"Serving manifest {name} has no checksum."
        )

    
def validate_serving_manifest(
    manifest: ServingReleaseManifest,
) -> None:
    """
    Raise ValueError if a serving release manifest is incomplete.
    """
    if manifest.schema_version < 1:
        raise ValueError(
            "Serving manifest has an invalid schema version."
        )

    if not manifest.release_id:
        raise ValueError(
            "Serving manifest has no release ID."
        )

    if not manifest.created_at_utc:
        raise ValueError(
            "Serving manifest has no creation timestamp."
        )

    if not manifest.model_name:
        raise ValueError(
            "Serving manifest has no model name."
        )

    if not manifest.model_version:
        raise ValueError(
            "Serving manifest has no model version."
        )

    if not manifest.model_run_id:
        raise ValueError(
            "Serving manifest has no model run ID."
        )

    if not manifest.model_uri:
        raise ValueError(
            "Serving manifest has no model URI."
        )

    if not manifest.model_type:
        raise ValueError(
            "Serving manifest has no model type."
        )

    if not isinstance(
        manifest.decision_threshold,
        (int, float),
    ):
        raise ValueError(
            "Serving manifest has an invalid decision threshold."
        )

    if not 0.0 <= float(
        manifest.decision_threshold
    ) <= 1.0:
        raise ValueError(
            "Serving manifest decision threshold must be "
            "between 0 and 1."
        )

    validate_artifact_reference(
        manifest.feature_schema,
        name="feature schema",
    )

    if manifest.prediction_probe is not None:
        validate_artifact_reference(
            manifest.prediction_probe,
            name="prediction probe",
        )

@dataclass(frozen=True)
class ServingBundle:
    """
    Complete validated state required for churn inference.

    A bundle is created completely before it replaces the currently
    active serving state.
    """
    release_id: str 
    manifest: ServingReleaseManifest

    model: Any
    model_name: str
    model_type: str

    decision_threshold: float
    feature_schema: dict[str, Any]

    serving_alias: str
    model_uri: str
    model_version: str
    model_run_id: str


def validate_serving_bundle(
    bundle: ServingBundle,
) -> None:
    """
    Raise ValueError if a churn serving bundle is incomplete or invalid.
    """
    if not bundle.release_id:
        raise ValueError(
            "Serving bundle has no release ID."
        )

    if not isinstance(
        bundle.manifest,
        ServingReleaseManifest,
    ):
        raise ValueError(
            "Serving bundle has no valid manifest."
        )

    validate_serving_manifest(
        bundle.manifest
    )

    if bundle.model is None:
        raise ValueError(
            "Serving bundle has no model."
        )

    if not bundle.model_name:
        raise ValueError(
            "Serving bundle has no model name."
        )

    if not bundle.model_type:
        raise ValueError(
            "Serving bundle has no model type."
        )

    if not bundle.serving_alias:
        raise ValueError(
            "Serving bundle has no serving alias."
        )

    if not bundle.model_uri:
        raise ValueError(
            "Serving bundle has no model URI."
        )

    if not bundle.model_version:
        raise ValueError(
            "Serving bundle has no model version."
        )

    if not bundle.model_run_id:
        raise ValueError(
            "Serving bundle has no model run ID."
        )

    if not isinstance(
        bundle.decision_threshold,
        (int, float),
    ):
        raise ValueError(
            "Serving bundle has an invalid decision threshold."
        )

    if not 0.0 <= float(
        bundle.decision_threshold
    ) <= 1.0:
        raise ValueError(
            "Serving bundle decision threshold must be between 0 and 1."
        )

    if not isinstance(
        bundle.feature_schema,
        dict,
    ):
        raise ValueError(
            "Serving bundle has an invalid feature schema."
        )

    columns = bundle.feature_schema.get(
        "columns"
    )

    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "Serving bundle feature schema has no columns."
        )

    if not all(
        isinstance(column, str) and column
        for column in columns
    ):
        raise ValueError(
            "Serving bundle feature schema contains invalid columns."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Serving bundle feature schema contains duplicate columns."
        )

    dtypes = bundle.feature_schema.get(
        "dtypes",
        {},
    )

    if not isinstance(dtypes, dict):
        raise ValueError(
            "Serving bundle feature schema has invalid dtypes."
        )

    unknown_dtype_columns = (
        set(dtypes) - set(columns)
    )

    if unknown_dtype_columns:
        raise ValueError(
            "Serving bundle feature schema contains dtypes "
            "for unknown columns: "
            f"{sorted(unknown_dtype_columns)}."
        )

    if (
        bundle.manifest.release_id
        != bundle.release_id
    ):
        raise ValueError(
            "Serving bundle release ID does not "
            "match manifest."
        )

    if (
        bundle.manifest.model_name
        != bundle.model_name
    ):
        raise ValueError(
            "Serving bundle model name does not "
            "match manifest."
        )

    if (
        bundle.manifest.model_version
        != bundle.model_version
    ):
        raise ValueError(
            "Serving bundle model version does not "
            "match manifest."
        )

    if (
        bundle.manifest.model_run_id
        != bundle.model_run_id
    ):
        raise ValueError(
            "Serving bundle model run ID does not "
            "match manifest."
        )

    if (
        bundle.manifest.model_uri
        != bundle.model_uri
    ):
        raise ValueError(
            "Serving bundle model URI does not "
            "match manifest."
        )

    if (
        bundle.manifest.model_type
        != bundle.model_type
    ):
        raise ValueError(
            "Serving bundle model type does not "
            "match manifest."
        )

    if (
        float(
            bundle.manifest.decision_threshold
        )
        != float(
            bundle.decision_threshold
        )
    ):
        raise ValueError(
            "Serving bundle decision threshold "
            "does not match manifest."
        )