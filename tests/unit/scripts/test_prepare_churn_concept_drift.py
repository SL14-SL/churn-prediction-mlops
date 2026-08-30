from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import (
    prepare_churn_concept_drift
    as concept_drift,
)


def build_source_data() -> pd.DataFrame:
    rows = []

    for index in range(40):
        rows.append(
            {
                "customerID": (
                    f"eligible-no-{index:03d}"
                ),
                "Churn": "No",
                "tenure": 24,
                "Contract": (
                    "Month-to-month"
                ),
                "PaymentMethod": (
                    "Bank transfer "
                    "(automatic)"
                ),
                "MonthlyCharges": 70.0,
            }
        )

    for index in range(10):
        rows.append(
            {
                "customerID": (
                    f"eligible-yes-{index:03d}"
                ),
                "Churn": "Yes",
                "tenure": 24,
                "Contract": (
                    "Month-to-month"
                ),
                "PaymentMethod": (
                    "Bank transfer "
                    "(automatic)"
                ),
                "MonthlyCharges": 75.0,
            }
        )

    for index in range(10):
        rows.append(
            {
                "customerID": (
                    f"ineligible-no-{index:03d}"
                ),
                "Churn": "No",
                "tenure": 3,
                "Contract": "One year",
                "PaymentMethod": (
                    "Electronic check"
                ),
                "MonthlyCharges": 40.0,
            }
        )

    return pd.DataFrame(rows)


def configure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> tuple:
    simulation_file = (
        tmp_path
        / "simulation_ground_truth.csv"
    )
    manifest_file = (
        tmp_path
        / "churn_concept_drift_manifest.json"
    )
    audit_file = (
        tmp_path
        / "churn_concept_drift_audit.csv"
    )

    monkeypatch.setattr(
        concept_drift,
        "SIMULATION_FILE",
        str(simulation_file),
    )
    monkeypatch.setattr(
        concept_drift,
        "MANIFEST_FILE",
        str(manifest_file),
    )
    monkeypatch.setattr(
        concept_drift,
        "AUDIT_FILE",
        str(audit_file),
    )

    return (
        simulation_file,
        manifest_file,
        audit_file,
    )


@pytest.mark.parametrize(
    (
        "day",
        "expected_rate",
    ),
    [
        (1, 0.0),
        (5, 0.0),
        (6, 0.2),
        (7, 0.4),
        (8, 0.6),
        (15, 0.6),
    ],
)
def test_target_flip_rate(
    day: int,
    expected_rate: float,
):
    rate = (
        concept_drift.target_flip_rate(
            day=day,
            drift_start_day=6,
            ramp_days=3,
            post_drift_rate=0.6,
        )
    )

    assert rate == pytest.approx(
        expected_rate
    )


def test_prepare_scenario_applies_only_controlled_label_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    (
        simulation_file,
        manifest_file,
        audit_file,
    ) = configure_paths(
        monkeypatch,
        tmp_path,
    )

    source = build_source_data()
    source.to_csv(
        simulation_file,
        index=False,
    )

    manifest = (
        concept_drift.prepare_scenario(
            max_days=3,
            batch_size=10,
            drift_start_day=2,
            ramp_days=1,
            post_drift_rate=1.0,
            random_state=42,
        )
    )

    prepared = pd.read_csv(
        simulation_file
    )
    audit = pd.read_csv(
        audit_file
    )

    assert len(prepared) == len(source)
    assert len(audit) == 30

    assert set(prepared.columns) == set(
        source.columns
    )
    assert (
        "Churn_original"
        not in prepared.columns
    )
    assert (
        "concept_drift_applied"
        not in prepared.columns
    )

    pre_drift = audit[
        audit["simulation_day"] == 1
    ]
    post_drift = audit[
        audit["simulation_day"] >= 2
    ]

    assert not pre_drift[
        "concept_drift_applied"
    ].any()

    eligible_post_drift = post_drift[
        post_drift[
            "concept_drift_eligible"
        ]
    ]
    assert not eligible_post_drift.empty
    assert eligible_post_drift[
        "concept_drift_applied"
    ].all()

    changed = audit[
        audit[
            "concept_drift_applied"
        ]
    ]
    assert not changed.empty
    assert (
        changed["Churn_original"]
        == "No"
    ).all()
    assert (
        changed["Churn_effective"]
        == "Yes"
    ).all()

    ineligible = audit[
        ~audit[
            "concept_drift_eligible"
        ]
    ]
    assert (
        ineligible["Churn_original"]
        == ineligible["Churn_effective"]
    ).all()

    assert manifest[
        "scenario"
    ] == (
        "controlled_synthetic_"
        "concept_drift"
    )
    assert (
        manifest["synthetic_features"]
        is False
    )
    assert (
        manifest["synthetic_labels"]
        is True
    )
    assert (
        manifest[
            "selected_labels_flipped"
        ]
        == int(
            audit[
                "concept_drift_applied"
            ].sum()
        )
    )

    with manifest_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        persisted_manifest = (
            json.load(file)
        )

    assert persisted_manifest == manifest


def test_prepare_scenario_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    (
        simulation_file,
        _,
        audit_file,
    ) = configure_paths(
        monkeypatch,
        tmp_path,
    )

    source = build_source_data()

    source.to_csv(
        simulation_file,
        index=False,
    )
    first_manifest = (
        concept_drift.prepare_scenario(
            max_days=3,
            batch_size=10,
            drift_start_day=2,
            ramp_days=1,
            post_drift_rate=0.5,
            random_state=42,
        )
    )
    first_audit = pd.read_csv(
        audit_file
    )

    source.to_csv(
        simulation_file,
        index=False,
    )
    second_manifest = (
        concept_drift.prepare_scenario(
            max_days=3,
            batch_size=10,
            drift_start_day=2,
            ramp_days=1,
            post_drift_rate=0.5,
            random_state=42,
        )
    )
    second_audit = pd.read_csv(
        audit_file
    )

    assert (
        first_manifest[
            "ordered_customer_id_sha256"
        ]
        == second_manifest[
            "ordered_customer_id_sha256"
        ]
    )
    assert (
        first_manifest[
            "selected_effective_label_sha256"
        ]
        == second_manifest[
            "selected_effective_label_sha256"
        ]
    )

    pd.testing.assert_frame_equal(
        first_audit,
        second_audit,
    )