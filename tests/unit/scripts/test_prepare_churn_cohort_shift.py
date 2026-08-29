import pandas as pd
import pytest

from scripts.prepare_churn_cohort_shift import (
    calculate_risk_score,
    target_high_risk_rate,
)


@pytest.mark.parametrize(
    (
        "day",
        "expected_rate",
    ),
    [
        (1, 0.20),
        (5, 0.20),
        (6, 0.40),
        (7, 0.60),
        (8, 0.80),
        (15, 0.80),
    ],
)
def test_target_high_risk_rate(
    day,
    expected_rate,
):
    result = target_high_risk_rate(
        day=day,
        drift_start_day=6,
        ramp_days=3,
        baseline_rate=0.20,
        post_drift_rate=0.80,
    )

    assert result == pytest.approx(
        expected_rate
    )


def test_calculate_risk_score_uses_existing_customer_attributes():
    dataframe = pd.DataFrame(
        [
            {
                "tenure": 4,
                "Contract": (
                    "Month-to-month"
                ),
                "InternetService": (
                    "Fiber optic"
                ),
                "PaymentMethod": (
                    "Electronic check"
                ),
                "PaperlessBilling": "Yes",
            },
            {
                "tenure": 60,
                "Contract": "Two year",
                "InternetService": "DSL",
                "PaymentMethod": (
                    "Mailed check"
                ),
                "PaperlessBilling": "No",
            },
        ]
    )

    result = calculate_risk_score(
        dataframe
    )

    assert result.tolist() == [
        5,
        0,
    ]