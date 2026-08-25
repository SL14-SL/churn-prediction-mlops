from unittest.mock import (
    MagicMock,
)

from src.monitoring import trigger
from src.monitoring.retraining_policy import (
    RetrainingAction,
    RetrainingDecision,
)


def test_evaluate_retraining_uses_collector_and_policy(
    monkeypatch,
):
    signals = MagicMock()

    decision = RetrainingDecision(
        action=(
            RetrainingAction.SKIP
        ),
        decision_id="retrain-test",
        reasons=(
            "No persistent signal.",
        ),
        trigger_types=(),
        evidence={},
    )

    collect = MagicMock(
        return_value=signals
    )
    decide = MagicMock(
        return_value=decision
    )

    monkeypatch.setattr(
        trigger,
        "collect_retraining_signals",
        collect,
    )
    monkeypatch.setattr(
        trigger,
        "decide_retraining",
        decide,
    )

    result = (
        trigger.evaluate_retraining()
    )

    assert result is decision

    collect.assert_called_once_with(
        evaluated_at=None
    )
    decide.assert_called_once_with(
        signals
    )