from dataclasses import replace

from src.monitoring.retraining_policy import (
    RetrainingAction,
    RetrainingSignals,
    decide_retraining,
)


def valid_signals() -> RetrainingSignals:
    return RetrainingSignals(
        dataset_version="dataset-v12",
        new_training_rows=1_000,
        minimum_training_rows=500,
        data_quality_ok=True,
        performance_degraded=False,
        feature_drift_persistent=False,
        cooldown_active=False,
        budget_available=True,
        performance_window_end="2026-08-13T00:00:00Z",
        drift_window_end="2026-08-13T00:00:00Z",
    )


def test_new_data_alone_does_not_trigger_retraining():
    decision = decide_retraining(valid_signals())

    assert decision.action == RetrainingAction.SKIP
    assert decision.should_train is False
    assert decision.trigger_types == ()


def test_persistent_performance_loss_triggers_candidate():
    signals = replace(
        valid_signals(),
        performance_degraded=True,
        performance_reason="Two bad performance windows.",
    )

    decision = decide_retraining(signals)

    assert (
        decision.action
        == RetrainingAction.TRAIN_CANDIDATE
    )
    assert decision.should_train is True
    assert decision.trigger_types == (
        "performance_degradation",
    )


def test_persistent_feature_drift_triggers_candidate():
    signals = replace(
        valid_signals(),
        feature_drift_persistent=True,
    )

    decision = decide_retraining(signals)

    assert (
        decision.action
        == RetrainingAction.TRAIN_CANDIDATE
    )
    assert decision.trigger_types == (
        "feature_drift",
    )


def test_insufficient_new_rows_prevents_retraining():
    signals = replace(
        valid_signals(),
        new_training_rows=499,
        performance_degraded=True,
    )

    decision = decide_retraining(signals)

    assert decision.action == RetrainingAction.SKIP
    assert "499/500" in decision.reasons[0]


def test_cooldown_prevents_retraining():
    signals = replace(
        valid_signals(),
        cooldown_active=True,
        performance_degraded=True,
    )

    decision = decide_retraining(signals)

    assert decision.action == RetrainingAction.SKIP
    assert decision.reasons == (
        "Retraining cooldown is active.",
    )


def test_data_quality_failure_blocks_retraining():
    signals = replace(
        valid_signals(),
        data_quality_ok=False,
        performance_degraded=True,
        data_quality_reason="Schema validation failed.",
    )

    decision = decide_retraining(signals)

    assert decision.action == RetrainingAction.BLOCK
    assert decision.trigger_types == ("data_quality",)
    assert decision.reasons == (
        "Schema validation failed.",
    )


def test_missing_budget_blocks_retraining():
    signals = replace(
        valid_signals(),
        budget_available=False,
        performance_degraded=True,
    )

    decision = decide_retraining(signals)

    assert decision.action == RetrainingAction.BLOCK
    assert decision.trigger_types == ("budget",)


def test_same_evidence_produces_same_decision_id():
    signals = valid_signals()

    first = decide_retraining(signals)
    second = decide_retraining(signals)

    assert first.decision_id == second.decision_id


def test_new_monitoring_window_changes_decision_id():
    first = decide_retraining(valid_signals())

    changed_signals = replace(
        valid_signals(),
        performance_window_end="2026-08-14T00:00:00Z",
    )
    second = decide_retraining(changed_signals)

    assert first.decision_id != second.decision_id

def test_scheduled_refresh_triggers_candidate():
    signals = replace(
        valid_signals(),
        scheduled_retraining_due=True,
        days_since_last_training=7.0,
    )

    decision = decide_retraining(signals)

    assert (
        decision.action
        == RetrainingAction.TRAIN_CANDIDATE
    )
    assert decision.trigger_types == (
        "scheduled_refresh",
    )


def test_scheduled_refresh_still_requires_new_rows():
    signals = replace(
        valid_signals(),
        new_training_rows=0,
        scheduled_retraining_due=True,
        days_since_last_training=7.0,
    )

    decision = decide_retraining(signals)

    assert decision.action == (
        RetrainingAction.SKIP
    )
    assert (
        "Insufficient new training rows"
        in decision.reasons[0]
    )


def test_scheduled_refresh_respects_cooldown():
    signals = replace(
        valid_signals(),
        scheduled_retraining_due=True,
        cooldown_active=True,
    )

    decision = decide_retraining(signals)

    assert decision.action == (
        RetrainingAction.SKIP
    )
    assert decision.reasons == (
        "Retraining cooldown is active.",
    )