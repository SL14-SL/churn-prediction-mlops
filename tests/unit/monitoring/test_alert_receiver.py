from unittest.mock import MagicMock

import pytest
from fastapi.testclient import (
    TestClient,
)

from src.monitoring import (
    alert_receiver,
)


@pytest.fixture
def client():
    return TestClient(
        alert_receiver.app
    )


def alert_payload(
    *,
    status: str = "firing",
    severity: str = "warning",
):
    return {
        "status": status,
        "commonLabels": {
            "alertname": (
                "ChurnPredictionLatencyHigh"
            ),
            "service": (
                "churn-api"
            ),
            "severity": severity,
        },
        "alerts": [
            {
                "status": status,
                "labels": {
                    "alertname": (
                        "ChurnPredictionLatencyHigh"
                    ),
                    "service": (
                        "churn-api"
                    ),
                    "severity": severity,
                },
                "annotations": {
                    "summary": (
                        "Prediction latency is high"
                    ),
                    "description": (
                        "p95 exceeded one second."
                    ),
                },
            }
        ],
    }


def test_builds_grouped_notification():
    result = (
        alert_receiver
        .build_alert_notification(
            alert_payload()
        )
    )

    assert result["severity"] == (
        "warning"
    )
    assert result["status"] == "firing"
    assert result["alert_count"] == 1
    assert (
        "ChurnPredictionLatencyHigh"
        in result["title"]
    )
    assert (
        "Prediction latency is high"
        in result["message"]
    )


def test_critical_severity_wins():
    payload = alert_payload()

    payload["alerts"].append(
        {
            "labels": {
                "alertname": (
                    "ChurnAPIUnavailable"
                ),
                "service": (
                    "churn-api"
                ),
                "severity": "critical",
            },
            "annotations": {
                "summary": "API is down",
            },
        }
    )

    result = (
        alert_receiver
        .build_alert_notification(
            payload
        )
    )

    assert result["severity"] == (
        "critical"
    )


def test_receiver_sends_notification(
    client,
    monkeypatch,
):
    sender = MagicMock(
        return_value=True
    )

    monkeypatch.setattr(
        alert_receiver,
        "send_alert",
        sender,
    )

    response = client.post(
        "/alerts",
        json=alert_payload(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "delivered": True,
        "alert_status": "firing",
        "severity": "warning",
        "alert_count": 1,
    }

    sender.assert_called_once()


def test_receiver_accepts_failed_delivery(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        alert_receiver,
        "send_alert",
        MagicMock(
            return_value=False
        ),
    )

    response = client.post(
        "/alerts",
        json=alert_payload(),
    )

    assert response.status_code == 200
    assert (
        response.json()["delivered"]
        is False
    )


def test_receiver_rejects_empty_alerts(
    client,
):
    response = client.post(
        "/alerts",
        json={
            "status": "firing",
            "alerts": [],
        },
    )

    assert response.status_code == 422