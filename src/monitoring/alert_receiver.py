from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.monitoring.alerts import (
    send_alert,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

app = FastAPI(
    title="Churn Alert Receiver",
)


@app.get("/livez")
def livez() -> dict[str, str]:
    return {
        "status": "alive",
        "service": "alert-receiver",
    }


def build_alert_notification(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one grouped Alertmanager webhook into a notification.
    """
    alerts = payload.get(
        "alerts"
    )

    if (
        not isinstance(alerts, list)
        or not alerts
    ):
        raise ValueError(
            "Alertmanager payload contains "
            "no alerts."
        )

    group_status = str(
        payload.get(
            "status",
            "firing",
        )
    ).lower()

    severities = {
        str(
            alert.get(
                "labels",
                {},
            ).get(
                "severity",
                "warning",
            )
        ).lower()
        for alert in alerts
        if isinstance(alert, dict)
    }

    if "critical" in severities:
        severity = "critical"
    elif "warning" in severities:
        severity = "warning"
    else:
        severity = "info"

    common_labels = payload.get(
        "commonLabels",
        {},
    )

    group_name = (
        common_labels.get(
            "alertname"
        )
        if isinstance(
            common_labels,
            dict,
        )
        else None
    )

    if not group_name:
        first_alert = alerts[0]
        group_name = (
            first_alert.get(
                "labels",
                {},
            ).get(
                "alertname",
                "Churn alert",
            )
        )

    title = (
        f"[{group_status.upper()}] "
        f"{group_name}"
    )

    message_lines = []

    for alert in alerts:
        if not isinstance(alert, dict):
            continue

        labels = alert.get(
            "labels",
            {},
        )
        annotations = alert.get(
            "annotations",
            {},
        )

        alert_name = labels.get(
            "alertname",
            "UnnamedAlert",
        )
        service = labels.get(
            "service",
            "unknown",
        )
        summary = annotations.get(
            "summary",
            "No summary provided.",
        )
        description = annotations.get(
            "description",
            "",
        )

        message_lines.append(
            f"- {alert_name} | "
            f"service={service} | "
            f"{summary}"
        )

        if description:
            message_lines.append(
                f"  {description}"
            )

    message_lines.append(
        f"Alert count: {len(alerts)}"
    )

    return {
        "title": title,
        "message": "\n".join(
            message_lines
        ),
        "severity": severity,
        "status": group_status,
        "alert_count": len(alerts),
    }


@app.post("/alerts")
def receive_alerts(
    payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        notification = (
            build_alert_notification(
                payload
            )
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    delivered = send_alert(
        title=notification["title"],
        message=notification["message"],
        severity=notification["severity"],
    )

    if delivered:
        logger.info(
            "Alertmanager notification "
            "delivered | status=%s | "
            "severity=%s | alerts=%s",
            notification["status"],
            notification["severity"],
            notification["alert_count"],
        )
    else:
        logger.warning(
            "Alertmanager notification accepted "
            "but external delivery was skipped "
            "or failed | status=%s | "
            "severity=%s | alerts=%s",
            notification["status"],
            notification["severity"],
            notification["alert_count"],
        )

    return {
        "status": "accepted",
        "delivered": delivered,
        "alert_status": (
            notification["status"]
        ),
        "severity": (
            notification["severity"]
        ),
        "alert_count": (
            notification["alert_count"]
        ),
    }