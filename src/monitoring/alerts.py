import json
import os
import urllib.request

from src.utils.logger import get_logger

logger = get_logger(__name__)


def send_slack_alert(title: str, message: str, severity: str = "warning") -> bool:
    """
    Send an operational alert to the configured Slack webhook.

    Returns:
        True when the alert was accepted, otherwise False.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    if not webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not configured. Skipping Slack alert.")
        return False

    payload = {
        "text": f"[{severity.upper()}] {title}\n{message}"
    }

    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if 200 <= response.status < 300:
                logger.info("Slack alert sent successfully.")
                return True

            logger.warning(f"Slack alert failed with status {response.status}")
            return False

    except Exception as exc:
        logger.error(f"Failed to send Slack alert: {exc}")
        return False


def send_alert(title: str, message: str, severity: str = "warning") -> bool:
    return send_slack_alert(title=title, message=message, severity=severity)