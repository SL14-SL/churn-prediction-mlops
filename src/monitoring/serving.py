# src/monitoring/serving.py
from __future__ import annotations

import threading
from collections import Counter, deque
from time import time
from typing import Iterable

from prometheus_client import Counter as PromCounter
from prometheus_client import Histogram, Gauge

DEFAULT_LATENCY_BUCKETS_SECONDS = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    1.000,
    3.000,
    5.000,
)

SERVING_READY = Gauge(
    "api_serving_ready",
    (
        "Whether a complete serving bundle "
        "is currently active"
    ),
)

REQUEST_COUNT = PromCounter(
    "api_request_count_total",
    "Total number of API requests",
    ["method", "path"],
)

REQUEST_SUCCESS = PromCounter(
    "api_request_success_total",
    "Total number of successful API requests",
    ["method", "path"],
)

REQUEST_ERROR = PromCounter(
    "api_request_error_total",
    "Total number of failed API requests",
    ["method", "path"],
)

RESPONSE_STATUS = PromCounter(
    "api_response_status_total",
    "Total number of API responses by status code",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path"],
    buckets=DEFAULT_LATENCY_BUCKETS_SECONDS,
)

# Kleine In-Memory Summary für Menschen / lokalen Betrieb
_RECENT_EVENTS = deque(maxlen=5000)
_LOCK = threading.Lock()


def should_ignore_path(path: str, ignored_paths: Iterable[str] | None = None) -> bool:
    ignored = set(ignored_paths or [])
    return path in ignored


def normalize_path(path: str, track_paths: Iterable[str] | None = None) -> str:
    if not track_paths:
        return path

    allowed = set(track_paths)
    return path if path in allowed else "other"


def observe_request(
    *,
    method: str,
    path: str,
    status_code: int,
    latency_seconds: float,
) -> None:
    method = method.upper()
    status_code_str = str(status_code)

    REQUEST_COUNT.labels(method=method, path=path).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(latency_seconds)
    RESPONSE_STATUS.labels(
        method=method,
        path=path,
        status_code=status_code_str,
    ).inc()

    if 200 <= status_code < 300:
        REQUEST_SUCCESS.labels(method=method, path=path).inc()
    else:
        REQUEST_ERROR.labels(method=method, path=path).inc()

    with _LOCK:
        _RECENT_EVENTS.append(
            {
                "ts": time(),
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency_ms": round(latency_seconds * 1000, 2),
            }
        )


def get_summary(window_seconds: int = 900) -> dict:
    now = time()
    cutoff = now - window_seconds

    with _LOCK:
        events = [e for e in _RECENT_EVENTS if e["ts"] >= cutoff]

    total = len(events)
    successes = sum(1 for e in events if 200 <= e["status_code"] < 300)
    errors = total - successes

    status_counts = Counter(str(e["status_code"]) for e in events)
    path_counts = Counter(e["path"] for e in events)
    latencies = sorted(e["latency_ms"] for e in events)

    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        idx = int(round((len(values) - 1) * p))
        return round(values[idx], 2)

    return {
        "window_seconds": window_seconds,
        "requests_total": total,
        "success_total": successes,
        "error_total": errors,
        "success_rate": round(successes / total, 4) if total else None,
        "error_rate": round(errors / total, 4) if total else None,
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "status_codes": dict(status_counts),
        "paths": dict(path_counts),
    }

def set_serving_readiness(
    is_ready: bool,
) -> None:
    """
    Update the current serving-readiness state.
    """
    SERVING_READY.set(
        1 if is_ready else 0
    )