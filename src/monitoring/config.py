# src/monitoring/config.py
from src.configs.loader import load_config

def get_monitoring_config() -> dict:
    return load_config("monitoring.yaml")

def get_feature_drift_settings() -> dict:
    cfg = get_monitoring_config().get("feature_drift", {})
    return {
        "enabled": cfg.get("enabled", True),
        "numeric_features": cfg.get("numeric_features", []),
        "categorical_features": cfg.get("categorical_features", []),
        "min_samples": cfg.get("min_samples", 50),
        "p_value_threshold": cfg.get("p_value_threshold", 0.01),
        "stat_threshold": cfg.get("stat_threshold", 0.10),
    }

def get_data_quality_settings() -> dict:
    cfg = get_monitoring_config().get("data_quality", {})
    return {
        "enabled": cfg.get("enabled", True),
        "categorical_reference_features": cfg.get(
            "categorical_reference_features", []
        ),
        "persist_history": cfg.get("persist_history", False),
    }

def get_serving_settings() -> dict:
    cfg = get_monitoring_config().get("serving", {})
    return {
        "enabled": cfg.get("enabled", True),
        "metrics_endpoint_enabled": cfg.get("metrics_endpoint_enabled", True),
        "summary_endpoint_enabled": cfg.get("summary_endpoint_enabled", True),
        "summary_window_seconds": cfg.get("summary_window_seconds", 900),
        "track_paths": cfg.get("track_paths", ["/predict", "/health"]),
        "ignored_paths": cfg.get(
            "ignored_paths",
            ["/metrics", "/monitoring/summary", "/docs", "/openapi.json", "/redoc"],
        ),
    }

def get_business_settings() -> dict:
    cfg = get_monitoring_config().get("business", {})
    return {
        "enabled": cfg.get("enabled", True),
        "min_expected_profit": cfg.get("min_expected_profit", 0.0),
        "customer_value": cfg.get("customer_value", 100.0),
        "cost_contact": cfg.get("cost_contact", 2.0),
        "cost_discount": cfg.get("cost_discount", 10.0),
        "contact_uplift": cfg.get("contact_uplift", 0.1),
        "discount_uplift": cfg.get("discount_uplift", 0.3),
        "max_discount_budget": cfg.get("max_discount_budget"),
        "max_discount_rate": cfg.get("max_discount_rate"),
    }

def get_retraining_settings() -> dict:
    """
    Return normalized retraining-policy settings with operational defaults.

    Returns:
        Thresholds and windows controlling data availability, cooldown, scheduled
        retraining, drift persistence, model performance and business outcomes.
    """
    monitoring_cfg = (
        get_monitoring_config()
    )

    retraining_cfg = (
        monitoring_cfg.get(
            "retraining",
            {},
        )
    )

    drift_cfg = (
        retraining_cfg.get(
            "drift",
            {},
        )
    )

    retraining_performance_cfg = (
        retraining_cfg.get(
            "performance",
            {},
        )
    )

    classification_thresholds = (
        monitoring_cfg.get(
            "performance",
            {},
        ).get(
            "retrain_thresholds",
            {},
        )
    )

    return {
        "minimum_new_training_rows": int(
            retraining_cfg.get(
                "minimum_new_training_rows",
                100,
            )
        ),
        "maximum_new_training_rows": int(
            retraining_cfg.get(
                "maximum_new_training_rows",
                100_000,
            )
        ),
        "cooldown_hours": int(
            retraining_cfg.get(
                "cooldown_hours",
                168,
            )
        ),
        "scheduled_interval_hours": int(
            retraining_cfg.get(
                "scheduled_interval_hours",
                168,
            )
        ),
        "drift": {
            "lookback_days": int(
                drift_cfg.get(
                    "lookback_days",
                    14,
                )
            ),
            "consecutive_windows": int(
                drift_cfg.get(
                    "consecutive_windows",
                    2,
                )
            ),
        },
        "performance": {
            "consecutive_windows": int(
                retraining_performance_cfg.get(
                    "consecutive_windows",
                    2,
                )
            ),
            "minimum_samples": int(
                retraining_performance_cfg.get(
                    "minimum_samples",
                    20,
                )
            ),
            "min_f1": float(
                classification_thresholds.get(
                    "min_f1",
                    0.60,
                )
            ),
            "min_recall": float(
                classification_thresholds.get(
                    "min_recall",
                    0.65,
                )
            ),
            "min_roc_auc": float(
                classification_thresholds.get(
                    "min_roc_auc",
                    0.75,
                )
            ),
            "max_brier_score": float(
                classification_thresholds.get(
                    "max_brier_score",
                    0.22,
                )
            ),
        },
    }