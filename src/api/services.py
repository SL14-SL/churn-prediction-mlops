import time
import os
from uuid import uuid4

from src.inference.adapters import request_to_dataframe
from src.inference.pipeline import (
    validate_prediction_input,
    align_features_for_model,
    predict_and_decide,
)
from src.data.features.build_features import build_features
from src.monitoring.data_quality import log_data_quality_runtime


def ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def run_prediction_pipeline(
    *,
    payload,
    model,
    model_type: str,
    feature_schema: dict | None,
    train_cfg: dict,
    dq_reference_categories: dict,
    decision_threshold: float = 0.5,
):
    """
    Execute the active model and decision pipeline for validated customer rows.

    Returns:
        Prediction and decision results together with serving lineage and timing.
    """
    request_started = time.perf_counter()
    timings: dict[str, float] = {}
    request_id = (
        payload.context
        or {}
    ).get(
        "request_id",
        str(uuid4()),
    )
    environment = os.getenv("APP_ENV", "dev")

    t = time.perf_counter()
    input_df = request_to_dataframe(payload.inputs)
    timings["request_to_dataframe"] = ms_since(t)

    t = time.perf_counter()
    validated_df = validate_prediction_input(input_df)
    timings["validate_input"] = ms_since(t)

    t = time.perf_counter()
    try:
        dq_summary = log_data_quality_runtime(
            validated_df,
            reference_categories=dq_reference_categories,
        )
    except Exception as dq_error:
        dq_summary = {"status": "error", "message": str(dq_error)}
    timings["data_quality"] = ms_since(t)

    t = time.perf_counter()
    processed_df = build_features(validated_df, config=train_cfg)
    timings["preprocessing"] = ms_since(t)

    t = time.perf_counter()
    final_df = align_features_for_model(
        processed_df=processed_df,
        model=model,
        model_type=model_type,
        feature_schema=feature_schema,
    )
    timings["alignment"] = ms_since(t)

    t = time.perf_counter()
    results = predict_and_decide(
        input_df=final_df,
        model=model,
        raw_input_df=validated_df,
        decision_threshold=decision_threshold,
    )
    timings["inference"] = ms_since(t)
    timings["total"] = ms_since(request_started)

    return {
        "results": results,
        "timings": timings,
        "dq_summary": dq_summary,
        "request_id": request_id,
        "environment": environment,
    }


def attach_customer_ids(inputs: list[dict], results: list[dict]) -> list[dict]:
    """Attach request customer identifiers to model and decision results."""
    enriched = []

    for original_input, result in zip(inputs, results):
        enriched.append({
            **result,
            "customer_id": original_input.get("customerID")
            or original_input.get("customer_id")
            or original_input.get("customerid"),
        })

    return enriched


def prioritize_results(
    results: list[dict],
    top_n: int | None = None,
    min_expected_value: float | None = None,
) -> list[dict]:
    """Sort customer decisions by descending expected retention value."""
    prioritized = results

    if min_expected_value is not None:
        prioritized = [
            r for r in prioritized
            if float(r.get("expected_value") or 0.0) >= min_expected_value
        ]

    prioritized = sorted(
        prioritized,
        key=lambda x: x.get("expected_value", 0.0),
        reverse=True,
    )

    if top_n is not None:
        prioritized = prioritized[:top_n]

    return prioritized


def compute_business_kpis(results: list[dict]) -> dict:
    """Aggregate customer-level decisions into campaign business KPIs."""
    total_expected_value = sum(float(r.get("expected_value") or 0.0) for r in results)

    return {
        "total_expected_value": round(total_expected_value, 2),
        "avg_expected_value": round(total_expected_value / len(results), 2)
        if results else 0.0,
        "total_customer_value": round(
            sum(float(r.get("customer_value") or 0.0) for r in results),
            2,
        ),
        "discounts_selected": sum(
            1 for r in results if r.get("action") == "offer_discount"
        ),
        "emails_selected": sum(
            1 for r in results if r.get("action") == "send_email"
        ),
        "no_action_selected": sum(
            1 for r in results if r.get("action") == "no_action"
        ),
    }

def simulate_campaign(results: list[dict]) -> dict:
    """Simulate campaign cost, retained value and net expected value."""
    total_expected_value = sum(
        float(r.get("expected_value") or 0.0) for r in results
    )
    total_customer_value = sum(
        float(r.get("customer_value") or 0.0) for r in results
    )

    action_counts = {
        "offer_discount": sum(1 for r in results if r.get("action") == "offer_discount"),
        "send_email": sum(1 for r in results if r.get("action") == "send_email"),
        "no_action": sum(1 for r in results if r.get("action") == "no_action"),
    }

    actionable_results = [
        r for r in results if r.get("action") != "no_action"
    ]

    return {
        "targeted_customers": len(results),
        "actionable_customers": len(actionable_results),
        "total_expected_value": round(total_expected_value, 2),
        "avg_expected_value": round(total_expected_value / len(results), 2)
        if results else 0.0,
        "total_customer_value": round(total_customer_value, 2),
        "action_counts": action_counts,
    }