import io
from datetime import (
    datetime,
    timezone,
)

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from fastapi.responses import (
    StreamingResponse,
)

from src.api.dependencies import (
    get_api_key,
)
from src.api.schema import (
    CampaignSimulationRequest,
    PrioritizeRequest,
    PredictionResponse,
)
from src.api.services import (
    attach_customer_ids,
    compute_business_kpis,
    prioritize_results,
    run_prediction_pipeline,
    simulate_campaign,
)
from src.api.serving_state import (
    MODEL_NAME,
    TRAIN_CFG,
    get_data_quality_reference_categories,
    require_active_serving_bundle,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    tags=["business"],
    dependencies=[
        Depends(get_api_key),
    ],
)

@router.post(
    "/prioritize",
    response_model=PredictionResponse,
)
def prioritize(payload: PrioritizeRequest):
    """
    Score and rank customers by expected retention value.

    Returns:
        Prioritized churn decisions and aggregated business KPIs.

    Raises:
        HTTPException: If scoring or prioritization fails.
    """
    bundle = require_active_serving_bundle()

    try:
        output = run_prediction_pipeline(
            payload=payload,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            dq_reference_categories=get_data_quality_reference_categories(),
            decision_threshold=bundle.decision_threshold,
        )

        enriched = attach_customer_ids(payload.inputs, output["results"])

        prioritized = prioritize_results(
            enriched,
            top_n=payload.top_n,
            min_expected_value=payload.min_expected_value,
        )        

        business_kpis = compute_business_kpis(prioritized)

        return {
            "predictions": prioritized,
            "status": "success",
            "metadata": {
                "rows": len(prioritized),
                "total_input_rows": len(payload.inputs),
                "top_n": payload.top_n,
                "business_kpis": business_kpis,
                "min_expected_value": payload.min_expected_value,
                "model_name": MODEL_NAME,
                "serving_alias": bundle.serving_alias,
                "request_id": output["request_id"],
                "timing_ms": output["timings"],
                "data_quality": output["dq_summary"],
            },
        }

    except Exception as e:
        logger.exception("Prioritization failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/prioritize/export",
)
def export_prioritized(payload: PrioritizeRequest):
    """
    Export prioritized customers as CSV.

    Uses the same pipeline as /prioritize but returns a downloadable CSV file.
    """
    bundle = require_active_serving_bundle()

    try:
        # 1. Run shared pipeline
        output = run_prediction_pipeline(
            payload=payload,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            dq_reference_categories=get_data_quality_reference_categories(),
            decision_threshold=bundle.decision_threshold,
        )

        # 2. Attach IDs + prioritize
        enriched = attach_customer_ids(payload.inputs, output["results"])
        prioritized = prioritize_results(
            enriched,
            top_n=payload.top_n,
            min_expected_value=payload.min_expected_value,
        )

        # 3. Convert to DataFrame
        df = pd.DataFrame(prioritized)

        # Optional: nicer column order
        preferred_cols = [
            "customer_id",
            "churn_probability",
            "customer_value",
            "action",
            "expected_value",
        ]

        cols = [c for c in preferred_cols if c in df.columns]
        df = df[cols]

        # 4. Convert to CSV
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        # 5. Filename
        date_str = datetime.now(timezone.utc).strftime("%y%m%d")

        filename = f"{date_str}_prioritized_customers.csv"

        # 6. Return as file download
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            },
        )

    except Exception as e:
        logger.exception("CSV export failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/campaign/simulate",
)
def simulate_retention_campaign(payload: CampaignSimulationRequest):
    """
    Simulate a retention campaign based on prioritized churn decisions.

    Returns campaign-level business impact metrics:
    - total expected value
    - action counts
    - targeted customers
    - actionable customers
    """
    bundle = require_active_serving_bundle()

    try:
        output = run_prediction_pipeline(
            payload=payload,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            dq_reference_categories=get_data_quality_reference_categories(),
            decision_threshold=bundle.decision_threshold,
        )

        enriched = attach_customer_ids(payload.inputs, output["results"])

        prioritized = prioritize_results(
            enriched,
            top_n=payload.top_n,
            min_expected_value=payload.min_expected_value,
        )

        simulation = simulate_campaign(prioritized)

        return {
            "status": "success",
            "campaign": {
                "name": payload.campaign_name or "retention_campaign",
                "top_n": payload.top_n,
                "min_expected_value": payload.min_expected_value,
                **simulation,
            },
            "metadata": {
                "total_input_rows": len(payload.inputs),
                "selected_rows": len(prioritized),
                "model_name": MODEL_NAME,
                "serving_alias": bundle.serving_alias,
                "request_id": output["request_id"],
                "timing_ms": output["timings"],
                "data_quality": output["dq_summary"],
            },
        }

    except Exception as e:
        logger.exception("Campaign simulation failed")
        raise HTTPException(status_code=500, detail=str(e))