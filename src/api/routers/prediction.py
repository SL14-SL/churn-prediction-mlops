import traceback

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from src.api.dependencies import get_api_key
from src.api.schema import (
    PredictionRequest,
    PredictionResponse
)
from src.api.services import (
    attach_customer_ids,
    run_prediction_pipeline,
)
from src.api.serving_state import (
    MODEL_NAME,
    TRAIN_CFG,
    get_data_quality_reference_categories,
    require_active_serving_bundle,
)
from src.data.features.build_features import (
    build_features,
)
from src.inference.adapters import (
    request_to_dataframe,
)
from src.inference.explain import (
    explain_single_prediction,
)
from src.inference.pipeline import (
    align_features_for_model,
    predict_and_decide,
    validate_prediction_input,
)
from src.monitoring.prediction_logger import (
    log_prediction,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)

router = APIRouter(
    tags=["prediction"],
    dependencies=[
        Depends(get_api_key),
    ],
)

@router.post("/explain")
def explain(payload: PredictionRequest, top_n: int = 5):
    """
    Return one churn prediction with its most influential feature contributions.

    The endpoint accepts exactly one customer and is intended for debugging,
    demonstrations and customer-level model interpretation.

    Raises:
        HTTPException: If the request contains more than one row or explanation
            generation fails.
    """
    bundle = require_active_serving_bundle()

    if len(payload.inputs) != 1:
        raise HTTPException(
            status_code=400,
            detail="Explain endpoint currently supports exactly one input row.",
        )

    try:
        input_df = request_to_dataframe(payload.inputs)
        validated_df = validate_prediction_input(input_df)
        processed_df = build_features(validated_df, config=TRAIN_CFG)

        final_df = align_features_for_model(
            processed_df=processed_df,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
        )

        prediction = predict_and_decide(
            input_df=final_df,
            model=bundle.model,
        )[0]

        explanation = explain_single_prediction(
            final_df=final_df,
            model=bundle.model,
            model_type=bundle.model_type,
            feature_schema=bundle.feature_schema,
            train_cfg=TRAIN_CFG,
            top_n=top_n,
        )

        return {
            "status": "success",
            "prediction": prediction,
            "top_reasons": explanation,
            "metadata": {
                "model_name": MODEL_NAME,
                "serving_alias": bundle.serving_alias,
                "model_version": bundle.model_version,
            },
        }

    except Exception as e:
        logger.error(f"Explanation failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(payload: PredictionRequest):
    """
    Predict churn risk and recommend a retention action for each customer.

    Predictions are logged with request, model and serving-release lineage.

    Returns:
        Customer-level churn decisions and operational request metadata.

    Raises:
        HTTPException: If validation, inference or response generation fails.
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

        results = attach_customer_ids(
            payload.inputs,
            output["results"],
        )

        for features, result in zip(payload.inputs, results):
            log_prediction(
                features,
                result["churn_probability"],
                model_alias=bundle.serving_alias,
                model_version=bundle.model_version,
                model_run_id=bundle.model_run_id,
                request_id=output["request_id"],
                environment=output["environment"],
                action=result["action"],
                expected_value=result["expected_value"],
                customer_value=result.get("customer_value"),
            )

        return {
            "predictions": results,
            "status": "success",
            "metadata": {
                "rows": len(results),
                "model_name": MODEL_NAME,
                "serving_alias": bundle.serving_alias,
                "request_id": output["request_id"],
                "timing_ms": output["timings"],
                "data_quality": output["dq_summary"],
                "release_id": bundle.release_id,
                "model_version": bundle.model_version,
                "model_run_id": bundle.model_run_id,
            },
        }

    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))
    