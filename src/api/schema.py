from typing import Any

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    """
    Generic prediction request contract for blueprint-ready APIs.

    Supports both single-row and batch inference by accepting a list of input records.
    Each record should contain the raw feature values expected by the model.

    Example:
    {
        "inputs": [
            {"tenure": 12, "monthly_charges": 70},
            {"tenure": 5, "monthly_charges": 50}
        ]
    }
    """
    inputs: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of input records for inference",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata, routing hints or execution context",
    )

    @model_validator(mode="after")
    def validate_inputs_not_empty(self):
        if not self.inputs:
            raise ValueError("inputs must not be empty")
        return self


class DecisionResult(BaseModel):
    """
    Structured decision output for a single prediction.

    This represents the business-facing output of the churn system,
    transforming raw model probabilities into actionable insights.

    Fields:
    - churn_probability:
        Model-estimated probability that the customer will churn (0.0–1.0)

    - segment:
        Risk bucket derived from probability thresholds
        (e.g. "high_risk", "medium_risk", "low_risk")

    - action:
        Recommended business action based on the segment
        (e.g. "offer_discount", "send_email", "no_action")

    - expected_value:
        Estimated monetary value of applying the action
        (simple expected value calculation based on business assumptions)
    """
    customer_id: str | None = None 
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    customer_value: float | None = Field(default=None)
    action: str = Field(..., description="Recommended business action")
    expected_value: float = Field(..., description="Estimated business value")


class PredictionResponse(BaseModel):
    """
    Churn prediction response contract.

    This schema returns decision-aware outputs instead of raw predictions.

    Supports batch inference:
    - Each input record produces one DecisionResult

    Fields:
    - predictions:
        List of decision results (one per input record)

    - status:
        Execution status indicator ("success" or "error")

    - metadata:
        Optional additional information such as:
        - model version
        - inference timestamp
        - threshold configuration
    """
    predictions: list[DecisionResult]

    status: str = Field(
        default="success",
        description="Execution status of the prediction request",
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata about the prediction run",
    )

class PrioritizeRequest(PredictionRequest):
    """Request containing customers to score and prioritize for retention."""
    top_n: int | None = Field(
        default=None,
        ge=1,
        description="Return only the top N customers sorted by expected value",
    )
    min_expected_value: float | None = Field(
        default=None,
        description="Only return customers with expected_value >= this threshold",
    )

class CampaignSimulationRequest(PrioritizeRequest):
    """Request parameters for simulating a retention campaign."""
    campaign_name: str | None = Field(
        default=None,
        description="Optional campaign name for reporting",
    )


class ServingRollbackRequest(BaseModel):
    """Request identifying the immutable serving release to reactivate."""
    release_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Immutable serving release "
            "to activate."
        ),
    )