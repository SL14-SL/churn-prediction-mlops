from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
)

from src.configs.loader import load_config
from src.inference.decision import (
    DecisionConfig,
    DecisionEngine,
)
from src.monitoring.performance import (
    compute_business_metrics,
)

from scripts.evaluate_tuning_candidates import (
    evaluate_at_threshold,
    load_data,
    select_threshold,
)

ENV_CFG = load_config()

DECISION_CONFIG = (
    DecisionConfig.from_config(
        ENV_CFG
    )
)

MODEL_PARAMETERS = {
    "learning_rate": (
        0.061428810158596955
    ),
    "max_depth": 2,
    "max_features": "sqrt",
    "min_samples_leaf": 9,
    "min_samples_split": 29,
    "n_estimators": 234,
    "subsample": (
        0.9475135022264298
    ),
}

NUMERIC_ENGINEERED_FEATURES = {
    "is_new_customer",
}

ENGINEERED_PREFIXES = (
    "tenure_group_",
)

FEATURE_GROUPS = {
    "original_features": {
        "columns": set(),
        "prefixes": (),
    },
    "new_customer_only": {
        "columns": {
            "is_new_customer",
        },
        "prefixes": (),
    },
    "tenure_group_only": {
        "columns": set(),
        "prefixes": (
            "tenure_group_",
        ),
    },
    "tenure_features": {
        "columns": {
            "is_new_customer",
        },
        "prefixes": (
            "tenure_group_",
        ),
    },
}


def is_engineered_feature(
    column: str,
) -> bool:
    return (
        column
        in NUMERIC_ENGINEERED_FEATURES
        or column.startswith(
            ENGINEERED_PREFIXES
        )
    )


def select_group_columns(
    columns: list[str],
    *,
    configured_columns: set[str],
    configured_prefixes: tuple[str, ...],
) -> list[str]:
    """
    Select a feature group while preserving the original column order.
    """
    return [
        column
        for column in columns
        if (
            not is_engineered_feature(
                column
            )
            or column
            in configured_columns
            or column.startswith(
                configured_prefixes
            )
        )
    ]


def main() -> None:
    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = load_data()

    results = []

    for group_name, group in (
        FEATURE_GROUPS.items()
    ):
        selected_columns = (
            select_group_columns(
                list(X_train.columns),
                configured_columns=(
                    group["columns"]
                ),
                configured_prefixes=(
                    group["prefixes"]
                ),
            )
        )

        model = (
            GradientBoostingClassifier(
                **MODEL_PARAMETERS,
                random_state=42,
            )
        )

        model.fit(
            X_train[selected_columns],
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_validation[
                    selected_columns
                ]
            )[:, 1]
        )

        threshold = select_threshold(
            y_validation,
            probabilities,
        )

        metrics = (
            evaluate_at_threshold(
                model_name=group_name,
                strategy="maximum_f1",
                threshold=threshold,
                y_true=y_validation,
                y_probability=probabilities,
            )
        )

        metrics["n_features"] = len(
            selected_columns
        )
        decision_engine = (
            DecisionEngine(
                DECISION_CONFIG
            )
        )

        decisions = (
            decision_engine.decide_batch(
                probabilities.tolist()
            )
        )

        business_df = pd.DataFrame(
            {
                "Churn": (
                    y_validation
                    .to_numpy()
                ),
                "churn_probability": (
                    probabilities
                ),
                "action": [
                    decision["action"]
                    for decision in decisions
                ],
                "customer_value": [
                    decision[
                        "customer_value"
                    ]
                    for decision in decisions
                ],
            }
        )

        business_metrics = (
            compute_business_metrics(
                business_df,
                customer_value=(
                    DECISION_CONFIG
                    .customer_value
                ),
                cost_contact=(
                    DECISION_CONFIG
                    .cost_contact
                ),
                cost_discount=(
                    DECISION_CONFIG
                    .cost_discount
                ),
                contact_uplift=(
                    DECISION_CONFIG
                    .contact_uplift
                ),
                discount_uplift=(
                    DECISION_CONFIG
                    .discount_uplift
                ),
            )
        )

        metrics.update(
            {
                "business_expected_profit": (
                    business_metrics[
                        "expected_profit"
                    ]
                ),
                "business_realized_profit": (
                    business_metrics[
                        "realized_profit"
                    ]
                ),
                "business_profit_per_action": (
                    business_metrics[
                        "realized_profit_per_action"
                    ]
                ),
                "business_intervention_rate": (
                    business_metrics[
                        "intervention_rate"
                    ]
                ),
                "business_intervention_cost": (
                    business_metrics[
                        "total_intervention_cost"
                    ]
                ),
                "business_intervened_churners": (
                    business_metrics[
                        "actual_intervened_churners"
                    ]
                ),
            }
        )

        results.append(metrics)

    result_df = pd.DataFrame(
        results
    ).sort_values(
        [
            "average_precision",
            "f1_score",
        ],
        ascending=False,
    )

    output_path = Path(
        "results/churn_tuning/"
        "feature_group_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    columns = [
        "model",
        "n_features",
        "decision_threshold",
        "precision",
        "recall",
        "f1_score",
        "average_precision",
        "roc_auc",
        "brier_score",
        "action_rate",
        "business_expected_profit",
        "business_realized_profit",
        "business_profit_per_action",
        "business_intervention_rate",
        "business_intervention_cost",
        "business_intervened_churners",
    ]

    print(
        result_df[
            columns
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()