from __future__ import annotations

import json
from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.configs.loader import (
    get_path,
    load_config,
)
from src.training.train import (
    normalize_feature_dtypes,
)


TRAIN_CFG = load_config("training.yaml")
CANDIDATES = {
    "baseline_gradient_boosting": {
        "model_type": (
            "gradient_boosting"
        ),
        "params": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
        },
    },
    "tuned_gradient_boosting": {
        "model_type": (
            "gradient_boosting"
        ),
        "params": {
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
        },
    },
    "tuned_xgboost": {
        "model_type": "xgboost",
        "params": {
            "colsample_bytree": (
                0.8460028906796679
            ),
            "gamma": (
                0.49502692505213164
            ),
            "learning_rate": (
                0.015214353605950144
            ),
            "max_depth": 4,
            "min_child_weight": 10,
            "n_estimators": 535,
            "reg_alpha": (
                0.002439498256807585
            ),
            "reg_lambda": (
                8.970458772060821
            ),
            "subsample": (
                0.6978174660047101
            ),
        },
    },
}


def map_target(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
            }
        )
        .fillna(0)
        .astype(int)
    )


def load_data():
    split_path = get_path("splits")

    train = pd.read_parquet(
        f"{split_path}/train.parquet"
    )
    validation = pd.read_parquet(
        f"{split_path}/val.parquet"
    )

    target = (
        TRAIN_CFG["data"]["target_column"]
        .lower()
        .replace(" ", "_")
    )

    drop_columns = [
        target,
        *TRAIN_CFG.get(
            "features",
            {},
        ).get(
            "drop_columns",
            [],
        ),
    ]

    X_train = normalize_feature_dtypes(
        train.drop(
            columns=drop_columns,
            errors="ignore",
        )
    )
    X_validation = (
        normalize_feature_dtypes(
            validation.drop(
                columns=drop_columns,
                errors="ignore",
            )
        )
    )

    y_train = map_target(
        train[target]
    )
    y_validation = map_target(
        validation[target]
    )

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
    )


def select_threshold(
    y_true,
    y_probability,
    *,
    minimum_recall: float | None = None,
) -> float:
    rows = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):
        prediction = (
            y_probability >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "precision": (
                    precision_score(
                        y_true,
                        prediction,
                        zero_division=0,
                    )
                ),
                "recall": recall_score(
                    y_true,
                    prediction,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    prediction,
                    zero_division=0,
                ),
            }
        )

    scores = pd.DataFrame(rows)

    if minimum_recall is None:
        selected = scores.sort_values(
            [
                "f1",
                "precision",
            ],
            ascending=False,
        ).iloc[0]
    else:
        eligible = scores[
            scores["recall"]
            >= minimum_recall
        ]

        if eligible.empty:
            raise RuntimeError(
                "No threshold satisfies the "
                "minimum recall."
            )

        selected = eligible.sort_values(
            [
                "precision",
                "f1",
            ],
            ascending=False,
        ).iloc[0]

    return float(
        selected["threshold"]
    )


def evaluate_at_threshold(
    *,
    model_name: str,
    strategy: str,
    threshold: float,
    y_true,
    y_probability,
) -> dict:
    prediction = (
        y_probability >= threshold
    ).astype(int)

    return {
        "model": model_name,
        "threshold_strategy": strategy,
        "decision_threshold": (
            threshold
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "action_rate": float(
            prediction.mean()
        ),
        "average_precision": (
            average_precision_score(
                y_true,
                y_probability,
            )
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "brier_score": brier_score_loss(
            y_true,
            y_probability,
        ),
    }

def build_candidate_model(
    candidate: dict,
):
    model_type = candidate[
        "model_type"
    ]
    parameters = candidate[
        "params"
    ]

    if model_type == "gradient_boosting":
        return (
            GradientBoostingClassifier(
                **parameters,
                random_state=42,
            )
        )

    if model_type == "xgboost":
        return xgb.XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported model type: "
        f"{model_type}"
    )


def main() -> None:
    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = load_data()

    results = []

    for name, candidate in (
        CANDIDATES.items()
    ):
        model = build_candidate_model(
            candidate
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        f1_threshold = select_threshold(
            y_validation,
            probabilities,
        )

        recall_threshold = (
            select_threshold(
                y_validation,
                probabilities,
                minimum_recall=0.65,
            )
        )

        f1_result = (
            evaluate_at_threshold(
                model_name=name,
                strategy="maximum_f1",
                threshold=f1_threshold,
                y_true=y_validation,
                y_probability=(
                    probabilities
                ),
            )
        )

        f1_result["model_type"] = (
            candidate["model_type"]
        )
        f1_result["n_features"] = int(
            X_train.shape[1]
        )

        results.append(
            f1_result
        )

        recall_result = (
            evaluate_at_threshold(
                model_name=name,
                strategy=(
                    "maximum_precision_at_"
                    "recall_0.65"
                ),
                threshold=(
                    recall_threshold
                ),
                y_true=y_validation,
                y_probability=(
                    probabilities
                ),
            )
        )

        recall_result["model_type"] = (
            candidate["model_type"]
        )
        recall_result["n_features"] = int(
            X_train.shape[1]
        )

        results.append(
            recall_result
        )

    result_df = pd.DataFrame(
        results
    )

    output_directory = Path(
        "results/churn_tuning"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_directory
        / "validation_comparison.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    print(
        json.dumps(
            result_df.to_dict(
                orient="records"
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()