from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

from src.configs.loader import get_path, load_config
from src.training.train import normalize_feature_dtypes


TRAIN_CFG = load_config("training.yaml")


def robust_map(series: pd.Series) -> pd.Series:
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


def load_tuning_data() -> tuple[pd.DataFrame, pd.Series]:
    split_path = get_path("splits")
    train_file = f"{split_path}/train.parquet"

    df_train = pd.read_parquet(train_file)

    target_column = (
        TRAIN_CFG["data"]["target_column"]
        .lower()
        .replace(" ", "_")
    )

    drop_columns = [
        target_column,
        *TRAIN_CFG.get(
            "features",
            {},
        ).get(
            "drop_columns",
            [],
        ),
    ]

    X_train = normalize_feature_dtypes(
        df_train.drop(
            columns=drop_columns,
            errors="ignore",
        )
    )
    y_train = robust_map(
        df_train[target_column]
    )

    return X_train, y_train


def build_parameter_distributions() -> dict:
    return {
        "n_estimators": randint(100, 601),
        "learning_rate": loguniform(
            0.01,
            0.20,
        ),
        "max_depth": randint(1, 6),
        "min_samples_split": randint(
            2,
            31,
        ),
        "min_samples_leaf": randint(
            1,
            31,
        ),
        "subsample": uniform(
            0.65,
            0.35,
        ),
        "max_features": [
            None,
            "sqrt",
            "log2",
            0.6,
            0.8,
        ],
    }


def tune_model(
    *,
    n_iter: int,
    random_state: int,
) -> RandomizedSearchCV:
    X_train, y_train = load_tuning_data()

    model = GradientBoostingClassifier(
        random_state=random_state,
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=(
            build_parameter_distributions()
        ),
        n_iter=n_iter,
        scoring={
            "average_precision": (
                "average_precision"
            ),
            "roc_auc": "roc_auc",
            "f1": "f1",
            "neg_brier_score": (
                "neg_brier_score"
            ),
        },
        refit="average_precision",
        cv=cross_validation,
        random_state=random_state,
        n_jobs=-1,
        verbose=2,
        return_train_score=False,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


def save_results(
    search: RandomizedSearchCV,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_parameters = {
        key: (
            value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in (
            search.best_params_.items()
        )
    }

    summary = {
        "best_cv_average_precision": (
            float(search.best_score_)
        ),
        "best_params": best_parameters,
    }

    with (
        output_directory
        / "best_parameters.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    results = pd.DataFrame(
        search.cv_results_
    )

    columns = [
        "rank_test_average_precision",
        "mean_test_average_precision",
        "std_test_average_precision",
        "mean_test_roc_auc",
        "mean_test_f1",
        "mean_test_neg_brier_score",
        "params",
    ]

    results[columns].sort_values(
        "rank_test_average_precision"
    ).to_csv(
        output_directory
        / "tuning_results.csv",
        index=False,
    )

    joblib.dump(
        search.best_estimator_,
        output_directory
        / "best_estimator.joblib",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-iter",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/churn_tuning"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    search = tune_model(
        n_iter=args.n_iter,
        random_state=args.random_state,
    )

    save_results(
        search,
        args.output_dir,
    )


if __name__ == "__main__":
    main()