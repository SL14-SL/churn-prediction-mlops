from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FuncFormatter,
    PercentFormatter,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


STATIC_COLOR = "#F05A40"
ADAPTIVE_COLOR = "#5865F2"
SCENARIO_COLOR = "#8C6BB1"
SHIFT_COLOR = "#F6C85F"
RETRAIN_COLOR = "#00A67E"
PROMOTION_COLOR = "#2CA02C"

COHORT_SHIFT_SCENARIO = (
    "controlled_real_cohort_shift"
)
CONCEPT_DRIFT_SCENARIO = (
    "controlled_synthetic_concept_drift"
)

DEFAULT_EXPERIMENT_DIRECTORY = Path(
    "results/churn_retraining_comparison/"
    "controlled_cohort_shift"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a controlled churn "
            "retraining comparison."
        )
    )
    parser.add_argument(
        "--experiment-directory",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIRECTORY,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--rolling-window-size",
        type=int,
        default=150,
    )
    return parser.parse_args()


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_history(
    experiment_directory: Path,
    branch: str,
) -> pd.DataFrame:
    path = (
        experiment_directory
        / branch
        / "performance_history.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Performance history not "
            f"found: {path}"
        )

    history = pd.read_parquet(path)

    required_columns = {
        "simulation_day",
        "f1_score",
        "business_realized_profit",
    }

    missing_columns = (
        required_columns
        - set(history.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}: "
            f"{sorted(missing_columns)}"
        )

    return (
        history
        .sort_values("simulation_day")
        .drop_duplicates(
            subset=["simulation_day"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def load_ground_truth(
    experiment_directory: Path,
    branch: str,
) -> pd.DataFrame:
    path = (
        experiment_directory
        / branch
        / "cumulative_ground_truth.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Ground truth not found: "
            f"{path}"
        )

    ground_truth = pd.read_csv(path)

    required_columns = {
        "prediction",
        "churn",
        "released_simulation_day",
    }

    missing_columns = (
        required_columns
        - set(ground_truth.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}: "
            f"{sorted(missing_columns)}"
        )

    ground_truth["prediction"] = (
        pd.to_numeric(
            ground_truth["prediction"],
            errors="coerce",
        )
    )
    ground_truth["churn"] = (
        pd.to_numeric(
            ground_truth["churn"],
            errors="coerce",
        )
    )
    ground_truth[
        "released_simulation_day"
    ] = pd.to_numeric(
        ground_truth[
            "released_simulation_day"
        ],
        errors="coerce",
    )

    sort_columns = [
        "released_simulation_day",
    ]

    if (
        "prediction_timestamp"
        in ground_truth.columns
    ):
        sort_columns.append(
            "prediction_timestamp"
        )

    return (
        ground_truth
        .dropna(
            subset=[
                "prediction",
                "churn",
                "released_simulation_day",
            ]
        )
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )


def build_rolling_history(
    history: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    window_size: int,
) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError(
            "Rolling window size must "
            "be positive."
        )

    rolling_history = (
        history.copy()
    )

    rolling_f1_values = []
    rolling_precision_values = []
    rolling_recall_values = []
    rolling_sample_counts = []

    for _, history_row in (
        rolling_history.iterrows()
    ):
        simulation_day = int(
            history_row[
                "simulation_day"
            ]
        )
        decision_threshold = float(
            history_row.get(
                "decision_threshold",
                0.5,
            )
        )

        available_labels = (
            ground_truth[
                ground_truth[
                    "released_simulation_day"
                ]
                <= simulation_day
            ]
            .tail(window_size)
        )

        if available_labels.empty:
            rolling_f1_values.append(
                float("nan")
            )
            rolling_precision_values.append(
                float("nan")
            )
            rolling_recall_values.append(
                float("nan")
            )
            rolling_sample_counts.append(0)
            continue

        y_true = (
            available_labels["churn"]
            .astype(int)
        )
        y_prediction = (
            available_labels["prediction"]
            >= decision_threshold
        ).astype(int)

        rolling_f1_values.append(
            f1_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        )
        rolling_precision_values.append(
            precision_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        )
        rolling_recall_values.append(
            recall_score(
                y_true,
                y_prediction,
                zero_division=0,
            )
        )
        rolling_sample_counts.append(
            len(available_labels)
        )

    rolling_history["f1_score"] = (
        rolling_f1_values
    )
    rolling_history["precision"] = (
        rolling_precision_values
    )
    rolling_history["recall"] = (
        rolling_recall_values
    )
    rolling_history[
        "rolling_n_samples"
    ] = rolling_sample_counts

    return rolling_history


def load_retraining_events(
    experiment_directory: Path,
) -> pd.DataFrame:
    path = (
        experiment_directory
        / "with_retraining"
        / "retraining_events.parquet"
    )

    if not path.exists():
        return pd.DataFrame()

    events = pd.read_parquet(path)

    if (
        "simulation_day"
        not in events.columns
    ):
        return pd.DataFrame()

    return (
        events
        .dropna(
            subset=["simulation_day"]
        )
        .sort_values("simulation_day")
        .reset_index(drop=True)
    )


def load_scenario(
    experiment_directory: Path,
) -> tuple[pd.DataFrame, dict]:
    path = (
        experiment_directory
        / "without_retraining"
        / "scenario_manifest.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Scenario manifest not "
            f"found: {path}"
        )

    manifest = load_json(path)

    if "daily_summary" not in manifest:
        raise ValueError(
            "Scenario manifest does not "
            "contain daily_summary."
        )

    daily_summary = pd.DataFrame(
        manifest["daily_summary"]
    )

    if (
        "simulation_day"
        not in daily_summary.columns
    ):
        raise ValueError(
            "Scenario daily summary does "
            "not contain simulation_day."
        )

    return daily_summary, manifest


def get_promotion_count(
    events: pd.DataFrame,
) -> int:
    if (
        events.empty
        or "champion_promoted"
        not in events.columns
    ):
        return 0

    return int(
        events[
            "champion_promoted"
        ].fillna(False).sum()
    )


def value_at_day(
    history: pd.DataFrame,
    *,
    day: int,
    column: str,
) -> float:
    distances = (
        history["simulation_day"]
        - day
    ).abs()

    row = history.loc[
        distances.idxmin()
    ]

    return float(row[column])


def add_experiment_markers(
    axes: list[plt.Axes],
    events: pd.DataFrame,
    adaptive_history: pd.DataFrame,
) -> None:
    if events.empty:
        return

    for _, event in events.iterrows():
        day = int(
            event["simulation_day"]
        )

        for axis in axes:
            axis.axvline(
                day,
                color=RETRAIN_COLOR,
                linewidth=1.4,
                linestyle=":",
                alpha=0.85,
                zorder=2,
            )

        f1_value = value_at_day(
            adaptive_history,
            day=day,
            column="f1_score",
        )
        profit_value = value_at_day(
            adaptive_history,
            day=day,
            column="business_profit_delta",
        )

        axes[0].scatter(
            day,
            f1_value,
            color=RETRAIN_COLOR,
            edgecolor="white",
            linewidth=0.8,
            marker="D",
            s=75,
            zorder=5,
        )
        axes[1].scatter(
            day,
            profit_value,
            color=RETRAIN_COLOR,
            edgecolor="white",
            linewidth=0.8,
            marker="D",
            s=75,
            zorder=5,
        )

        if bool(
            event.get(
                "champion_promoted",
                False,
            )
        ):
            axes[0].scatter(
                day,
                f1_value,
                color=PROMOTION_COLOR,
                edgecolor="white",
                linewidth=0.9,
                marker="*",
                s=180,
                zorder=6,
            )
            axes[1].scatter(
                day,
                profit_value,
                color=PROMOTION_COLOR,
                edgecolor="white",
                linewidth=0.9,
                marker="*",
                s=180,
                zorder=6,
            )


def histories_overlap(
    static_history: pd.DataFrame,
    adaptive_history: pd.DataFrame,
) -> bool:
    comparison = static_history[
        [
            "simulation_day",
            "f1_score",
            "business_realized_profit",
        ]
    ].merge(
        adaptive_history[
            [
                "simulation_day",
                "f1_score",
                "business_realized_profit",
            ]
        ],
        on="simulation_day",
        how="inner",
        suffixes=(
            "_static",
            "_adaptive",
        ),
    )

    if comparison.empty:
        return False

    f1_difference = (
        comparison[
            "f1_score_static"
        ]
        - comparison[
            "f1_score_adaptive"
        ]
    ).abs()

    profit_difference = (
        comparison[
            "business_realized_profit_static"
        ]
        - comparison[
            "business_realized_profit_adaptive"
        ]
    ).abs()

    return bool(
        (
            f1_difference.fillna(0)
            <= 1e-12
        ).all()
        and (
            profit_difference.fillna(0)
            <= 1e-9
        ).all()
    )


def build_summary(
    static_history: pd.DataFrame,
    adaptive_history: pd.DataFrame,
    events: pd.DataFrame,
    manifest: dict,
    *,
    post_shift_start: int,
) -> str:
    static_post_shift = static_history[
        static_history[
            "simulation_day"
        ]
        >= post_shift_start
    ]
    adaptive_post_shift = (
        adaptive_history[
            adaptive_history[
                "simulation_day"
            ]
            >= post_shift_start
        ]
    )

    static_f1 = float(
        static_post_shift[
            "f1_score"
        ].mean()
    )
    adaptive_f1 = float(
        adaptive_post_shift[
            "f1_score"
        ].mean()
    )

    static_profit = float(
        static_history[
            "business_realized_profit"
        ].iloc[-1]
    )
    adaptive_profit = float(
        adaptive_history[
            "business_realized_profit"
        ].iloc[-1]
    )

    retraining_count = len(events)
    promotion_count = (
        get_promotion_count(events)
    )

    summary_parts = [
        (
            "Post-shift mean rolling F1: "
            f"{static_f1:.3f} static | "
            f"{adaptive_f1:.3f} adaptive"
        ),
        (
            "Final realized profit: "
            f"€{static_profit:,.0f} static | "
            f"€{adaptive_profit:,.0f} adaptive"
        ),
        (
            "Profit difference: "
            f"€{adaptive_profit - static_profit:+,.0f}"
        ),
        (
            "Retraining events: "
            f"{retraining_count} | "
            f"promotions: {promotion_count}"
        ),
    ]

    if (
        manifest.get("scenario")
        == CONCEPT_DRIFT_SCENARIO
    ):
        summary_parts.append(
            "Synthetic label flips: "
            f"{manifest.get('selected_labels_flipped', 0)}"
        )

    return "   |   ".join(
        summary_parts
    )


def plot_scenario_panel(
    axis: plt.Axes,
    scenario: pd.DataFrame,
    manifest: dict,
) -> None:
    scenario_name = manifest.get(
        "scenario",
        "",
    )

    if (
        scenario_name
        == CONCEPT_DRIFT_SCENARIO
    ):
        required_columns = {
            "simulation_day",
            "target_flip_rate",
        }

        missing_columns = (
            required_columns
            - set(scenario.columns)
        )

        if missing_columns:
            raise ValueError(
                "Concept-drift manifest is "
                "missing columns: "
                f"{sorted(missing_columns)}"
            )

        values = scenario[
            "target_flip_rate"
        ]

        axis.plot(
            scenario["simulation_day"],
            values,
            color=SCENARIO_COLOR,
            marker="o",
            linewidth=2.5,
        )
        axis.fill_between(
            scenario["simulation_day"],
            values,
            color=SCENARIO_COLOR,
            alpha=0.15,
        )
        axis.set_ylabel(
            "Target label-flip rate"
        )
    elif (
        scenario_name
        == COHORT_SHIFT_SCENARIO
    ):
        if (
            "actual_high_risk_rate"
            in scenario.columns
        ):
            value_column = (
                "actual_high_risk_rate"
            )
        elif (
            "target_high_risk_rate"
            in scenario.columns
        ):
            value_column = (
                "target_high_risk_rate"
            )
        else:
            raise ValueError(
                "Cohort-shift manifest does "
                "not contain a high-risk rate."
            )

        values = scenario[
            value_column
        ]

        axis.plot(
            scenario["simulation_day"],
            values,
            color=SCENARIO_COLOR,
            marker="o",
            linewidth=2.5,
        )
        axis.fill_between(
            scenario["simulation_day"],
            values,
            color=SCENARIO_COLOR,
            alpha=0.15,
        )
        axis.set_ylabel(
            "High-risk share"
        )
    else:
        raise ValueError(
            "Unsupported scenario: "
            f"{scenario_name}"
        )

    axis.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )
    axis.set_ylim(
        0,
        min(
            1.0,
            float(values.max()) + 0.12,
        ),
    )


def get_presentation(
    manifest: dict,
    *,
    promotion_count: int,
) -> dict:
    scenario_name = manifest.get(
        "scenario",
        "",
    )

    adaptive_label = (
        "Adaptive retraining"
        if promotion_count > 0
        else (
            "Retraining enabled "
            "(no promotion)"
        )
    )

    if (
        scenario_name
        == CONCEPT_DRIFT_SCENARIO
    ):
        return {
            "title": (
                "Controlled Concept Drift: "
                "With vs Without Retraining"
            ),
            "subtitle": (
                "Identical real Telco customer "
                "features in both branches; "
                "controlled synthetic target "
                "drift is explicitly audited."
            ),
            "shift_label": (
                "Concept-drift ramp"
            ),
            "adaptive_label": (
                adaptive_label
            ),
            "footer": (
                "Synthetic target changes are "
                "restricted to the documented "
                "drift cohort. Business profit "
                "uses configured retention-cost "
                "and uplift assumptions."
            ),
        }

    if (
        scenario_name
        == COHORT_SHIFT_SCENARIO
    ):
        return {
            "title": (
                "Controlled Customer-Cohort "
                "Shift: Retraining Policy "
                "Evaluation"
            ),
            "subtitle": (
                "Identical real Telco customer "
                "observations and labels in both "
                "branches; no synthetic feature "
                "or target values."
            ),
            "shift_label": (
                "Customer-cohort shift"
            ),
            "adaptive_label": (
                adaptive_label
            ),
            "footer": (
                "Business profit is simulated "
                "using the configured retention "
                "costs, customer value and uplift "
                "assumptions."
            ),
        }

    raise ValueError(
        "Unsupported scenario: "
        f"{scenario_name}"
    )


def build_comparison_figure(
    *,
    experiment_directory: Path,
    output_path: Path,
    rolling_window_size: int = 150,
) -> Path:
    static_history = load_history(
        experiment_directory,
        "without_retraining",
    )
    adaptive_history = load_history(
        experiment_directory,
        "with_retraining",
    )

    static_ground_truth = (
        load_ground_truth(
            experiment_directory,
            "without_retraining",
        )
    )
    adaptive_ground_truth = (
        load_ground_truth(
            experiment_directory,
            "with_retraining",
        )
    )

    static_history = (
        build_rolling_history(
            static_history,
            static_ground_truth,
            window_size=(
                rolling_window_size
            ),
        )
    )
    adaptive_history = (
        build_rolling_history(
            adaptive_history,
            adaptive_ground_truth,
            window_size=(
                rolling_window_size
            ),
        )
    )

    static_profit_by_day = (
        static_history.set_index(
            "simulation_day"
        )[
            "business_realized_profit"
        ]
    )

    adaptive_history[
        "business_profit_delta"
    ] = (
        adaptive_history[
            "business_realized_profit"
        ]
        - adaptive_history[
            "simulation_day"
        ].map(static_profit_by_day)
    )

    if adaptive_history[
        "business_profit_delta"
    ].isna().any():
        raise ValueError(
            "Static and adaptive histories "
            "do not contain matching simulation days."
        )

    events = load_retraining_events(
        experiment_directory
    )
    scenario, manifest = load_scenario(
        experiment_directory
    )

    shift_start = int(
        manifest.get(
            "drift_start_day",
            6,
        )
    )
    ramp_days = int(
        manifest.get(
            "ramp_days",
            3,
        )
    )
    shift_end = (
        shift_start
        + ramp_days
        - 1
    )
    post_shift_start = shift_end

    promotion_count = (
        get_promotion_count(events)
    )
    presentation = get_presentation(
        manifest,
        promotion_count=promotion_count,
    )
    adaptive_label = presentation[
        "adaptive_label"
    ]

    plt.style.use(
        "seaborn-v0_8-whitegrid"
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(15, 14),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                0.75,
                1.25,
                1.25,
            ],
        },
    )

    scenario_axis = axes[0]
    f1_axis = axes[1]
    profit_axis = axes[2]

    plot_scenario_panel(
        scenario_axis,
        scenario,
        manifest,
    )

    f1_axis.plot(
        adaptive_history[
            "simulation_day"
        ],
        adaptive_history[
            "f1_score"
        ],
        color=ADAPTIVE_COLOR,
        marker="o",
        linewidth=3.2,
        label=adaptive_label,
        zorder=2,
    )
    f1_axis.plot(
        static_history[
            "simulation_day"
        ],
        static_history[
            "f1_score"
        ],
        color=STATIC_COLOR,
        linestyle="--",
        marker="o",
        linewidth=2.0,
        label="Without retraining",
        zorder=3,
    )
    f1_axis.set_ylabel(
        "Rolling F1 score\n"
        f"(last {rolling_window_size} labels)"
    )

    all_f1_values = pd.concat(
        [
            static_history["f1_score"],
            adaptive_history["f1_score"],
        ],
        ignore_index=True,
    ).dropna()

    if not all_f1_values.empty:
        f1_axis.set_ylim(
            max(
                0.0,
                float(
                    all_f1_values.min()
                )
                - 0.05,
            ),
            min(
                1.0,
                float(
                    all_f1_values.max()
                )
                + 0.05,
            ),
        )

    profit_days = adaptive_history[
        "simulation_day"
    ]

    profit_delta = adaptive_history[
        "business_profit_delta"
    ]

    profit_axis.axhline(
        0,
        color="#666666",
        linewidth=1.2,
        linestyle="--",
        zorder=1,
    )

    profit_axis.plot(
        profit_days,
        profit_delta,
        color=ADAPTIVE_COLOR,
        marker="o",
        linewidth=3.0,
        label="Cumulative profit uplift",
        zorder=3,
    )

    profit_axis.fill_between(
        profit_days,
        0,
        profit_delta,
        where=(profit_delta >= 0),
        color="#2CA02C",
        alpha=0.16,
        interpolate=True,
        label="Adaptive ahead",
    )

    profit_axis.fill_between(
        profit_days,
        0,
        profit_delta,
        where=(profit_delta < 0),
        color=STATIC_COLOR,
        alpha=0.16,
        interpolate=True,
        label="Static ahead",
    )

    profit_axis.set_ylabel(
        "Cumulative simulated\nprofit uplift"
    )
    profit_axis.set_xlabel(
        "Simulation day"
    )
    profit_axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"€{value:,.0f}"
            )
        )
    )

    for axis in axes:
        axis.axvspan(
            shift_start - 0.5,
            shift_end + 0.5,
            color=SHIFT_COLOR,
            alpha=0.18,
            zorder=0,
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )
        axis.grid(
            axis="x",
            alpha=0.08,
        )
        axis.spines[
            "top"
        ].set_visible(False)
        axis.spines[
            "right"
        ].set_visible(False)

    add_experiment_markers(
        [
            f1_axis,
            profit_axis,
        ],
        events,
        adaptive_history,
    )

    maximum_day = int(
        max(
            scenario[
                "simulation_day"
            ].max(),
            static_history[
                "simulation_day"
            ].max(),
            adaptive_history[
                "simulation_day"
            ].max(),
        )
    )
    profit_axis.set_xticks(
        range(
            1,
            maximum_day + 1,
        )
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STATIC_COLOR,
            linestyle="--",
            marker="o",
            linewidth=2.5,
            label="Without retraining",
        ),
        Line2D(
            [0],
            [0],
            color=ADAPTIVE_COLOR,
            marker="o",
            linewidth=2.8,
            label=adaptive_label,
        ),
        Patch(
            facecolor=SHIFT_COLOR,
            alpha=0.3,
            label=presentation[
                "shift_label"
            ],
        ),
        Line2D(
            [0],
            [0],
            color=RETRAIN_COLOR,
            linestyle=":",
            marker="D",
            linewidth=1.5,
            label="Retraining executed",
        ),
    ]

    if promotion_count > 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=PROMOTION_COLOR,
                marker="*",
                linestyle="None",
                markersize=13,
                label="Champion promoted",
            )
        )

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.925,
        ),
        ncol=min(
            len(legend_handles),
            5,
        ),
        frameon=False,
    )

    summary = build_summary(
        static_history,
        adaptive_history,
        events,
        manifest,
        post_shift_start=(
            post_shift_start
        ),
    )

    figure.text(
        0.5,
        0.895,
        summary,
        ha="center",
        va="center",
        fontsize=9.2,
        color="#444444",
    )

    overlap_note = None

    if histories_overlap(
        static_history,
        adaptive_history,
    ):
        if promotion_count == 0:
            overlap_note = (
                "Static and retraining-enabled "
                "curves overlap because no "
                "candidate passed the promotion "
                "policy."
            )
        else:
            overlap_note = (
                "The plotted curves overlap over "
                "the observed period despite a "
                "recorded promotion."
            )

    if overlap_note is not None:
        figure.text(
            0.5,
            0.875,
            overlap_note,
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
            style="italic",
        )

    figure.suptitle(
        presentation["title"],
        fontsize=21,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.953,
        presentation["subtitle"],
        ha="center",
        fontsize=11,
        color="#555555",
    )
    figure.text(
        0.5,
        0.012,
        presentation["footer"],
        ha="center",
        fontsize=9,
        color="#666666",
    )

    top_limit = (
        0.84
        if overlap_note is not None
        else 0.86
    )

    figure.tight_layout(
        rect=[
            0.04,
            0.04,
            0.98,
            top_limit,
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    return output_path


def main() -> None:
    args = parse_args()

    output_path = (
        args.output
        if args.output is not None
        else (
            args.experiment_directory
            / "churn_retraining_comparison.png"
        )
    )

    generated_path = (
        build_comparison_figure(
            experiment_directory=(
                args.experiment_directory
            ),
            output_path=output_path,
            rolling_window_size=(
                args.rolling_window_size
            ),
        )
    )

    print(
        "Comparison figure generated: "
        f"{generated_path}"
    )


if __name__ == "__main__":
    main()