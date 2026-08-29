from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


STATIC_COLOR = "#F05A40"
ADAPTIVE_COLOR = "#5865F2"
SHIFT_COLOR = "#F6C85F"
RETRAIN_COLOR = "#00A67E"
PROMOTION_COLOR = "#2CA02C"

DEFAULT_EXPERIMENT_DIRECTORY = Path(
    "results/churn_retraining_comparison/"
    "controlled_cohort_shift"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the controlled churn retraining comparison."
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
    return parser.parse_args()


def load_json(path: Path) -> dict:
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
            f"Performance history not found: {path}"
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
            f"Ground truth not found: {path}"
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

    return (
        ground_truth
        .dropna(
            subset=[
                "prediction",
                "churn",
                "released_simulation_day",
            ]
        )
        .sort_values(
            [
                "released_simulation_day",
                "prediction_timestamp",
            ]
        )
        .reset_index(drop=True)
    )

def build_rolling_history(
    history: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    window_size: int = 150,
) -> pd.DataFrame:
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

        available_labels = ground_truth[
            ground_truth[
                "released_simulation_day"
            ]
            <= simulation_day
        ].tail(window_size)

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

    if "simulation_day" not in events.columns:
        return pd.DataFrame()

    return (
        events
        .dropna(subset=["simulation_day"])
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
            f"Scenario manifest not found: {path}"
        )

    manifest = load_json(path)

    daily_summary = pd.DataFrame(
        manifest["daily_summary"]
    )

    if "simulation_day" not in daily_summary.columns:
        if "day" in daily_summary.columns:
            daily_summary = daily_summary.rename(
                columns={
                    "day": "simulation_day",
                }
            )
        else:
            daily_summary["simulation_day"] = range(
                1,
                len(daily_summary) + 1,
            )

    share_column = None

    for candidate in [
        "actual_high_risk_rate",
        "target_high_risk_rate",
        "actual_high_risk_share",
        "high_risk_share",
        "target_high_risk_share",
    ]:
        if candidate in daily_summary.columns:
            share_column = candidate
            break

    if share_column is None:
        raise ValueError(
            "No high-risk share column found in "
            f"{path}."
        )

    daily_summary = daily_summary.rename(
        columns={
            share_column: "high_risk_share",
        }
    )

    return daily_summary, manifest


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
            column="business_realized_profit",
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


def build_summary(
    static_history: pd.DataFrame,
    adaptive_history: pd.DataFrame,
    events: pd.DataFrame,
    *,
    post_shift_start: int,
) -> str:
    static_post_shift = static_history[
        static_history["simulation_day"]
        >= post_shift_start
    ]
    adaptive_post_shift = adaptive_history[
        adaptive_history["simulation_day"]
        >= post_shift_start
    ]

    static_f1 = float(
        static_post_shift["f1_score"].mean()
    )
    adaptive_f1 = float(
        adaptive_post_shift["f1_score"].mean()
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
        int(
            events[
                "champion_promoted"
            ].fillna(False).sum()
        )
        if "champion_promoted" in events.columns
        else 0
    )

    return "\n".join(
        [
            (
                f"Post-shift mean rolling F1: "
                f"{static_f1:.3f} static | "
                f"{adaptive_f1:.3f} adaptive"
            ),
            (
                f"Final realized profit: "
                f"€{static_profit:,.0f} static | "
                f"€{adaptive_profit:,.0f} adaptive"
            ),
            (
                f"Profit difference: "
                f"€{adaptive_profit - static_profit:+,.0f}"
            ),
            (
                f"Retraining events: "
                f"{retraining_count} | "
                f"promotions: {promotion_count}"
            ),
        ]
    )


def build_comparison_figure(
    *,
    experiment_directory: Path,
    output_path: Path,
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
            window_size=150,
        )
    )
    adaptive_history = (
        build_rolling_history(
            adaptive_history,
            adaptive_ground_truth,
            window_size=150,
        )
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

    cohort_axis = axes[0]
    f1_axis = axes[1]
    profit_axis = axes[2]

    cohort_axis.plot(
        scenario["simulation_day"],
        scenario["high_risk_share"],
        color="#8C6BB1",
        marker="o",
        linewidth=2.5,
        label="High-risk customer share",
    )
    cohort_axis.fill_between(
        scenario["simulation_day"],
        scenario["high_risk_share"],
        color="#8C6BB1",
        alpha=0.15,
    )
    cohort_axis.set_ylabel(
        "High-risk share"
    )
    cohort_axis.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )
    cohort_axis.set_ylim(
        0,
        min(
            1.0,
            float(
                scenario[
                    "high_risk_share"
                ].max()
            )
            + 0.12,
        ),
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
        label=(
            "Retraining enabled "
            "(no promotion)"
        ),
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
        "Rolling F1 score\n(last 150 labels)"
    )

    all_f1_values = pd.concat(
        [
            static_history["f1_score"],
            adaptive_history["f1_score"],
        ],
        ignore_index=True,
    )
    f1_axis.set_ylim(
        max(
            0.0,
            float(all_f1_values.min()) - 0.05,
        ),
        min(
            1.0,
            float(all_f1_values.max()) + 0.05,
        ),
    )

    profit_axis.plot(
        adaptive_history[
            "simulation_day"
        ],
        adaptive_history[
            "business_realized_profit"
        ],
        color=ADAPTIVE_COLOR,
        marker="o",
        linewidth=3.2,
        label=(
            "Retraining enabled "
            "(no promotion)"
        ),
        zorder=2,
    )
    profit_axis.plot(
        static_history[
            "simulation_day"
        ],
        static_history[ 
            "business_realized_profit"
        ],
        color=STATIC_COLOR,
        linestyle="--",
        marker="o",
        linewidth=2.0,
        label="Without retraining",
        zorder=3,
    )
    profit_axis.set_ylabel(
        "Cumulative simulated\nrealized profit"
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
            label="Retraining enabled (no promotion)",
        ),
        Patch(
            facecolor=SHIFT_COLOR,
            alpha=0.3,
            label="Customer-cohort shift",
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

    if (
        "champion_promoted"
        in events.columns
        and events[
            "champion_promoted"
        ].fillna(False).any()
    ):
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
        post_shift_start=post_shift_start,
    )

    figure.text(
        0.5,
        0.895,
        summary.replace(
            "\n",
            "   |   ",
        ),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.text(
        0.5,
        0.875,
        (
            "Static and retraining-enabled curves overlap "
            "because both retraining candidates were rejected "
            "by the promotion policy."
        ),
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
        style="italic",
    )



    figure.suptitle(
        (
            "Controlled Customer-Cohort Shift: "
            "Retraining Policy Evaluation"
        ),
        fontsize=21,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.953,
        (
            "Identical real Telco customer observations "
            "and labels in both branches; "
            "no synthetic feature or target values."
        ),
        ha="center",
        fontsize=11,
        color="#555555",
    )
    figure.text(
        0.5,
        0.012,
        (
            "Business profit is simulated using the "
            "configured retention costs, customer value "
            "and uplift assumptions."
        ),
        ha="center",
        fontsize=9,
        color="#666666",
    )

    figure.tight_layout(
        rect=[
            0.04,
            0.04,
            0.98,
            0.86,
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
        )
    )

    print(
        "Comparison figure generated: "
        f"{generated_path}"
    )


if __name__ == "__main__":
    main()