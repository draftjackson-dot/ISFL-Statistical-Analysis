from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from isfl_epa.storage.database import get_engine
from isfl_epa.wp.model import WPModel
from isfl_epa.wp.training import (
    load_wp_training_data,
)


MODEL_PATH = Path(
    "models/wp_model_2022.joblib"
)

OUTPUT_DIR = Path(
    "wp_validation"
)


def calibration_table(
    df: pd.DataFrame,
    bins: int = 20,
) -> pd.DataFrame:
    result = df.copy()

    edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    result["wp_bucket"] = pd.cut(
        result["predicted_wp"],
        bins=edges,
        include_lowest=True,
        right=True,
    )

    calibration = (
        result
        .groupby(
            "wp_bucket",
            observed=False,
        )
        .agg(
            states=("possession_won", "count"),
            predicted_wp=("predicted_wp", "mean"),
            actual_win_rate=("possession_won", "mean"),
        )
        .reset_index()
    )

    calibration["difference"] = (
        calibration["actual_win_rate"]
        - calibration["predicted_wp"]
    )

    calibration["absolute_error"] = (
        calibration["difference"].abs()
    )

    return calibration


def expected_calibration_error(
    calibration: pd.DataFrame,
) -> float:
    total = calibration["states"].sum()

    if total == 0:
        return float("nan")

    weighted_error = (
        calibration["states"]
        * calibration["absolute_error"]
    ).sum()

    return float(
        weighted_error / total
    )


def evaluate(
    df: pd.DataFrame,
) -> dict:
    y = df[
        "possession_won"
    ].astype(int)

    p = df[
        "predicted_wp"
    ].astype(float)

    calibration = calibration_table(
        df
    )

    return {
        "states": len(df),
        "games": df["game_id"].nunique(),
        "brier": brier_score_loss(
            y,
            p,
        ),
        "log_loss": log_loss(
            y,
            p,
            labels=[0, 1],
        ),
        "roc_auc": roc_auc_score(
            y,
            p,
        ),
        "mean_predicted_wp": p.mean(),
        "actual_win_rate": y.mean(),
        "ece": expected_calibration_error(
            calibration
        ),
    }


def time_bucket(
    seconds: float,
) -> str:
    if seconds > 2700:
        return "Q1"

    if seconds > 1800:
        return "Q2"

    if seconds > 900:
        return "Q3"

    if seconds > 300:
        return "Q4 > 5 min"

    if seconds > 120:
        return "Final 5 min"

    if seconds > 0:
        return "Final 2 min"

    return "OT"


def calibration_by_time(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["time_bucket"] = (
        result["game_seconds"]
        .apply(time_bucket)
    )

    rows = []

    bucket_order = [
        "Q1",
        "Q2",
        "Q3",
        "Q4 > 5 min",
        "Final 5 min",
        "Final 2 min",
        "OT",
    ]

    for bucket in bucket_order:
        sub = result[
            result["time_bucket"] == bucket
        ]

        if sub.empty:
            continue

        metrics = evaluate(
            sub
        )

        rows.append(
            {
                "time_bucket": bucket,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def calibration_by_season(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for season in sorted(
        df["season"].unique()
    ):
        sub = df[
            df["season"] == season
        ]

        if sub.empty:
            continue

        metrics = evaluate(
            sub
        )

        rows.append(
            {
                "season": int(season),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def score_margin_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    bins = [
        -100,
        -15,
        -8,
        -1,
        0,
        1,
        8,
        15,
        100,
    ]

    labels = [
        "Down 15+",
        "Down 8-14",
        "Down 1-7",
        "Tied",
        "Up 1",
        "Up 2-7",
        "Up 8-14",
        "Up 15+",
    ]

    result["score_bucket"] = pd.cut(
        result["score_differential"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    rows = []

    for bucket in labels:
        sub = result[
            result["score_bucket"] == bucket
        ]

        if sub.empty:
            continue

        metrics = evaluate(
            sub
        )

        rows.append(
            {
                "score_bucket": bucket,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def plot_calibration(
    calibration: pd.DataFrame,
    title: str,
    output_path: Path,
):
    valid = calibration.dropna(
        subset=[
            "predicted_wp",
            "actual_win_rate",
        ]
    )

    plt.figure(
        figsize=(9, 9)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    plt.plot(
        valid["predicted_wp"],
        valid["actual_win_rate"],
        marker="o",
        linewidth=2,
        label="WP model",
    )

    plt.xlabel(
        "Predicted Win Probability"
    )

    plt.ylabel(
        "Actual Win Rate"
    )

    plt.title(
        title
    )

    plt.xlim(
        0,
        1,
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def print_metrics(
    name: str,
    metrics: dict,
):
    print()
    print(
        f"===== {name} ====="
    )

    print(
        f"States:              "
        f"{metrics['states']:,}"
    )

    print(
        f"Games:               "
        f"{metrics['games']:,}"
    )

    print(
        f"Brier score:         "
        f"{metrics['brier']:.4f}"
    )

    print(
        f"Log loss:            "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"ROC-AUC:             "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Mean predicted WP:   "
        f"{metrics['mean_predicted_wp']:.4f}"
    )

    print(
        f"Actual win rate:     "
        f"{metrics['actual_win_rate']:.4f}"
    )

    print(
        f"Calibration ECE:     "
        f"{metrics['ece']:.4f}"
    )


def run_validation(
    model: WPModel,
    engine,
    seasons: list[int],
    label: str,
    filename_prefix: str,
):
    print()
    print(
        f"Loading {label}..."
    )

    df = load_wp_training_data(
        engine=engine,
        seasons=seasons,
        game_type="regular",
    )

    if df.empty:
        print(
            "No rows found."
        )
        return

    df = df.copy()

    df["predicted_wp"] = (
        model.predict_wp(
            df
        )
    )

    df["possession_won"] = (
        df["possession_won"]
        .astype(int)
    )

    metrics = evaluate(
        df
    )

    print_metrics(
        label,
        metrics,
    )

    calibration = calibration_table(
        df
    )

    season_table = calibration_by_season(
        df
    )

    time_table = calibration_by_time(
        df
    )

    margin_table = score_margin_table(
        df
    )

    calibration.to_csv(
        OUTPUT_DIR
        / f"{filename_prefix}-calibration.csv",
        index=False,
    )

    season_table.to_csv(
        OUTPUT_DIR
        / f"{filename_prefix}-by-season.csv",
        index=False,
    )

    time_table.to_csv(
        OUTPUT_DIR
        / f"{filename_prefix}-by-time.csv",
        index=False,
    )

    margin_table.to_csv(
        OUTPUT_DIR
        / f"{filename_prefix}-by-score.csv",
        index=False,
    )

    plot_calibration(
        calibration=calibration,
        title=f"{label} WP Calibration",
        output_path=(
            OUTPUT_DIR
            / f"{filename_prefix}-calibration.png"
        ),
    )

    print()
    print(
        "Calibration buckets:"
    )

    print(
        calibration[
            [
                "wp_bucket",
                "states",
                "predicted_wp",
                "actual_win_rate",
                "difference",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Calibration by time:"
    )

    print(
        time_table[
            [
                "time_bucket",
                "states",
                "brier",
                "mean_predicted_wp",
                "actual_win_rate",
                "ece",
            ]
        ].to_string(
            index=False
        )
    )

    return df


def main():
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    engine = get_engine()

    print(
        f"Loading model from "
        f"{MODEL_PATH}..."
    )

    model = WPModel.load(
        MODEL_PATH
    )

    # ------------------------------------------------------
    # TRUE HOLDOUT VALIDATION
    #
    # Model was trained on S27-S61.
    # S62 is therefore the most important validation.
    # ------------------------------------------------------

    run_validation(
        model=model,
        engine=engine,
        seasons=[62],
        label="S62 HOLDOUT",
        filename_prefix="s62",
    )

    # ------------------------------------------------------
    # FULL HISTORICAL SANITY CHECK
    #
    # S27-S61 are training seasons, so this is NOT a fully
    # out-of-sample test. It is still useful for detecting
    # major calibration problems and historical drift.
    # ------------------------------------------------------

    run_validation(
        model=model,
        engine=engine,
        seasons=list(
            range(
                27,
                63,
            )
        ),
        label="S27-S62 FULL DATASET",
        filename_prefix="all-seasons",
    )

    print()
    print(
        "Validation files saved in:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()