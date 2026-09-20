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


FIRST_TRAIN_SEASON = 27
FIRST_TEST_SEASON = 54
LAST_TEST_SEASON = 62

OUTPUT_DIR = Path(
    "wp_validation"
)

BIN_EDGES = np.linspace(
    0.0,
    1.0,
    21,
)


def calibration_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["wp_bucket"] = pd.cut(
        result["predicted_wp"],
        bins=BIN_EDGES,
        include_lowest=True,
    )

    calibration = (
        result
        .groupby(
            "wp_bucket",
            observed=False,
        )
        .agg(
            states=(
                "possession_won",
                "count",
            ),
            predicted_wp=(
                "predicted_wp",
                "mean",
            ),
            actual_win_rate=(
                "possession_won",
                "mean",
            ),
            games=(
                "game_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    calibration["difference"] = (
        calibration["actual_win_rate"]
        - calibration["predicted_wp"]
    )

    calibration["absolute_error"] = (
        calibration["difference"]
        .abs()
    )

    return calibration


def calculate_ece(
    calibration: pd.DataFrame,
) -> float:
    total = calibration[
        "states"
    ].sum()

    if total == 0:
        return float("nan")

    return float(
        (
            calibration["states"]
            * calibration[
                "absolute_error"
            ]
        ).sum()
        / total
    )


def evaluate(
    df: pd.DataFrame,
) -> dict:
    y = (
        df["possession_won"]
        .astype(int)
        .to_numpy()
    )

    p = (
        df["predicted_wp"]
        .astype(float)
        .to_numpy()
    )

    calibration = (
        calibration_table(df)
    )

    return {
        "states": len(df),
        "games": df[
            "game_id"
        ].nunique(),
        "brier": (
            brier_score_loss(
                y,
                p,
            )
        ),
        "log_loss": (
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),
        "roc_auc": (
            roc_auc_score(
                y,
                p,
            )
        ),
        "mean_wp": float(
            p.mean()
        ),
        "actual_win_rate": float(
            y.mean()
        ),
        "ece": calculate_ece(
            calibration
        ),
    }


def print_metrics(
    season: int,
    metrics: dict,
):
    print()
    print(
        f"===== S{season} "
        f"OUT-OF-SAMPLE ====="
    )

    print(
        f"States:     "
        f"{metrics['states']:,}"
    )

    print(
        f"Games:      "
        f"{metrics['games']:,}"
    )

    print(
        f"Brier:      "
        f"{metrics['brier']:.4f}"
    )

    print(
        f"Log loss:   "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"ROC-AUC:    "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Mean WP:    "
        f"{metrics['mean_wp']:.4f}"
    )

    print(
        f"Actual:     "
        f"{metrics['actual_win_rate']:.4f}"
    )

    print(
        f"ECE:        "
        f"{metrics['ece']:.4f}"
    )


def plot_calibration(
    calibration: pd.DataFrame,
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
        label="Rolling OOS WP",
    )

    plt.xlabel(
        "Predicted Win Probability"
    )

    plt.ylabel(
        "Actual Win Rate"
    )

    plt.title(
        "S54–S62 Rolling "
        "Out-of-Sample WP Calibration"
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


def main():
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    engine = get_engine()

    season_results = []
    prediction_frames = []

    # ------------------------------------------------------
    # Expanding-window validation
    # ------------------------------------------------------

    for test_season in range(
        FIRST_TEST_SEASON,
        LAST_TEST_SEASON + 1,
    ):
        train_seasons = list(
            range(
                FIRST_TRAIN_SEASON,
                test_season,
            )
        )

        print()
        print(
            "=" * 60
        )

        print(
            f"TESTING S{test_season}"
        )

        print(
            f"Training on "
            f"S{train_seasons[0]}"
            f"–S{train_seasons[-1]}"
        )

        print(
            "=" * 60
        )

        print()
        print(
            "Loading training data..."
        )

        train_df = (
            load_wp_training_data(
                engine=engine,
                seasons=train_seasons,
                game_type="regular",
            )
        )

        print()
        print(
            f"Training rows: "
            f"{len(train_df):,}"
        )

        print()
        print(
            "Loading test data..."
        )

        test_df = (
            load_wp_training_data(
                engine=engine,
                seasons=[
                    test_season
                ],
                game_type="regular",
            )
        )

        print()
        print(
            f"Test rows: "
            f"{len(test_df):,}"
        )

        if (
            train_df.empty
            or test_df.empty
        ):
            print(
                "Skipping due to "
                "missing data."
            )

            continue

        # --------------------------------------------------
        # Train fresh model using only prior seasons
        # --------------------------------------------------

        print()
        print(
            "Training WP model..."
        )

        model = WPModel()

        model.fit(
            train_df
        )

        # --------------------------------------------------
        # Predict the unseen season
        # --------------------------------------------------

        test_df = test_df.copy()

        test_df[
            "predicted_wp"
        ] = model.predict_wp(
            test_df
        )

        test_df[
            "possession_won"
        ] = (
            test_df[
                "possession_won"
            ]
            .astype(int)
        )

        test_df[
            "test_season"
        ] = test_season

        test_df[
            "train_through"
        ] = (
            test_season - 1
        )

        metrics = evaluate(
            test_df
        )

        print_metrics(
            test_season,
            metrics,
        )

        season_results.append(
            {
                "season": (
                    test_season
                ),
                "train_start": (
                    FIRST_TRAIN_SEASON
                ),
                "train_end": (
                    test_season - 1
                ),
                **metrics,
            }
        )

        prediction_frames.append(
            test_df[
                [
                    "id",
                    "game_id",
                    "season",
                    "play_index",
                    "game_seconds",
                    "score_differential",
                    "possession_won",
                    "predicted_wp",
                    "test_season",
                    "train_through",
                ]
            ].copy()
        )

    # ------------------------------------------------------
    # Save season-by-season results
    # ------------------------------------------------------

    season_df = pd.DataFrame(
        season_results
    )

    season_output = (
        OUTPUT_DIR
        / "rolling-oos-by-season.csv"
    )

    season_df.to_csv(
        season_output,
        index=False,
    )

    print()
    print()
    print(
        "===== ROLLING OOS "
        "SEASON SUMMARY ====="
    )

    print(
        season_df[
            [
                "season",
                "states",
                "games",
                "brier",
                "log_loss",
                "roc_auc",
                "ece",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Combine all true out-of-sample predictions
    # ------------------------------------------------------

    if not prediction_frames:
        print(
            "No prediction frames "
            "were generated."
        )

        return

    combined = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    prediction_output = (
        OUTPUT_DIR
        / "rolling-oos-predictions.csv"
    )

    combined.to_csv(
        prediction_output,
        index=False,
    )

    combined_metrics = (
        evaluate(
            combined
        )
    )

    print()
    print()
    print(
        "===== S54–S62 "
        "COMBINED OUT-OF-SAMPLE ====="
    )

    print(
        f"States:            "
        f"{combined_metrics['states']:,}"
    )

    print(
        f"Games:             "
        f"{combined_metrics['games']:,}"
    )

    print(
        f"Brier:             "
        f"{combined_metrics['brier']:.4f}"
    )

    print(
        f"Log loss:          "
        f"{combined_metrics['log_loss']:.4f}"
    )

    print(
        f"ROC-AUC:           "
        f"{combined_metrics['roc_auc']:.4f}"
    )

    print(
        f"Mean predicted WP: "
        f"{combined_metrics['mean_wp']:.4f}"
    )

    print(
        f"Actual win rate:   "
        f"{combined_metrics['actual_win_rate']:.4f}"
    )

    print(
        f"ECE:               "
        f"{combined_metrics['ece']:.4f}"
    )

    # ------------------------------------------------------
    # Combined calibration table
    # ------------------------------------------------------

    combined_calibration = (
        calibration_table(
            combined
        )
    )

    calibration_output = (
        OUTPUT_DIR
        / "rolling-oos-calibration.csv"
    )

    combined_calibration.to_csv(
        calibration_output,
        index=False,
    )

    print()
    print(
        "===== COMBINED "
        "CALIBRATION ====="
    )

    print(
        combined_calibration[
            [
                "wp_bucket",
                "states",
                "games",
                "predicted_wp",
                "actual_win_rate",
                "difference",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Reliability plot
    # ------------------------------------------------------

    plot_output = (
        OUTPUT_DIR
        / "rolling-oos-calibration.png"
    )

    plot_calibration(
        calibration=(
            combined_calibration
        ),
        output_path=plot_output,
    )

    print()
    print(
        "Saved:"
    )

    print(
        season_output
    )

    print(
        prediction_output
    )

    print(
        calibration_output
    )

    print(
        plot_output
    )


if __name__ == "__main__":
    main()