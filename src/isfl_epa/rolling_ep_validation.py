from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from isfl_epa.epa.dataset import (
    load_training_plays,
    label_drive_outcome,
    build_drive_feature_matrix,
)
from isfl_epa.epa.model import EPModel


FIRST_DATA_SEASON = 27
FIRST_TEST_SEASON = 54
LAST_TEST_SEASON = 62

OUTPUT_DIR = Path(
    "ep_validation"
)

# Half-point EP buckets.
#
# This comfortably covers normal football drive outcomes while
# keeping the calibration plot interpretable.
EP_BIN_EDGES = np.arange(
    -2.5,
    8.51,
    0.5,
)


def weighted_mean(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return float(
        np.average(
            values,
            weights=weights,
        )
    )


def weighted_mae(
    actual,
    predicted,
    weights,
):
    errors = np.abs(
        np.asarray(actual)
        - np.asarray(predicted)
    )

    return weighted_mean(
        errors,
        weights,
    )


def weighted_rmse(
    actual,
    predicted,
    weights,
):
    squared_errors = (
        np.asarray(actual)
        - np.asarray(predicted)
    ) ** 2

    return float(
        np.sqrt(
            weighted_mean(
                squared_errors,
                weights,
            )
        )
    )


def calibration_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["ep_bucket"] = pd.cut(
        result["predicted_ep"],
        bins=EP_BIN_EDGES,
        include_lowest=True,
    )

    calibration = (
        result
        .groupby(
            "ep_bucket",
            observed=False,
        )
        .agg(
            states=(
                "actual_drive_points",
                "count",
            ),
            predicted_ep=(
                "predicted_ep",
                "mean",
            ),
            actual_points=(
                "actual_drive_points",
                "mean",
            ),
        )
        .reset_index()
    )

    calibration["residual"] = (
        calibration["actual_points"]
        - calibration["predicted_ep"]
    )

    calibration["absolute_error"] = (
        calibration["residual"]
        .abs()
    )

    return calibration


def evaluate(
    df: pd.DataFrame,
) -> dict:
    actual = (
        df["actual_drive_points"]
        .astype(float)
        .to_numpy()
    )

    predicted = (
        df["predicted_ep"]
        .astype(float)
        .to_numpy()
    )

    weights = (
        df["weight"]
        .astype(float)
        .to_numpy()
    )

    residual = (
        actual
        - predicted
    )

    return {
        "states": len(df),
        "mae": mean_absolute_error(
            actual,
            predicted,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "r2": r2_score(
            actual,
            predicted,
        ),
        "mean_predicted_ep": float(
            predicted.mean()
        ),
        "mean_actual_points": float(
            actual.mean()
        ),
        "mean_residual": float(
            residual.mean()
        ),
        "weighted_mae": weighted_mae(
            actual,
            predicted,
            weights,
        ),
        "weighted_rmse": weighted_rmse(
            actual,
            predicted,
            weights,
        ),
        "weighted_mean_predicted_ep": (
            weighted_mean(
                predicted,
                weights,
            )
        ),
        "weighted_mean_actual_points": (
            weighted_mean(
                actual,
                weights,
            )
        ),
        "weighted_mean_residual": (
            weighted_mean(
                residual,
                weights,
            )
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
        f"States:                 "
        f"{metrics['states']:,}"
    )

    print(
        f"MAE:                    "
        f"{metrics['mae']:.4f}"
    )

    print(
        f"RMSE:                   "
        f"{metrics['rmse']:.4f}"
    )

    print(
        f"R²:                     "
        f"{metrics['r2']:.4f}"
    )

    print(
        f"Mean predicted EP:      "
        f"{metrics['mean_predicted_ep']:.4f}"
    )

    print(
        f"Mean actual points:     "
        f"{metrics['mean_actual_points']:.4f}"
    )

    print(
        f"Mean residual:          "
        f"{metrics['mean_residual']:+.4f}"
    )

    print(
        f"Weighted MAE:           "
        f"{metrics['weighted_mae']:.4f}"
    )

    print(
        f"Weighted RMSE:          "
        f"{metrics['weighted_rmse']:.4f}"
    )

    print(
        f"Weighted mean residual: "
        f"{metrics['weighted_mean_residual']:+.4f}"
    )


def plot_calibration(
    calibration: pd.DataFrame,
    output_path: Path,
):
    valid = calibration.dropna(
        subset=[
            "predicted_ep",
            "actual_points",
        ]
    ).copy()

    # Avoid plotting bins with essentially no sample support.
    valid = valid[
        valid["states"] >= 100
    ]

    if valid.empty:
        print(
            "No populated EP calibration "
            "bins available for plotting."
        )
        return

    minimum = min(
        valid["predicted_ep"].min(),
        valid["actual_points"].min(),
    )

    maximum = max(
        valid["predicted_ep"].max(),
        valid["actual_points"].max(),
    )

    padding = 0.25

    minimum -= padding
    maximum += padding

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Perfect calibration",
    )

    ax.plot(
        valid["predicted_ep"],
        valid["actual_points"],
        marker="o",
        markersize=7,
        linewidth=2,
        label="ISFL EP model",
    )

    ax.set_xlabel(
        "Predicted Expected Points",
        fontsize=13,
    )

    ax.set_ylabel(
        "Actual Average Drive Points",
        fontsize=13,
    )

    ax.set_title(
        "ISFL Expected Points Model Calibration",
        fontsize=17,
    )

    ax.set_xlim(
        minimum,
        maximum,
    )

    ax.set_ylim(
        minimum,
        maximum,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=11,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def residual_by_season(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for season, group in combined.groupby(
        "season"
    ):
        metrics = evaluate(
            group
        )

        rows.append(
            {
                "season": int(
                    season
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def residual_by_down(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    if "down" not in combined.columns:
        return pd.DataFrame()

    result = (
        combined
        .groupby(
            "down",
            dropna=False,
        )
        .agg(
            states=(
                "actual_drive_points",
                "count",
            ),
            predicted_ep=(
                "predicted_ep",
                "mean",
            ),
            actual_points=(
                "actual_drive_points",
                "mean",
            ),
        )
        .reset_index()
    )

    result["residual"] = (
        result["actual_points"]
        - result["predicted_ep"]
    )

    return result


def residual_by_distance(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    if "distance" not in combined.columns:
        return pd.DataFrame()

    result = combined.copy()

    result["distance_bucket"] = pd.cut(
        result["distance"],
        bins=[
            -0.001,
            1,
            3,
            5,
            7,
            10,
            15,
            np.inf,
        ],
        labels=[
            "1",
            "2-3",
            "4-5",
            "6-7",
            "8-10",
            "11-15",
            "16+",
        ],
        include_lowest=True,
    )

    grouped = (
        result
        .groupby(
            "distance_bucket",
            observed=False,
        )
        .agg(
            states=(
                "actual_drive_points",
                "count",
            ),
            predicted_ep=(
                "predicted_ep",
                "mean",
            ),
            actual_points=(
                "actual_drive_points",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["residual"] = (
        grouped["actual_points"]
        - grouped["predicted_ep"]
    )

    return grouped


def residual_by_field_position(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    if "yardline_100" not in combined.columns:
        return pd.DataFrame()

    result = combined.copy()

    result["field_position_bucket"] = pd.cut(
        result["yardline_100"],
        bins=[
            -0.001,
            10,
            20,
            40,
            60,
            80,
            90,
            100,
        ],
        labels=[
            "Own 0-10",
            "Own 11-20",
            "Own 21-40",
            "Own 41-50 / Opp 49-40",
            "Opp 39-20",
            "Opp 19-10",
            "Opp 9-goal",
        ],
        include_lowest=True,
    )

    grouped = (
        result
        .groupby(
            "field_position_bucket",
            observed=False,
        )
        .agg(
            states=(
                "actual_drive_points",
                "count",
            ),
            predicted_ep=(
                "predicted_ep",
                "mean",
            ),
            actual_points=(
                "actual_drive_points",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["residual"] = (
        grouped["actual_points"]
        - grouped["predicted_ep"]
    )

    return grouped


def main():
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    print(
        f"Loading S{FIRST_DATA_SEASON}"
        f"–S{LAST_TEST_SEASON} "
        f"EP training plays..."
    )

    df = load_training_plays(
        list(
            range(
                FIRST_DATA_SEASON,
                LAST_TEST_SEASON + 1,
            )
        ),
        league="ISFL",
    )

    print()
    print(
        f"Raw rows: {len(df):,}"
    )

    print()
    print(
        "Labeling drive outcomes..."
    )

    df = label_drive_outcome(
        df
    )

    print()
    print(
        "Building EP feature matrix..."
    )

    (
        X,
        y,
        weights,
        drive_starts,
    ) = build_drive_feature_matrix(
        df
    )

    print()
    print(
        f"Valid EP states: "
        f"{len(X):,}"
    )

    # ------------------------------------------------------
    # Attach metadata to each valid feature row.
    #
    # build_drive_feature_matrix preserves the source index,
    # so we can recover season and state variables from df.
    # ------------------------------------------------------

    metadata_columns = [
        column
        for column in [
            "season",
            "game_id",
            "down",
            "distance",
            "yardline_100",
            "half_seconds_remaining",
            "score_differential",
        ]
        if column in df.columns
    ]

    metadata = (
        df.loc[
            X.index,
            metadata_columns,
        ]
        .copy()
    )

    metadata[
        "actual_drive_points"
    ] = (
        y
        .astype(float)
        .to_numpy()
    )

    metadata[
        "weight"
    ] = np.asarray(
        weights,
        dtype=float,
    )

    season_results = []
    prediction_frames = []

    # ------------------------------------------------------
    # Expanding-window validation
    # ------------------------------------------------------

    for test_season in range(
        FIRST_TEST_SEASON,
        LAST_TEST_SEASON + 1,
    ):
        train_mask = (
            metadata["season"]
            < test_season
        )

        test_mask = (
            metadata["season"]
            == test_season
        )

        X_train = X.loc[
            train_mask
        ]

        y_train = y.loc[
            train_mask
        ]

        train_weights = np.asarray(
            weights
        )[
            train_mask.to_numpy()
        ]

        X_test = X.loc[
            test_mask
        ]

        if (
            len(X_train) == 0
            or len(X_test) == 0
        ):
            print()
            print(
                f"Skipping S{test_season}: "
                f"missing train/test data."
            )
            continue

        print()
        print(
            "=" * 60
        )

        print(
            f"TESTING S{test_season}"
        )

        print(
            f"Training on "
            f"S{FIRST_DATA_SEASON}"
            f"–S{test_season - 1}"
        )

        print(
            f"Training states: "
            f"{len(X_train):,}"
        )

        print(
            f"Test states: "
            f"{len(X_test):,}"
        )

        print(
            "=" * 60
        )

        model = EPModel()

        print()
        print(
            "Training EP model..."
        )

        model.train(
            X_train,
            y_train,
            model_type="hgb_reg",
            sample_weight=train_weights,
        )

        print(
            "Predicting unseen season..."
        )

        predicted_ep = (
            model.predict_ep(
                X_test
            )
        )

        test_result = (
            metadata.loc[
                test_mask
            ]
            .copy()
        )

        test_result[
            "predicted_ep"
        ] = predicted_ep

        test_result[
            "test_season"
        ] = test_season

        test_result[
            "train_through"
        ] = test_season - 1

        metrics = evaluate(
            test_result
        )

        print_metrics(
            test_season,
            metrics,
        )

        season_results.append(
            {
                "season": test_season,
                "train_start": (
                    FIRST_DATA_SEASON
                ),
                "train_end": (
                    test_season - 1
                ),
                **metrics,
            }
        )

        prediction_frames.append(
            test_result
        )

    # ------------------------------------------------------
    # Season summary
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
        "EP SEASON SUMMARY ====="
    )

    print(
        season_df[
            [
                "season",
                "states",
                "mae",
                "rmse",
                "r2",
                "mean_predicted_ep",
                "mean_actual_points",
                "mean_residual",
            ]
        ].to_string(
            index=False
        )
    )

    if not prediction_frames:
        print(
            "No out-of-sample "
            "predictions generated."
        )
        return

    # ------------------------------------------------------
    # Combine all true out-of-sample predictions
    # ------------------------------------------------------

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

    combined_metrics = evaluate(
        combined
    )

    print()
    print()
    print(
        "===== S54–S62 "
        "COMBINED OUT-OF-SAMPLE EP ====="
    )

    print(
        f"States:                 "
        f"{combined_metrics['states']:,}"
    )

    print(
        f"MAE:                    "
        f"{combined_metrics['mae']:.4f}"
    )

    print(
        f"RMSE:                   "
        f"{combined_metrics['rmse']:.4f}"
    )

    print(
        f"R²:                     "
        f"{combined_metrics['r2']:.4f}"
    )

    print(
        f"Mean predicted EP:      "
        f"{combined_metrics['mean_predicted_ep']:.4f}"
    )

    print(
        f"Mean actual points:     "
        f"{combined_metrics['mean_actual_points']:.4f}"
    )

    print(
        f"Mean residual:          "
        f"{combined_metrics['mean_residual']:+.4f}"
    )

    print()
    print(
        "Drive-weighted metrics:"
    )

    print(
        f"Weighted MAE:           "
        f"{combined_metrics['weighted_mae']:.4f}"
    )

    print(
        f"Weighted RMSE:          "
        f"{combined_metrics['weighted_rmse']:.4f}"
    )

    print(
        f"Weighted mean predicted:"
        f" "
        f"{combined_metrics['weighted_mean_predicted_ep']:.4f}"
    )

    print(
        f"Weighted mean actual:   "
        f"{combined_metrics['weighted_mean_actual_points']:.4f}"
    )

    print(
        f"Weighted mean residual: "
        f"{combined_metrics['weighted_mean_residual']:+.4f}"
    )

    # ------------------------------------------------------
    # Combined EP calibration
    # ------------------------------------------------------

    calibration = calibration_table(
        combined
    )

    calibration_output = (
        OUTPUT_DIR
        / "rolling-oos-calibration.csv"
    )

    calibration.to_csv(
        calibration_output,
        index=False,
    )

    print()
    print(
        "===== COMBINED "
        "EP CALIBRATION ====="
    )

    populated_calibration = (
        calibration[
            calibration["states"] > 0
        ]
    )

    print(
        populated_calibration[
            [
                "ep_bucket",
                "states",
                "predicted_ep",
                "actual_points",
                "residual",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Additional residual diagnostics
    # ------------------------------------------------------

    by_season = residual_by_season(
        combined
    )

    by_season.to_csv(
        OUTPUT_DIR
        / "rolling-oos-residual-by-season.csv",
        index=False,
    )

    by_down = residual_by_down(
        combined
    )

    if not by_down.empty:
        by_down.to_csv(
            OUTPUT_DIR
            / "rolling-oos-residual-by-down.csv",
            index=False,
        )

    by_distance = (
        residual_by_distance(
            combined
        )
    )

    if not by_distance.empty:
        by_distance.to_csv(
            OUTPUT_DIR
            / "rolling-oos-residual-by-distance.csv",
            index=False,
        )

    by_field_position = (
        residual_by_field_position(
            combined
        )
    )

    if not by_field_position.empty:
        by_field_position.to_csv(
            OUTPUT_DIR
            / "rolling-oos-residual-by-field-position.csv",
            index=False,
        )

    # ------------------------------------------------------
    # Calibration chart
    # ------------------------------------------------------

    plot_output = (
        OUTPUT_DIR
        / "rolling-oos-calibration.png"
    )

    plot_calibration(
        calibration,
        plot_output,
    )

    print()
    print(
        "Saved validation files to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()