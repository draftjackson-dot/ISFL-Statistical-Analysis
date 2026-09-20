import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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


TRAIN_SEASONS = list(range(27, 58))
CALIBRATION_SEASONS = list(range(58, 62))
TEST_SEASONS = [62]


def logit(p):
    p = np.clip(
        p,
        1e-6,
        1 - 1e-6,
    )

    return np.log(
        p / (1 - p)
    )


def calibration_table(
    y,
    probabilities,
    bins=20,
):
    df = pd.DataFrame(
        {
            "actual": y,
            "wp": probabilities,
        }
    )

    edges = np.linspace(
        0,
        1,
        bins + 1,
    )

    df["bucket"] = pd.cut(
        df["wp"],
        bins=edges,
        include_lowest=True,
    )

    result = (
        df
        .groupby(
            "bucket",
            observed=False,
        )
        .agg(
            states=("actual", "count"),
            predicted_wp=("wp", "mean"),
            actual_win_rate=("actual", "mean"),
        )
        .reset_index()
    )

    result["difference"] = (
        result["actual_win_rate"]
        - result["predicted_wp"]
    )

    result["absolute_error"] = (
        result["difference"].abs()
    )

    return result


def ece(
    calibration,
):
    total = calibration[
        "states"
    ].sum()

    if total == 0:
        return float("nan")

    return float(
        (
            calibration["states"]
            * calibration["absolute_error"]
        ).sum()
        / total
    )


def evaluate(
    y,
    probabilities,
):
    table = calibration_table(
        y,
        probabilities,
    )

    return {
        "brier": brier_score_loss(
            y,
            probabilities,
        ),
        "log_loss": log_loss(
            y,
            probabilities,
            labels=[0, 1],
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "mean_wp": float(
            np.mean(probabilities)
        ),
        "actual_win_rate": float(
            np.mean(y)
        ),
        "ece": ece(
            table
        ),
        "table": table,
    }


def print_metrics(
    name,
    result,
):
    print()
    print(
        f"===== {name} ====="
    )

    print(
        f"Brier:          "
        f"{result['brier']:.4f}"
    )

    print(
        f"Log loss:       "
        f"{result['log_loss']:.4f}"
    )

    print(
        f"ROC-AUC:        "
        f"{result['roc_auc']:.4f}"
    )

    print(
        f"Mean WP:        "
        f"{result['mean_wp']:.4f}"
    )

    print(
        f"Actual win rate:"
        f" {result['actual_win_rate']:.4f}"
    )

    print(
        f"ECE:            "
        f"{result['ece']:.4f}"
    )


def main():
    engine = get_engine()

    print(
        "Loading training data "
        "S27-S57..."
    )

    train_df = load_wp_training_data(
        engine=engine,
        seasons=TRAIN_SEASONS,
        game_type="regular",
    )

    print()
    print(
        "Loading calibration data "
        "S58-S61..."
    )

    calibration_df = (
        load_wp_training_data(
            engine=engine,
            seasons=CALIBRATION_SEASONS,
            game_type="regular",
        )
    )

    print()
    print(
        "Loading S62 holdout..."
    )

    test_df = load_wp_training_data(
        engine=engine,
        seasons=TEST_SEASONS,
        game_type="regular",
    )

    print()
    print(
        "Training fresh base WP model "
        "on S27-S57..."
    )

    base_model = WPModel()

    base_model.fit(
        train_df
    )

    # ------------------------------------------------------
    # Base predictions on the calibration seasons
    # ------------------------------------------------------

    calibration_raw_wp = (
        base_model.predict_wp(
            calibration_df
        )
    )

    calibration_y = (
        calibration_df[
            "possession_won"
        ]
        .astype(int)
        .to_numpy()
    )

    # ------------------------------------------------------
    # Platt calibration
    #
    # Logistic regression on the logit of the base model's
    # probabilities.
    # ------------------------------------------------------

    calibration_x = logit(
        calibration_raw_wp
    ).reshape(
        -1,
        1,
    )

    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=42,
    )

    calibrator.fit(
        calibration_x,
        calibration_y,
    )

    print()
    print(
        "Platt calibration:"
    )

    print(
        "Intercept:",
        float(
            calibrator.intercept_[0]
        ),
    )

    print(
        "Slope:",
        float(
            calibrator.coef_[0][0]
        ),
    )

    # ------------------------------------------------------
    # Evaluate on untouched S62
    # ------------------------------------------------------

    test_raw_wp = (
        base_model.predict_wp(
            test_df
        )
    )

    test_y = (
        test_df[
            "possession_won"
        ]
        .astype(int)
        .to_numpy()
    )

    test_calibrated_wp = (
        calibrator.predict_proba(
            logit(
                test_raw_wp
            ).reshape(
                -1,
                1,
            )
        )[:, 1]
    )

    raw_result = evaluate(
        test_y,
        test_raw_wp,
    )

    calibrated_result = evaluate(
        test_y,
        test_calibrated_wp,
    )

    print_metrics(
        "S62 RAW TEMPORAL MODEL",
        raw_result,
    )

    print_metrics(
        "S62 PLATT-CALIBRATED MODEL",
        calibrated_result,
    )

    print()
    print(
        "===== CHANGE FROM CALIBRATION ====="
    )

    print(
        "Brier change:",
        round(
            calibrated_result["brier"]
            - raw_result["brier"],
            5,
        ),
    )

    print(
        "Log-loss change:",
        round(
            calibrated_result["log_loss"]
            - raw_result["log_loss"],
            5,
        ),
    )

    print(
        "ECE change:",
        round(
            calibrated_result["ece"]
            - raw_result["ece"],
            5,
        ),
    )

    print()
    print(
        "===== RAW S62 CALIBRATION ====="
    )

    print(
        raw_result["table"][
            [
                "bucket",
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
        "===== CALIBRATED S62 CALIBRATION ====="
    )

    print(
        calibrated_result["table"][
            [
                "bucket",
                "states",
                "predicted_wp",
                "actual_win_rate",
                "difference",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()