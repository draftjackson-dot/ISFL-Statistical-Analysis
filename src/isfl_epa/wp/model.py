from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from isfl_epa.wp.training import (
    WP_FEATURE_COLS,
    load_wp_training_data,
)


class WPModel:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )

    def fit(self, df: pd.DataFrame) -> None:
        X = df[WP_FEATURE_COLS]
        y = df["possession_won"].astype(int)
        self.model.fit(X, y)

    def predict_wp(self, df: pd.DataFrame):
        X = df[WP_FEATURE_COLS]
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str):
        return joblib.load(path)


def evaluate_wp_model(
    model: WPModel,
    df: pd.DataFrame,
) -> dict:
    y = df["possession_won"].astype(int)
    wp = model.predict_wp(df)

    return {
        "rows": len(df),
        "brier": brier_score_loss(y, wp),
        "log_loss": log_loss(y, wp),
        "roc_auc": roc_auc_score(y, wp),
        "mean_wp": float(wp.mean()),
        "actual_win_rate": float(y.mean()),
    }


def train_and_test_wp(
    engine,
    train_seasons: list[int],
    test_seasons: list[int],
):
    print("Loading training data...")
    train_df = load_wp_training_data(
        engine,
        train_seasons,
    )

    print()
    print("Training rows:", len(train_df))

    print()
    print("Loading test data...")
    test_df = load_wp_training_data(
        engine,
        test_seasons,
    )

    print()
    print("Test rows:", len(test_df))

    model = WPModel()
    model.fit(train_df)

    metrics = evaluate_wp_model(
        model,
        test_df,
    )

    return model, metrics, test_df