from pathlib import Path

import pandas as pd

from isfl_epa.storage.database import get_engine
from isfl_epa.wp.model import train_and_test_wp


engine = get_engine()

model, metrics, test_df = train_and_test_wp(
    engine=engine,
    train_seasons=list(range(27, 62)),
    test_seasons=[62],
)

print()
print("===== S62 WP MODEL TEST =====")

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

test_df = test_df.copy()
test_df["wp"] = model.predict_wp(test_df)

test_df["wp_bucket"] = pd.cut(
    test_df["wp"],
    bins=[
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ],
    include_lowest=True,
)

calibration = (
    test_df
    .groupby("wp_bucket", observed=True)
    .agg(
        predicted_wp=("wp", "mean"),
        actual_win_rate=("possession_won", "mean"),
        states=("wp", "size"),
    )
)

print()
print("===== CALIBRATION =====")
print(calibration.to_string())

model_path = Path("models/wp_model_2022.joblib")
model.save(model_path)

print()
print(f"Saved model to {model_path}")

def time_bucket(seconds):
    if seconds > 1800:
        return "1st half"
    if seconds > 900:
        return "3rd quarter"
    if seconds > 300:
        return "4th quarter >5m"
    if seconds > 120:
        return "final 5m"
    if seconds > 0:
        return "final 2m"
    return "OT / 0"

test_df["time_bucket"] = test_df["game_seconds"].apply(
    time_bucket
)

time_calibration = (
    test_df
    .groupby("time_bucket")
    .agg(
        states=("wp", "size"),
        predicted_wp=("wp", "mean"),
        actual_win_rate=("possession_won", "mean"),
    )
)

print()
print("===== CALIBRATION BY TIME =====")
print(time_calibration.to_string())