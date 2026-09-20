import argparse
from pathlib import Path

from isfl_epa.storage.database import get_engine
from isfl_epa.wp.assignment import (
    compute_play_wpa,
    write_play_wpa,
)
from isfl_epa.wp.model import WPModel


MODEL_PATH = Path(
    "models/wp_model_2022.joblib"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--game-type",
        default="regular",
        choices=[
            "regular",
            "playoff",
        ],
    )

    args = parser.parse_args()

    engine = get_engine()

    print(
        f"Loading WP model from "
        f"{MODEL_PATH}..."
    )

    model = WPModel.load(
        MODEL_PATH
    )

    print(
        f"Computing S{args.season} "
        f"{args.game_type} WPA..."
    )

    df = compute_play_wpa(
        engine=engine,
        season=args.season,
        model=model,
        game_type=args.game_type,
    )

    print()
    print("===== WPA SUMMARY =====")
    print("Rows:", len(df))

    if not df.empty:
        print()
        print(
            df[
                [
                    "wp_before",
                    "wp_after",
                    "wpa",
                ]
            ].describe()
        )

        print()
        print("Largest positive WPA:")
        print(
            df[
                [
                    "game_id",
                    "play_index",
                    "possession_team",
                    "score_differential",
                    "game_seconds",
                    "wp_before",
                    "wp_after",
                    "wpa",
                ]
            ]
            .sort_values(
                "wpa",
                ascending=False,
            )
            .head(20)
            .to_string(index=False)
        )

        print()
        print("Largest negative WPA:")
        print(
            df[
                [
                    "game_id",
                    "play_index",
                    "possession_team",
                    "score_differential",
                    "game_seconds",
                    "wp_before",
                    "wp_after",
                    "wpa",
                ]
            ]
            .sort_values("wpa")
            .head(20)
            .to_string(index=False)
        )

    written = write_play_wpa(
        engine=engine,
        df=df,
        season=args.season,
        game_type=args.game_type,
    )

    print()
    print(
        f"Wrote {written} rows "
        f"to play_wp for "
        f"S{args.season} "
        f"{args.game_type}."
    )


if __name__ == "__main__":
    main()