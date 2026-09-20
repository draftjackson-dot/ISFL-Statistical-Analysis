import argparse

from isfl_epa.storage.database import get_engine
from isfl_epa.wp.player_wpa import (
    build_player_season_wpa,
    write_player_season_wpa,
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
        f"Building player WPA for "
        f"S{args.season} {args.game_type}..."
    )

    df = build_player_season_wpa(
        engine=engine,
        season=args.season,
        game_type=args.game_type,
    )

    print()
    print("Player rows:", len(df))

    if not df.empty:
        print()
        print("Top passing WPA:")
        print(
            df.sort_values(
                "pass_wpa",
                ascending=False,
            )[
                [
                    "player",
                    "pass_wpa",
                    "dropbacks",
                    "wpa_per_dropback",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        print()
        print("Top rushing WPA:")
        print(
            df.sort_values(
                "rush_wpa",
                ascending=False,
            )[
                [
                    "player",
                    "rush_wpa",
                    "rush_attempts",
                    "wpa_per_rush",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    written = write_player_season_wpa(
        engine=engine,
        df=df,
        season=args.season,
        game_type=args.game_type,
    )

    print()
    print(
        f"Wrote {written} player rows "
        f"for S{args.season} {args.game_type}."
    )


if __name__ == "__main__":
    main()