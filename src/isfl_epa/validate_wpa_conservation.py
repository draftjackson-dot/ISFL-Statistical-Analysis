import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from isfl_epa.storage.database import get_engine
from isfl_epa.wp.training import load_wp_training_data


FIRST_SEASON = 27
LAST_SEASON = 62

ROW_TOLERANCE = 1e-10
CHAIN_WARNING = 0.01
GAME_WARNING = 0.01
TERMINAL_WARNING = 0.01


def main():
    parser = argparse.ArgumentParser(
        description="Validate stored WPA conservation."
    )
    parser.add_argument(
        "--game-type",
        default="regular",
        choices=["regular", "playoff"],
    )
    args = parser.parse_args()

    game_type = args.game_type

    engine = get_engine()

    seasons = list(
        range(
            FIRST_SEASON,
            LAST_SEASON + 1,
        )
    )

    print(
        f"Loading valid {game_type} WP states "
        f"S{FIRST_SEASON}-S{LAST_SEASON}..."
    )

    states = load_wp_training_data(
        engine=engine,
        seasons=seasons,
        game_type=game_type,
    )

    print()
    print(
        f"Valid states: {len(states):,}"
    )

    # ------------------------------------------------------
    # Load stored WPA
    # ------------------------------------------------------

    sql = f"""
        SELECT
            pw.play_id,
            pw.game_id,
            pw.season,
            pw.wp_before,
            pw.wp_after,
            pw.wpa
        FROM play_wp pw
        JOIN plays p
          ON p.id = pw.play_id
        WHERE pw.season BETWEEN
            {FIRST_SEASON}
            AND {LAST_SEASON}
          AND p.game_type = '{game_type}'
        ORDER BY
            pw.game_id,
            pw.play_id
    """

    play_wp = pd.read_sql(
        sql,
        engine,
    )

    print(
        f"play_wp rows: {len(play_wp):,}"
    )

    # ------------------------------------------------------
    # Merge stored WPA with the state information used by
    # the WP model.
    #
    # id in the training dataframe is the plays.id value.
    # ------------------------------------------------------

    needed_columns = [
        "id",
        "game_id",
        "season",
        "play_index",
        "is_home",
        "possession_won",
    ]

    missing = [
        column
        for column in needed_columns
        if column not in states.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns from "
            f"WP training data: {missing}"
        )

    state_info = states[
        needed_columns
    ].copy()

    merged = play_wp.merge(
        state_info,
        left_on="play_id",
        right_on="id",
        how="inner",
        suffixes=(
            "_wp",
            "_state",
        ),
    )

    print(
        f"Merged valid WPA rows: "
        f"{len(merged):,}"
    )

    # ------------------------------------------------------
    # Keep states where the entire WPA transition exists.
    # ------------------------------------------------------

    complete = merged.dropna(
        subset=[
            "wp_before",
            "wp_after",
            "wpa",
            "is_home",
        ]
    ).copy()

    print(
        f"Complete WPA transitions: "
        f"{len(complete):,}"
    )

    complete = complete.sort_values(
        [
            "game_id_wp",
            "play_index",
            "play_id",
        ]
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------------
    # Convert everything to one fixed perspective:
    # the HOME team's win probability.
    #
    # If the possession team is home:
    #     home WP = current possession WP
    #
    # If possession team is away:
    #     home WP = 1 - current possession WP
    #
    # WPA needs the same sign flip.
    # ------------------------------------------------------

    home_possession = (
        complete["is_home"]
        .astype(int)
        == 1
    )

    complete[
        "home_wp_before"
    ] = np.where(
        home_possession,
        complete["wp_before"],
        1.0
        - complete["wp_before"],
    )

    complete[
        "home_wp_after"
    ] = np.where(
        home_possession,
        complete["wp_after"],
        1.0
        - complete["wp_after"],
    )

    complete[
        "home_wpa"
    ] = np.where(
        home_possession,
        complete["wpa"],
        -complete["wpa"],
    )

    # ------------------------------------------------------
    # CHECK 1:
    #
    # Every individual row should satisfy:
    #
    # home WPA =
    # home WP after - home WP before
    # ------------------------------------------------------

    complete[
        "row_identity_error"
    ] = (
        complete["home_wp_after"]
        - complete["home_wp_before"]
        - complete["home_wpa"]
    )

    complete[
        "abs_row_identity_error"
    ] = (
        complete[
            "row_identity_error"
        ].abs()
    )

    print()
    print(
        "===== PLAY-LEVEL WPA IDENTITY ====="
    )

    print(
        "Maximum absolute error:",
        f"{complete['abs_row_identity_error'].max():.12f}",
    )

    print(
        "Mean absolute error:",
        f"{complete['abs_row_identity_error'].mean():.12f}",
    )

    bad_rows = complete[
        complete[
            "abs_row_identity_error"
        ]
        > ROW_TOLERANCE
    ]

    print(
        "Rows exceeding tolerance:",
        f"{len(bad_rows):,}",
    )

    # ------------------------------------------------------
    # CHECK 2:
    #
    # For consecutive valid WPA states:
    #
    # current home_wp_after
    # should equal
    # next home_wp_before.
    #
    # This is the most useful check for possession flips,
    # skipped states, and transition bugs.
    # ------------------------------------------------------

    complete[
        "next_game_id"
    ] = complete[
        "game_id_wp"
    ].shift(
        -1
    )

    complete[
        "next_home_wp_before"
    ] = complete[
        "home_wp_before"
    ].shift(
        -1
    )

    same_game_next = (
        complete[
            "game_id_wp"
        ]
        == complete[
            "next_game_id"
        ]
    )

    complete[
        "chain_gap"
    ] = np.nan

    complete.loc[
        same_game_next,
        "chain_gap",
    ] = (
        complete.loc[
            same_game_next,
            "next_home_wp_before",
        ]
        - complete.loc[
            same_game_next,
            "home_wp_after",
        ]
    )

    complete[
        "abs_chain_gap"
    ] = (
        complete[
            "chain_gap"
        ].abs()
    )

    chain_rows = complete[
        same_game_next
    ].copy()

    print()
    print(
        "===== STATE-TO-STATE CONTINUITY ====="
    )

    print(
        "Transitions checked:",
        f"{len(chain_rows):,}",
    )

    print(
        "Mean absolute gap:",
        f"{chain_rows['abs_chain_gap'].mean():.6f}",
    )

    print(
        "Median absolute gap:",
        f"{chain_rows['abs_chain_gap'].median():.6f}",
    )

    print(
        "Maximum absolute gap:",
        f"{chain_rows['abs_chain_gap'].max():.6f}",
    )

    chain_warning_rows = chain_rows[
        chain_rows[
            "abs_chain_gap"
        ]
        > CHAIN_WARNING
    ]

    print(
        f"Transitions with gap > "
        f"{CHAIN_WARNING:.2f}:",
        f"{len(chain_warning_rows):,}",
    )

    # ------------------------------------------------------
    # CHECK 3:
    #
    # Game-level telescoping:
    #
    # opening home WP
    # + sum(home WPA)
    # = closing home WP
    #
    # Any residual indicates that some state transitions
    # did not fully telescope.
    # ------------------------------------------------------

    game_rows = []

    for game_id, group in complete.groupby(
        "game_id_wp",
        sort=False,
    ):
        group = group.sort_values(
            [
                "play_index",
                "play_id",
            ]
        )

        first = group.iloc[0]
        last = group.iloc[-1]

        opening_home_wp = float(
            first[
                "home_wp_before"
            ]
        )

        closing_home_wp = float(
            last[
                "home_wp_after"
            ]
        )

        total_home_wpa = float(
            group[
                "home_wpa"
            ].sum()
        )

        reconstructed_closing = (
            opening_home_wp
            + total_home_wpa
        )

        conservation_residual = (
            reconstructed_closing
            - closing_home_wp
        )

        # ----------------------------------------------
        # Actual final result from the perspective of
        # the home team.
        #
        # possession_won is the eventual game outcome
        # from the current possession team's perspective.
        # ----------------------------------------------

        if pd.isna(
            last["possession_won"]
        ):
            home_result = np.nan
        else:
            if int(
                last["is_home"]
            ) == 1:
                home_result = float(
                    last[
                        "possession_won"
                    ]
                )
            else:
                home_result = (
                    1.0
                    - float(
                        last[
                            "possession_won"
                        ]
                    )
                )

        if pd.isna(
            home_result
        ):
            terminal_gap = np.nan
        else:
            terminal_gap = (
                closing_home_wp
                - home_result
            )

        game_rows.append(
            {
                "game_id": game_id,
                "season": int(
                    first[
                        "season_wp"
                    ]
                ),
                "states": len(
                    group
                ),
                "opening_home_wp":
                    opening_home_wp,
                "sum_home_wpa":
                    total_home_wpa,
                "reconstructed_closing_wp":
                    reconstructed_closing,
                "closing_home_wp":
                    closing_home_wp,
                "conservation_residual":
                    conservation_residual,
                "abs_conservation_residual":
                    abs(
                        conservation_residual
                    ),
                "home_result":
                    home_result,
                "terminal_gap":
                    terminal_gap,
                "abs_terminal_gap":
                    (
                        abs(
                            terminal_gap
                        )
                        if not pd.isna(
                            terminal_gap
                        )
                        else np.nan
                    ),
            }
        )

    games = pd.DataFrame(
        game_rows
    )

    print()
    print(
        "===== GAME-LEVEL WPA CONSERVATION ====="
    )

    print(
        "Games checked:",
        f"{len(games):,}",
    )

    print(
        "Mean absolute conservation residual:",
        f"{games['abs_conservation_residual'].mean():.6f}",
    )

    print(
        "Median absolute conservation residual:",
        f"{games['abs_conservation_residual'].median():.6f}",
    )

    print(
        "Maximum absolute conservation residual:",
        f"{games['abs_conservation_residual'].max():.6f}",
    )

    bad_games = games[
        games[
            "abs_conservation_residual"
        ]
        > GAME_WARNING
    ]

    print(
        f"Games with residual > "
        f"{GAME_WARNING:.2f}:",
        f"{len(bad_games):,}",
    )

    # ------------------------------------------------------
    # CHECK 4:
    #
    # Does the final stored WP actually reach the final
    # game result?
    #
    # Winner should end at 1.0,
    # loser at 0.0.
    # ------------------------------------------------------

    terminal_valid = games.dropna(
        subset=[
            "terminal_gap"
        ]
    )

    print()
    print(
        "===== TERMINAL GAME STATE ====="
    )

    print(
        "Games with final result:",
        f"{len(terminal_valid):,}",
    )

    print(
        "Mean absolute terminal gap:",
        f"{terminal_valid['abs_terminal_gap'].mean():.6f}",
    )

    print(
        "Maximum absolute terminal gap:",
        f"{terminal_valid['abs_terminal_gap'].max():.6f}",
    )

    terminal_bad = terminal_valid[
        terminal_valid[
            "abs_terminal_gap"
        ]
        > TERMINAL_WARNING
    ]

    print(
        f"Games with terminal gap > "
        f"{TERMINAL_WARNING:.2f}:",
        f"{len(terminal_bad):,}",
    )

    # ------------------------------------------------------
    # Largest suspicious state transitions
    # ------------------------------------------------------

    print()
    print(
        "===== LARGEST STATE-TO-STATE GAPS ====="
    )

    columns = [
        "season_wp",
        "game_id_wp",
        "play_id",
        "play_index",
        "home_wp_after",
        "next_home_wp_before",
        "chain_gap",
    ]

    print(
        chain_rows.sort_values(
            "abs_chain_gap",
            ascending=False,
        )[
            columns
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Largest game conservation residuals
    # ------------------------------------------------------

    print()
    print(
        "===== LARGEST GAME RESIDUALS ====="
    )

    print(
        games.sort_values(
            "abs_conservation_residual",
            ascending=False,
        )
        .head(25)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Save diagnostic files
    # ------------------------------------------------------

    output_dir = (
        Path("wp_validation")
        / game_type
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    game_output = (
        output_dir
        / "wpa-game-conservation.csv"
    )

    chain_output = (
        output_dir
        / "wpa-chain-gap-warnings.csv"
    )

    identity_output = (
        output_dir
        / "wpa-play-identity-warnings.csv"
    )

    games.to_csv(
        game_output,
        index=False,
    )

    chain_warning_rows.to_csv(
        chain_output,
        index=False,
    )

    bad_rows.to_csv(
        identity_output,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        game_output
    )

    print(
        chain_output
    )

    print(
        identity_output
    )


if __name__ == "__main__":
    main()