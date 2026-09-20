import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from isfl_epa.wp.model import WPModel
from isfl_epa.wp.training import load_wp_training_data


# These play types may appear after the final meaningful WP state
# without representing a new meaningful game-state transition.
#
# Example:
#   onside kick -> kneel -> end of game
#
# In that case, the onside kick should still be allowed to receive
# the terminal game result.
TERMINAL_NONBLOCKING_PLAY_TYPES = {
    "kneel",
    "quarter_marker",
}


def _load_terminal_play_info(
    engine: Engine,
    season: int,
    game_type: str,
) -> dict:
    """
    Load all counting plays for each game so we can determine
    whether any meaningful raw plays remain after the last valid
    WP state.

    Returns
    -------
    dict
        game_id -> list of dicts containing:
            play_index
            play_type
    """

    terminal_sql = text(
        """
        SELECT
            game_id,
            play_index,
            play_type
        FROM plays
        WHERE season = :season
          AND game_type = :game_type
          AND counts_as_play = TRUE
        ORDER BY
            game_id,
            play_index
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(
            terminal_sql,
            {
                "season": season,
                "game_type": game_type,
            },
        ).fetchall()

    plays_by_game = {}

    for row in rows:
        plays_by_game.setdefault(
            row.game_id,
            [],
        ).append(
            {
                "play_index": row.play_index,
                "play_type": row.play_type,
            }
        )

    return plays_by_game


def _has_meaningful_later_play(
    plays_by_game: dict,
    game_id: int,
    current_play_index: int,
) -> bool:
    """
    Return True if a meaningful counting play exists after the
    current valid WP state.

    Kneels and quarter markers are treated as non-blocking terminal
    bookkeeping / clock-killing rows.

    This deliberately remains conservative: any other later play
    type prevents us from forcing the current state directly to the
    final game result.
    """

    game_plays = plays_by_game.get(
        game_id,
        [],
    )

    for play in game_plays:
        if (
            play["play_index"]
            <= current_play_index
        ):
            continue

        play_type = play[
            "play_type"
        ]

        if (
            play_type
            not in TERMINAL_NONBLOCKING_PLAY_TYPES
        ):
            return True

    return False


def compute_play_wpa(
    engine: Engine,
    season: int,
    model: WPModel,
    game_type: str = "regular",
) -> pd.DataFrame:
    """
    Compute WP before, WP after, and WPA from the perspective
    of the team possessing the ball before each play.
    """

    df = load_wp_training_data(
        engine,
        [season],
        game_type=game_type,
    )

    if df.empty:
        return df

    df = df.copy()

    # WP for every valid pre-play state.
    df["wp_before"] = model.predict_wp(
        df
    )

    # Make sure states are chronological.
    df = df.sort_values(
        [
            "game_id",
            "play_index",
        ]
    ).reset_index(
        drop=True
    )

    # ----------------------------------------------------------
    # Load all raw counting plays.
    #
    # We use these only for determining whether meaningful
    # football action remains after the final valid WP state.
    # ----------------------------------------------------------

    plays_by_game = (
        _load_terminal_play_info(
            engine=engine,
            season=season,
            game_type=game_type,
        )
    )

    df["wp_after"] = pd.NA

    # ----------------------------------------------------------
    # Process one game at a time.
    # ----------------------------------------------------------

    for game_id, game_df in df.groupby(
        "game_id",
        sort=False,
    ):
        indices = game_df.index.tolist()

        for position, idx in enumerate(
            indices
        ):
            current = df.loc[
                idx
            ]

            # --------------------------------------------------
            # Normal case:
            # another valid WP state exists after this play.
            # --------------------------------------------------

            if position < len(indices) - 1:
                next_idx = indices[
                    position + 1
                ]

                nxt = df.loc[
                    next_idx
                ]

                next_wp = float(
                    nxt[
                        "wp_before"
                    ]
                )

                # ----------------------------------------------
                # Regulation -> overtime boundary.
                #
                # Do not let the final regulation play absorb
                # the modeled value of the later OT kickoff.
                #
                # If regulation ends tied, use a neutral 50%
                # boundary state.
                # ----------------------------------------------

                crosses_into_overtime = (
                    current["quarter"] <= 4
                    and nxt["quarter"] >= 5
                )

                game_tied_after_current = (
                    current[
                        "score_home"
                    ]
                    == current[
                        "score_away"
                    ]
                )

                if (
                    crosses_into_overtime
                    and game_tied_after_current
                ):
                    wp_after = 0.5

                # ----------------------------------------------
                # Same team still possesses the ball.
                # ----------------------------------------------

                elif (
                    current[
                        "possession_team"
                    ]
                    == nxt[
                        "possession_team"
                    ]
                ):
                    wp_after = next_wp

                # ----------------------------------------------
                # Possession changed.
                #
                # next_wp is from the new possession team's
                # perspective. Convert it back to the current
                # possession team's perspective.
                # ----------------------------------------------

                else:
                    wp_after = (
                        1.0
                        - next_wp
                    )

                df.at[
                    idx,
                    "wp_after",
                ] = wp_after

                continue

            # --------------------------------------------------
            # No later valid WP state exists.
            #
            # Determine whether any meaningful raw plays remain.
            #
            # If meaningful action remains, do not manufacture
            # the entire remaining game result on this state.
            #
            # If only kneels / quarter markers remain, this is
            # effectively the final meaningful WP transition and
            # can receive the terminal result.
            # --------------------------------------------------

            has_meaningful_later_play = (
                _has_meaningful_later_play(
                    plays_by_game=(
                        plays_by_game
                    ),
                    game_id=game_id,
                    current_play_index=(
                        current[
                            "play_index"
                        ]
                    ),
                )
            )

            if has_meaningful_later_play:
                df.at[
                    idx,
                    "wp_after",
                ] = pd.NA

                continue

            # --------------------------------------------------
            # Terminal state.
            # --------------------------------------------------

            final_home = current[
                "final_home_score"
            ]

            final_away = current[
                "final_away_score"
            ]

            if (
                final_home
                == final_away
            ):
                wp_after = 0.5

            elif (
                current["is_home"]
                == 1
            ):
                wp_after = (
                    1.0
                    if (
                        final_home
                        > final_away
                    )
                    else 0.0
                )

            else:
                wp_after = (
                    1.0
                    if (
                        final_away
                        > final_home
                    )
                    else 0.0
                )

            df.at[
                idx,
                "wp_after",
            ] = wp_after

    # ----------------------------------------------------------
    # WPA
    # ----------------------------------------------------------

    df["wp_after"] = pd.to_numeric(
        df["wp_after"],
        errors="coerce",
    )

    df["wpa"] = (
        df["wp_after"]
        - df["wp_before"]
    )

    return df


def write_play_wpa(
    engine: Engine,
    df: pd.DataFrame,
    season: int,
    game_type: str = "regular",
) -> int:
    """
    Replace play_wp rows for one season and game type.

    play_wp does not currently store game_type directly, so deletion
    is scoped through the corresponding rows in plays. This prevents
    rebuilding playoff WPA from deleting regular-season WPA, and vice
    versa.
    """

    if df.empty:
        return 0

    rows_df = df[
        [
            "id",
            "game_id",
            "season",
            "wp_before",
            "wp_after",
            "wpa",
        ]
    ].rename(
        columns={
            "id": "play_id",
        }
    )

    # Convert pandas NaN/NA to Python None so PostgreSQL receives
    # NULL rather than an invalid numeric value.
    rows_df = (
        rows_df
        .astype(object)
        .where(
            pd.notna(
                rows_df
            ),
            None,
        )
    )

    rows = rows_df.to_dict(
        orient="records"
    )

    # Delete only WPA rows that belong to plays from the requested
    # season and game type.
    delete_sql = text(
        """
        DELETE FROM play_wp
        WHERE play_id IN (
            SELECT id
            FROM plays
            WHERE season = :season
              AND game_type = :game_type
        )
        """
    )

    insert_sql = text(
        """
        INSERT INTO play_wp (
            play_id,
            game_id,
            season,
            wp_before,
            wp_after,
            wpa
        )
        VALUES (
            :play_id,
            :game_id,
            :season,
            :wp_before,
            :wp_after,
            :wpa
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(
            delete_sql,
            {
                "season": season,
                "game_type": game_type,
            },
        )

        conn.execute(
            insert_sql,
            rows,
        )

    return len(
        rows
    )