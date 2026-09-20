import pandas as pd
from sqlalchemy import bindparam, select, text
from sqlalchemy.engine import Engine

from isfl_epa.storage.database import (
    get_team_id_to_abbr,
    plays_table,
)


WP_FEATURE_COLS = [
    "down",
    "distance",
    "yardline_100",
    "score_differential",
    "game_seconds",
    "is_home",
    "is_overtime",
    "is_kickoff",
    "is_onside_kick",
    "is_two_point_attempt",
]


def _resolve_possession_teams(
    engine: Engine,
    seasons: list[int],
) -> dict[tuple[int, int], str]:
    """Map (game_id, possession_team_id) to team abbreviation."""
    mapping = {}

    with engine.connect() as conn:
        for season in seasons:
            ptid_to_team = get_team_id_to_abbr(
                engine,
                season,
            )

            stmt = (
                select(
                    plays_table.c.game_id,
                    plays_table.c.possession_team_id,
                )
                .where(plays_table.c.season == season)
                .where(
                    plays_table.c.possession_team_id.isnot(None)
                )
                .distinct()
            )

            for row in conn.execute(stmt):
                team = ptid_to_team.get(
                    int(row.possession_team_id)
                )

                if team:
                    mapping[
                        (
                            row.game_id,
                            row.possession_team_id,
                        )
                    ] = team

    return mapping


def load_wp_training_data(
    engine: Engine,
    seasons: list[int],
    game_type: str = "regular",
) -> pd.DataFrame:
    sql = text(
        """
        SELECT
            p.id,
            p.game_id,
            p.season,
            p.description,
            p.play_index,
            p.quarter,
            p.clock,
            p.play_type,
            p.down,
            p.distance,
            p.yardline_100,
            p.game_seconds,
            p.score_home,
            p.score_away,
            p.home_team,
            p.away_team,
            p.possession_team_id,
            p.counts_as_play,
            h.points_for AS final_home_score,
            a.points_for AS final_away_score
        FROM plays p
        JOIN team_games h
          ON h.game_id = p.game_id
         AND h.is_home = TRUE
         AND h.game_type = :game_type
        JOIN team_games a
          ON a.game_id = p.game_id
         AND a.is_home = FALSE
         AND a.game_type = :game_type
        WHERE p.season IN :seasons
          AND p.game_type = :game_type
          AND p.counts_as_play = TRUE
        """
    ).bindparams(
        bindparam("seasons", expanding=True)
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "seasons": seasons,
                "game_type": game_type,
            },
        )

    print("RAW SQL ROWS:", len(df))

    if df.empty:
        return df

    # ----------------------------------------------------------
    # Sort chronologically before constructing pre-play scores.
    #
    # score_home / score_away represent the score AFTER the
    # current play, so the previous row's score is the score
    # BEFORE this play.
    # ----------------------------------------------------------

    df = df.sort_values(
        ["game_id", "play_index"]
    ).reset_index(drop=True)

    df["score_home_before"] = (
        df.groupby("game_id")["score_home"]
        .shift(1)
    )

    df["score_away_before"] = (
        df.groupby("game_id")["score_away"]
        .shift(1)
    )

    # First playable state of each game begins 0-0.
    first_play_mask = (
        df.groupby("game_id").cumcount() == 0
    )

    df.loc[
        first_play_mask,
        "score_home_before",
    ] = 0

    df.loc[
        first_play_mask,
        "score_away_before",
    ] = 0

    # ----------------------------------------------------------
    # Resolve possession team
    # ----------------------------------------------------------

    possession_map = _resolve_possession_teams(
        engine,
        seasons,
    )

    df["possession_team"] = [
        possession_map.get(
            (game_id, possession_team_id)
        )
        for game_id, possession_team_id in zip(
            df["game_id"],
            df["possession_team_id"],
        )
    ]

    # ----------------------------------------------------------
    # Special-state indicators
    # ----------------------------------------------------------

    df["is_kickoff"] = (
        df["play_type"] == "kickoff"
    ).astype(int)

    df["is_onside_kick"] = (
        (df["play_type"] == "kickoff")
        & (
            df["description"]
            .fillna("")
            .str.contains(
                "onside",
                case=False,
                regex=False,
            )
        )
    ).astype(int)

    description_lower = (
        df["description"]
        .fillna("")
        .str.lower()
    )

    df["is_two_point_attempt"] = (
        description_lower.str.contains(
            "2 point conversion"
        )
    ).astype(int)

    kickoff_mask = (
        df["play_type"] == "kickoff"
    )

    two_point_mask = (
        df["is_two_point_attempt"] == 1
    )

        # ----------------------------------------------------------
    # Normalize kickoff states.
    #
    # Raw kickoff yardline_100 can reflect the result of the
    # kickoff/return, which would leak post-play field position
    # into WP-before. Use a fixed synthetic pre-kick state.
    # ----------------------------------------------------------

    df.loc[
        kickoff_mask,
        "down",
    ] = 1

    df.loc[
        kickoff_mask,
        "distance",
    ] = 10

    df.loc[
        kickoff_mask,
        "yardline_100",
    ] = 75

    # ----------------------------------------------------------
    # Normalize two-point conversion states
    #
    # Treat all conversion attempts as a consistent synthetic
    # state at the opponent's 2-yard line.
    # ----------------------------------------------------------

    df.loc[
        two_point_mask,
        "down",
    ] = 1

    df.loc[
        two_point_mask,
        "distance",
    ] = 2

    df.loc[
        two_point_mask,
        "yardline_100",
    ] = 98

    # ----------------------------------------------------------
    # Possession perspective
    # ----------------------------------------------------------

    df["is_home"] = (
        df["possession_team"] == df["home_team"]
    ).astype(int)

    df["score_differential"] = (
        df["score_home_before"]
        - df["score_away_before"]
    )

    away_mask = (
        df["is_home"] == 0
    )

    df.loc[
        away_mask,
        "score_differential",
    ] *= -1

    df["is_overtime"] = (
        df["quarter"] >= 5
    ).astype(int)

    # ----------------------------------------------------------
    # Win/loss target
    # ----------------------------------------------------------

    tied_final = (
        df["final_home_score"]
        == df["final_away_score"]
    )

    home_won = (
        df["final_home_score"]
        > df["final_away_score"]
    )

    df["possession_won"] = 0.0

    # Home possession + home eventually won.
    df.loc[
        (df["is_home"] == 1)
        & home_won,
        "possession_won",
    ] = 1.0

    # Away possession + away eventually won.
    df.loc[
        (df["is_home"] == 0)
        & ~home_won
        & ~tied_final,
        "possession_won",
    ] = 1.0

    # Ignore tied games for classifier training.
    df.loc[
        tied_final,
        "possession_won",
    ] = pd.NA

    # ----------------------------------------------------------
    # Valid WP states
    # ----------------------------------------------------------

    valid = (
        df["possession_team"].notna()
        & df["down"].notna()
        & df["distance"].notna()
        & df["yardline_100"].notna()
        & df["game_seconds"].notna()
        & df["score_home_before"].notna()
        & df["score_away_before"].notna()
        & df["score_differential"].notna()
        & df["possession_won"].notna()
    )

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------

    print("NULL COUNTS:")
    print(
        df[
            [
                "possession_team",
                "down",
                "distance",
                "yardline_100",
                "game_seconds",
                "score_home_before",
                "score_away_before",
                "score_differential",
                "possession_won",
            ]
        ].isna().sum()
    )

    print()
    print("KICKOFF STATES:")
    print(
        int(
            (
                valid
                & (df["is_kickoff"] == 1)
            ).sum()
        )
    )

    print()
    print("ONSIDE KICK STATES:")
    print(
        int(
            (
                valid
                & (df["is_onside_kick"] == 1)
            ).sum()
        )
    )

    print()
    print("TWO-POINT STATES:")
    print(
        int(
            (
                valid
                & (df["is_two_point_attempt"] == 1)
            ).sum()
        )
    )

    print()
    print("POSSESSION MAP SAMPLE:")
    print(list(possession_map.items())[:20])

    print()
    print("VALID ROWS:", int(valid.sum()))

    return df.loc[valid].copy()