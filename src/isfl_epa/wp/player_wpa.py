import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from isfl_epa.wp.training import _resolve_possession_teams


def build_player_season_wpa(
    engine: Engine,
    season: int,
    game_type: str = "regular",
) -> pd.DataFrame:
    sql = text(
        """
        SELECT
            p.id AS play_id,
            p.game_id,
            p.season,
            p.game_type,
            p.play_type,
            p.home_team,
            p.away_team,
            p.possession_team_id,

            p.passer,
            p.player_id_passer,

            p.rusher,
            p.player_id_rusher,

            w.wpa
        FROM plays p
        JOIN play_wp w
          ON w.play_id = p.id
        WHERE p.season = :season
          AND p.game_type = :game_type
          AND w.wpa IS NOT NULL
          AND p.play_type IN (
              'pass',
              'sack',
              'rush'
          )
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            sql,
            conn,
            params={
                "season": season,
                "game_type": game_type,
            },
        )

    if df.empty:
        return df

    # ----------------------------------------------------------
    # Resolve possession team abbreviations
    # ----------------------------------------------------------

    possession_map = _resolve_possession_teams(
        engine,
        [season],
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
    # Passing WPA
    # ----------------------------------------------------------

    pass_plays = df[
        df["play_type"].isin(["pass", "sack"])
        & df["player_id_passer"].notna()
    ].copy()

    if not pass_plays.empty:
        pass_agg = (
            pass_plays
            .groupby("player_id_passer")
            .agg(
                player=("passer", "first"),
                pass_wpa=("wpa", "sum"),
                dropbacks=("wpa", "count"),
            )
            .reset_index()
            .rename(
                columns={
                    "player_id_passer": "player_id",
                }
            )
        )
    else:
        pass_agg = pd.DataFrame(
            columns=[
                "player_id",
                "player",
                "pass_wpa",
                "dropbacks",
            ]
        )

    # ----------------------------------------------------------
    # Rushing WPA
    # ----------------------------------------------------------

    rush_plays = df[
        (df["play_type"] == "rush")
        & df["player_id_rusher"].notna()
    ].copy()

    if not rush_plays.empty:
        rush_agg = (
            rush_plays
            .groupby("player_id_rusher")
            .agg(
                player=("rusher", "first"),
                rush_wpa=("wpa", "sum"),
                rush_attempts=("wpa", "count"),
            )
            .reset_index()
            .rename(
                columns={
                    "player_id_rusher": "player_id",
                }
            )
        )
    else:
        rush_agg = pd.DataFrame(
            columns=[
                "player_id",
                "player",
                "rush_wpa",
                "rush_attempts",
            ]
        )

    # ----------------------------------------------------------
    # Merge passing + rushing
    # ----------------------------------------------------------

    result = pd.merge(
        pass_agg,
        rush_agg,
        on="player_id",
        how="outer",
        suffixes=("_pass", "_rush"),
    )

    if result.empty:
        return result

    result["player"] = (
        result["player_pass"]
        .combine_first(result["player_rush"])
    )

    result["pass_wpa"] = (
        result["pass_wpa"]
        .fillna(0.0)
    )

    result["rush_wpa"] = (
        result["rush_wpa"]
        .fillna(0.0)
    )

    result["dropbacks"] = (
        result["dropbacks"]
        .fillna(0)
        .astype(int)
    )

    result["rush_attempts"] = (
        result["rush_attempts"]
        .fillna(0)
        .astype(int)
    )

    result["total_wpa"] = (
        result["pass_wpa"]
        + result["rush_wpa"]
    )

    result["wpa_per_dropback"] = (
        result["pass_wpa"]
        / result["dropbacks"].replace(0, pd.NA)
    )

    result["wpa_per_rush"] = (
        result["rush_wpa"]
        / result["rush_attempts"].replace(0, pd.NA)
    )

    result["season"] = season
    result["game_type"] = game_type

    # ----------------------------------------------------------
    # Player -> team lookup
    #
    # Prefer a passing play for passers, otherwise use a rushing
    # play. This gives us the season-level team abbreviation.
    # ----------------------------------------------------------

    passer_teams = (
        df[
            df["player_id_passer"].notna()
            & df["possession_team"].notna()
        ]
        .groupby("player_id_passer")["possession_team"]
        .first()
        .to_dict()
    )

    rusher_teams = (
        df[
            df["player_id_rusher"].notna()
            & df["possession_team"].notna()
        ]
        .groupby("player_id_rusher")["possession_team"]
        .first()
        .to_dict()
    )

    result["team"] = [
        passer_teams.get(player_id)
        or rusher_teams.get(player_id)
        for player_id in result["player_id"]
    ]

    return result[
        [
            "player_id",
            "player",
            "team",
            "season",
            "game_type",
            "pass_wpa",
            "rush_wpa",
            "total_wpa",
            "dropbacks",
            "rush_attempts",
            "wpa_per_dropback",
            "wpa_per_rush",
        ]
    ]


def write_player_season_wpa(
    engine: Engine,
    df: pd.DataFrame,
    season: int,
    game_type: str = "regular",
) -> int:
    if df.empty:
        return 0

    delete_sql = text(
        """
        DELETE FROM player_season_wpa
        WHERE season = :season
          AND game_type = :game_type
        """
    )

    insert_sql = text(
        """
        INSERT INTO player_season_wpa (
            player_id,
            player,
            team,
            season,
            game_type,
            pass_wpa,
            rush_wpa,
            total_wpa,
            dropbacks,
            rush_attempts,
            wpa_per_dropback,
            wpa_per_rush
        )
        VALUES (
            :player_id,
            :player,
            :team,
            :season,
            :game_type,
            :pass_wpa,
            :rush_wpa,
            :total_wpa,
            :dropbacks,
            :rush_attempts,
            :wpa_per_dropback,
            :wpa_per_rush
        )
        """
    )

    rows_df = df.astype(object).where(
        pd.notna(df),
        None,
    )

    rows = rows_df.to_dict(
        orient="records"
    )

    with engine.begin() as conn:
        conn.execute(
            delete_sql,
            {
                "season": season,
                "game_type": game_type,
            },
        )

        if rows:
            conn.execute(
                insert_sql,
                rows,
            )

    return len(rows)