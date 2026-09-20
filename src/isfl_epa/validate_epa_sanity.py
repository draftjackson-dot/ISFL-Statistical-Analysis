from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from isfl_epa.storage.database import get_engine


FIRST_SEASON = 27
LAST_SEASON = 62

OUTPUT_DIR = Path(
    "ep_validation"
)


IDENTITY_TOLERANCE = 1e-10


def pct(
    series: pd.Series,
) -> float:
    if len(series) == 0:
        return float("nan")

    return float(
        series.mean() * 100
    )


def summarize_group(
    group: pd.DataFrame,
) -> pd.Series:
    epa = group[
        "epa"
    ].dropna()

    if epa.empty:
        return pd.Series(
            {
                "plays": len(group),
                "epa_plays": 0,
                "mean_epa": np.nan,
                "median_epa": np.nan,
                "std_epa": np.nan,
                "p05": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "positive_pct": np.nan,
                "negative_pct": np.nan,
            }
        )

    return pd.Series(
        {
            "plays": len(group),
            "epa_plays": len(epa),
            "mean_epa": epa.mean(),
            "median_epa": epa.median(),
            "std_epa": epa.std(),
            "p05": epa.quantile(0.05),
            "p25": epa.quantile(0.25),
            "p75": epa.quantile(0.75),
            "p95": epa.quantile(0.95),
            "positive_pct": pct(
                epa > 0
            ),
            "negative_pct": pct(
                epa < 0
            ),
        }
    )


def print_event_summary(
    name: str,
    group: pd.DataFrame,
):
    epa = group[
        "epa"
    ].dropna()

    print()
    print(
        f"===== {name} ====="
    )

    print(
        f"Rows:       "
        f"{len(group):,}"
    )

    print(
        f"EPA rows:   "
        f"{len(epa):,}"
    )

    if epa.empty:
        print(
            "No non-null EPA values."
        )
        return

    print(
        f"Mean EPA:   "
        f"{epa.mean():+.4f}"
    )

    print(
        f"Median EPA: "
        f"{epa.median():+.4f}"
    )

    print(
        f"Positive:   "
        f"{pct(epa > 0):.1f}%"
    )

    print(
        f"Negative:   "
        f"{pct(epa < 0):.1f}%"
    )

    print(
        f"Minimum:    "
        f"{epa.min():+.4f}"
    )

    print(
        f"Maximum:    "
        f"{epa.max():+.4f}"
    )


def main():
    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    engine = get_engine()

    sql = text(
        """
        SELECT
            p.id AS play_id,
            p.game_id,
            p.season,
            p.game_type,
            p.play_index,
            p.quarter,
            p.clock,
            p.play_type,
            p.description,
            p.down,
            p.distance,
            p.yardline_100,
            p.score_home,
            p.score_away,
            p.possession_team_id,
            p.counts_as_play,
            pe.ep_before,
            pe.ep_after,
            pe.epa
        FROM plays p
        JOIN play_epa pe
          ON pe.play_id = p.id
        WHERE p.season BETWEEN
            :first_season
            AND :last_season
          AND p.game_type = 'regular'
        ORDER BY
            p.season,
            p.game_id,
            p.play_index
        """
    )

    field_goal_sql = text(
        """
        SELECT
            p.id AS play_id,
            p.game_id,
            p.season,
            p.play_index,
            pe.play_id AS epa_play_id,
            pe.epa
        FROM plays p
        LEFT JOIN play_epa pe
          ON pe.play_id = p.id
        WHERE p.season BETWEEN
            :first_season
            AND :last_season
          AND p.game_type = 'regular'
          AND p.play_type = 'field_goal'
        ORDER BY
            p.season,
            p.game_id,
            p.play_index
        """
    )

    print(
        f"Loading EPA plays "
        f"S{FIRST_SEASON}-S{LAST_SEASON}..."
    )

    params = {
        "first_season": FIRST_SEASON,
        "last_season": LAST_SEASON,
    }

    df = pd.read_sql(
        sql,
        engine,
        params=params,
    )

    field_goals = pd.read_sql(
        field_goal_sql,
        engine,
        params=params,
    )

    print()
    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Games: "
        f"{df['game_id'].nunique():,}"
    )

    print(
        f"Seasons: "
        f"{df['season'].nunique():,}"
    )

    # ------------------------------------------------------
    # Basic null audit
    # ------------------------------------------------------

    print()
    print(
        "===== NULL COUNTS ====="
    )

    print(
        df[
            [
                "ep_before",
                "ep_after",
                "epa",
            ]
        ]
        .isna()
        .sum()
        .to_string()
    )

    # ------------------------------------------------------
    # EPA arithmetic identity
    #
    # For every play with all three values:
    #
    # EPA = EP_after - EP_before
    # ------------------------------------------------------

    identity = df.dropna(
        subset=[
            "ep_before",
            "ep_after",
            "epa",
        ]
    ).copy()

    identity[
        "identity_error"
    ] = (
        identity["epa"]
        - (
            identity["ep_after"]
            - identity["ep_before"]
        )
    )

    identity[
        "abs_identity_error"
    ] = (
        identity[
            "identity_error"
        ].abs()
    )

    bad_identity = identity[
        identity[
            "abs_identity_error"
        ]
        > IDENTITY_TOLERANCE
    ].copy()

    print()
    print(
        "===== EPA IDENTITY ====="
    )

    print(
        f"Rows checked: "
        f"{len(identity):,}"
    )

    print(
        "Maximum absolute error:",
        f"{identity['abs_identity_error'].max():.12f}",
    )

    print(
        "Mean absolute error:",
        f"{identity['abs_identity_error'].mean():.12f}",
    )

    print(
        "Rows exceeding tolerance:",
        f"{len(bad_identity):,}",
    )

    # ------------------------------------------------------
    # Play-type distribution
    # ------------------------------------------------------

    by_play_type = (
        df.groupby(
            "play_type",
            dropna=False,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index()
        .sort_values(
            "mean_epa",
            ascending=False,
            na_position="last",
        )
    )

    print()
    print(
        "===== EPA BY PLAY TYPE ====="
    )

    print(
        by_play_type.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # ------------------------------------------------------
    # Description-based event flags
    # ------------------------------------------------------

    description = (
        df["description"]
        .fillna("")
        .str.lower()
    )

    df[
        "is_touchdown"
    ] = description.str.contains(
        "touchdown",
        regex=False,
    )

    df[
        "is_interception"
    ] = description.str.contains(
        "intercept",
        regex=False,
    )

    df[
        "is_fumble"
    ] = description.str.contains(
        "fumble",
        regex=False,
    )

    df[
        "is_turnover_on_downs"
    ] = description.str.contains(
        "turnover on downs",
        regex=False,
    )

    df[
        "is_first_down"
    ] = description.str.contains(
        "first down",
        regex=False,
    )

    df[
        "is_safety"
    ] = description.str.contains(
        "safety",
        regex=False,
    )

    # "Turnover" is intentionally broad for this diagnostic.
    df[
        "is_turnover"
    ] = (
        df[
            "is_interception"
        ]
        | df[
            "is_fumble"
        ]
        | df[
            "is_turnover_on_downs"
        ]
    )

    # ------------------------------------------------------
    # Key football-event summaries
    # ------------------------------------------------------

    print_event_summary(
        "PASS PLAYS",
        df[
            df["play_type"]
            == "pass"
        ],
    )

    print_event_summary(
        "RUSH PLAYS",
        df[
            df["play_type"]
            == "rush"
        ],
    )

    print_event_summary(
        "SACKS",
        df[
            df["play_type"]
            == "sack"
        ],
    )

    print_event_summary(
        "TOUCHDOWNS",
        df[
            df[
                "is_touchdown"
            ]
        ],
    )

    print_event_summary(
        "INTERCEPTIONS",
        df[
            df[
                "is_interception"
            ]
        ],
    )

    print_event_summary(
        "FUMBLES",
        df[
            df[
                "is_fumble"
            ]
        ],
    )

    print_event_summary(
        "TURNOVERS ON DOWNS",
        df[
            df[
                "is_turnover_on_downs"
            ]
        ],
    )

    print_event_summary(
        "ALL TURNOVERS",
        df[
            df[
                "is_turnover"
            ]
        ],
    )

    print_event_summary(
        "FIRST DOWNS",
        df[
            df[
                "is_first_down"
            ]
        ],
    )

    print_event_summary(
        "SAFETIES",
        df[
            df[
                "is_safety"
            ]
        ],
    )

    # ------------------------------------------------------
    # Field goals
    #
    # By current design, field goals remain useful states
    # for EP training but are not assigned play EPA.
    #
    # This check deliberately queries plays independently of
    # the main EPA dataframe so field goals are visible even
    # when they correctly have no play_epa row.
    # ------------------------------------------------------

    field_goals_in_play_epa = (
        field_goals[
            "epa_play_id"
        ]
        .notna()
        .sum()
    )

    field_goals_with_epa = (
        field_goals[
            "epa"
        ]
        .notna()
        .sum()
    )

    print()
    print(
        "===== FIELD GOAL EPA CHECK ====="
    )

    print(
        f"Field-goal plays: "
        f"{len(field_goals):,}"
    )

    print(
        "Rows present in play_epa:",
        f"{field_goals_in_play_epa:,}",
    )

    print(
        "Rows with non-null EPA:",
        f"{field_goals_with_epa:,}",
    )

    if (
        field_goals_in_play_epa == 0
        and field_goals_with_epa == 0
    ):
        print(
            "PASS: field goals have no "
            "play_epa rows or assigned EPA."
        )
    else:
        print(
            "WARNING: some field goals "
            "are present in play_epa or "
            "have assigned EPA."
        )

    # ------------------------------------------------------
    # Season-level EPA stability
    # ------------------------------------------------------

    season_rows = []

    scrimmage = df[
        df["play_type"].isin(
            [
                "pass",
                "rush",
                "sack",
            ]
        )
        & df[
            "epa"
        ].notna()
    ].copy()

    for season, group in scrimmage.groupby(
        "season"
    ):
        epa = group[
            "epa"
        ]

        season_rows.append(
            {
                "season": int(
                    season
                ),
                "plays": len(
                    group
                ),
                "mean_epa": epa.mean(),
                "median_epa": epa.median(),
                "std_epa": epa.std(),
                "positive_pct": pct(
                    epa > 0
                ),
                "p05": epa.quantile(
                    0.05
                ),
                "p95": epa.quantile(
                    0.95
                ),
            }
        )

    by_season = pd.DataFrame(
        season_rows
    )

    print()
    print(
        "===== SCRIMMAGE EPA BY SEASON ====="
    )

    print(
        by_season.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # ------------------------------------------------------
    # Overall distribution
    # ------------------------------------------------------

    all_epa = df[
        "epa"
    ].dropna()

    print()
    print(
        "===== OVERALL EPA DISTRIBUTION ====="
    )

    print(
        f"EPA plays: "
        f"{len(all_epa):,}"
    )

    print(
        f"Mean:   "
        f"{all_epa.mean():+.4f}"
    )

    print(
        f"Median: "
        f"{all_epa.median():+.4f}"
    )

    print(
        f"Std dev:"
        f" {all_epa.std():.4f}"
    )

    quantiles = [
        0.001,
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
        0.999,
    ]

    for q in quantiles:
        print(
            f"P{q * 100:05.1f}: "
            f"{all_epa.quantile(q):+.4f}"
        )

    print(
        f"Minimum: "
        f"{all_epa.min():+.4f}"
    )

    print(
        f"Maximum: "
        f"{all_epa.max():+.4f}"
    )

    # ------------------------------------------------------
    # Extreme values for manual inspection
    # ------------------------------------------------------

    extreme_columns = [
        "season",
        "game_id",
        "play_id",
        "play_index",
        "quarter",
        "clock",
        "play_type",
        "down",
        "distance",
        "yardline_100",
        "description",
        "ep_before",
        "ep_after",
        "epa",
    ]

    lowest = (
        df[
            df["epa"].notna()
        ]
        .nsmallest(
            50,
            "epa",
        )[
            extreme_columns
        ]
    )

    highest = (
        df[
            df["epa"].notna()
        ]
        .nlargest(
            50,
            "epa",
        )[
            extreme_columns
        ]
    )

    print()
    print(
        "===== 20 MOST NEGATIVE EPA PLAYS ====="
    )

    print(
        lowest.head(
            20
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "===== 20 MOST POSITIVE EPA PLAYS ====="
    )

    print(
        highest.head(
            20
        ).to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Save diagnostics
    # ------------------------------------------------------

    by_play_type.to_csv(
        OUTPUT_DIR
        / "epa-by-play-type.csv",
        index=False,
    )

    by_season.to_csv(
        OUTPUT_DIR
        / "epa-by-season.csv",
        index=False,
    )

    lowest.to_csv(
        OUTPUT_DIR
        / "epa-most-negative-plays.csv",
        index=False,
    )

    highest.to_csv(
        OUTPUT_DIR
        / "epa-most-positive-plays.csv",
        index=False,
    )

    bad_identity.to_csv(
        OUTPUT_DIR
        / "epa-identity-warnings.csv",
        index=False,
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