import isfl_epa.wp.training as training
from sqlalchemy import text

from isfl_epa.storage.database import get_engine
from isfl_epa.wp.training import (
    WP_FEATURE_COLS,
    load_wp_training_data,
)

print("TRAINING FILE:")
print(training.__file__)
print()

engine = get_engine()

with engine.connect() as conn:
    count = conn.execute(
        text(
            """
            SELECT COUNT(*)
            FROM plays p
            JOIN team_games h
              ON h.game_id=p.game_id
             AND h.is_home=TRUE
            JOIN team_games a
              ON a.game_id=p.game_id
             AND a.is_home=FALSE
            WHERE p.season=62
              AND p.game_type='regular'
              AND h.game_type='regular'
              AND a.game_type='regular'
              AND p.counts_as_play=TRUE
            """
        )
    ).scalar()

print("PYTHON DATABASE COUNT:", count)
print()

df = load_wp_training_data(engine, [62])

print("Rows:", len(df))

from isfl_epa.storage.database import get_engine
from isfl_epa.wp.training import (
    WP_FEATURE_COLS,
    load_wp_training_data,
)

engine = get_engine()

df = load_wp_training_data(
    engine,
    [62],
)

print("Rows:", len(df))
print()
print("Win target:")
print(df["possession_won"].value_counts(dropna=False))
print()
print("Features:")
print(df[WP_FEATURE_COLS].describe())
print()
print("Sample:")
print(
    df[
        [
            "game_id",
            "play_index",
            "quarter",
            "clock",
            "possession_team",
            "is_home",
            "score_differential",
            "game_seconds",
            "down",
            "distance",
            "yardline_100",
            "possession_won",
        ]
    ].head(30).to_string(index=False)
)
print("Rows:", len(df))
print()

print("Possession teams:")
print(df["possession_team"].value_counts(dropna=False))
print()

print("Win target:")
print(df["possession_won"].value_counts(dropna=False))