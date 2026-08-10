"""
Build a model-ready feature row for a game that has not been played yet.

The training pipeline computes features for every historical game at once,
vectorized across the whole table. Inference needs the opposite shape: one
matchup, on demand, fast. This module reimplements the same feature
definitions for that shape — which makes it the single most dangerous file
in the project, because any divergence from the pipeline produces features
that look perfectly reasonable and are quietly wrong, and no error is ever
raised. A model fed subtly different features than it trained on just
returns worse predictions.

Two defences against that:

  1. Every constant and formula is IMPORTED from the pipeline scripts that
     built the training data - the metric list, the season boundary, the
     rest-day cap, the Elo K-factor and season regression. Nothing about
     the feature definitions is restated here, so the pipeline and
     inference cannot drift apart without an import breaking.

  2. verify_against_training_data() below rebuilds features for real
     historical games and compares them, value by value, against the rows
     the pipeline produced for those same games. Run it after any change to
     either side.

OPERATIONAL CAVEAT: this function is only ever as current as the
games_final.csv handed to it. It cannot know about a game played last night
if the pipeline has not been re-run since — it will silently compute
features from stale history and return them with no indication that
anything is missing. Keeping that file fresh is a scheduling problem for
the post-v1 retraining pipeline, not something this module can solve. A
caller that cares should check how recent the newest GAME_DATE is.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURE_COLUMNS, PER_SIDE_FEATURES

# The pipeline scripts live outside any importable package (the folder name
# "data-pipeline" is not a valid module name), so their directory goes on
# the path directly. Importing the real definitions is worth this ugliness:
# the alternative is copying K_FACTOR, the 1/3 season regression, the 7-day
# rest cap and the metric list into this file, where they would rot.
PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_elo_ratings import (  # noqa: E402
    BASELINE_RATING,
    K_FACTOR,
    SEASON_REGRESSION_FRACTION,
    expected_score,
)
from build_rest_days import REST_DAYS_CAP  # noqa: E402
from build_rolling_features import METRICS, WINDOWS, derive_season  # noqa: E402


def season_of(game_date: pd.Timestamp) -> int:
    """Season label for a single date, via the pipeline's own rule.

    derive_season works on a Series; wrapping one date rather than
    reimplementing the comparison keeps the August boundary defined in
    exactly one place.
    """
    return int(derive_season(pd.Series([pd.Timestamp(game_date)])).iloc[0])


def team_history(games_final_df: pd.DataFrame, team_id: int, before: pd.Timestamp) -> pd.DataFrame:
    """That team's completed games strictly before the given date, oldest first.

    Strictly before matters: including the game being predicted would leak
    its result into its own features.
    """
    rows = games_final_df[
        (games_final_df["TEAM_ID"] == team_id) & (games_final_df["GAME_DATE"] < before)
    ]
    return rows.sort_values(["GAME_DATE", "GAME_ID"])


def rolling_features(history: pd.DataFrame, season: int) -> dict:
    """Trailing means over the team's most recent games THIS SEASON.

    Season-scoped to match build_rolling_features.py, which groups by
    (team, season) so windows reset at the season boundary instead of
    blending across an offseason of roster turnover.

    Fewer games than the window means NaN, exactly as in training: the
    pipeline's shift(1).rolling(n) leaves early-season rows empty, and the
    XGBoost models learned a default split direction for them.
    """
    in_season = history[history["SEASON"] == season]

    # WIN is derived rather than stored: build_rolling_features.py creates it
    # from WL, averages it into ROLL*_WIN_PCT, then drops the column again.
    source = in_season.assign(WIN=(in_season["WL"] == "W").astype(int))

    features = {}
    for window in WINDOWS:
        recent = source.tail(window)
        complete = len(recent) == window
        for source_col, feature_name in METRICS.items():
            features[f"ROLL{window}_{feature_name}"] = (
                recent[source_col].mean() if complete else np.nan
            )
    return features


def rest_features(history: pd.DataFrame, game_date: pd.Timestamp) -> dict:
    """Days since the team's last game, capped, plus the back-to-back flag.

    Not season-scoped, matching build_rest_days.py, which measures real
    calendar gaps across the team's whole history. The cap is what stops a
    150-day offseason from registering as meaningful rest.
    """
    if history.empty:
        # No prior game at all: the pipeline's diff() yields NaN here, and
        # NaN == 1 is False, so the flag is 0 rather than unknown.
        return {"REST_DAYS": np.nan, "IS_BACK_TO_BACK": 0}

    last_date = history["GAME_DATE"].iloc[-1]
    rest_days = min((pd.Timestamp(game_date) - last_date).days, REST_DAYS_CAP)
    return {"REST_DAYS": float(rest_days), "IS_BACK_TO_BACK": int(rest_days == 1)}


def current_elo(history: pd.DataFrame, season: int) -> float:
    """The team's rating going into this game.

    Rather than replaying every game in league history, this reads the last
    completed game's stored pre-game ratings and applies one update. The
    stored TEAM_ELO is what the sequential replay held at that moment, so
    advancing it by that game's result lands exactly where the replay would
    have — and nothing else can have moved the rating since, because the
    team has not played since.
    """
    if history.empty:
        return BASELINE_RATING

    last = history.iloc[-1]
    rating, opponent_rating = last["TEAM_ELO"], last["OPPONENT_ELO"]

    actual = 1.0 if last["WL"] == "W" else 0.0
    rating = rating + K_FACTOR * (actual - expected_score(rating, opponent_rating))

    # New season: regress toward the mean once, for roster turnover. Applied
    # on season change rather than per season elapsed, matching the
    # pipeline's `last_season[team] != season` check.
    if last["SEASON"] != season:
        rating = BASELINE_RATING + (rating - BASELINE_RATING) * (1 - SEASON_REGRESSION_FRACTION)

    return rating


def team_features(games_final_df: pd.DataFrame, team_id: int, game_date: pd.Timestamp) -> dict:
    """All 17 pre-game features for one team, keyed by unprefixed name."""
    season = season_of(game_date)
    history = team_history(games_final_df, team_id, game_date)

    features = rolling_features(history, season)
    features.update(rest_features(history, game_date))
    features["TEAM_ELO"] = current_elo(history, season)
    return features


def get_live_features(
    home_team_id: int,
    away_team_id: int,
    game_date,
    games_final_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the 34-column feature row for one upcoming matchup.

    games_final_df is passed in rather than loaded here so a serving process
    can read it once and reuse it across every request, instead of parsing
    several megabytes of CSV per prediction.

    Returns a single-row DataFrame with columns in FEATURE_COLUMNS order,
    ready for any of the seven models' .predict(). The order is not
    cosmetic: the saved models carry these names and validate against them.
    """
    game_date = pd.Timestamp(game_date)

    row = {}
    for prefix, team_id in (("HOME", home_team_id), ("AWAY", away_team_id)):
        for name, value in team_features(games_final_df, team_id, game_date).items():
            row[f"{prefix}_{name}"] = value

    missing = set(FEATURE_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Feature assembly incomplete, missing: {sorted(missing)}")

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def load_games_final() -> pd.DataFrame:
    """Convenience loader for scripts and the verification below.

    A serving process should call this once at startup, not per request.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "data-pipeline"
        / "data"
        / "processed"
        / "games_final.csv"
    )
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def verify_against_training_data(sample_size: int = 200, seed: int = 42) -> bool:
    """Rebuild features for real past games and diff them against the pipeline.

    The only honest test available: there is no ground truth for a future
    game, but every historical game has a row in model_dataset.csv that the
    pipeline already computed. Feeding this function nothing but a matchup
    and a date should reproduce that row exactly.

    Deliberately samples across the whole history rather than picking one
    convenient game, so early-season NaN rows, season-opening Elo
    regression and back-to-backs are all covered.
    """
    from train_baseline import load_dataset  # local import: avoids a cycle

    games_final_df = load_games_final()
    dataset = load_dataset()

    sample = dataset.sample(n=min(sample_size, len(dataset)), random_state=seed)

    mismatches = []
    max_difference = 0.0

    for _, expected_row in sample.iterrows():
        actual = get_live_features(
            expected_row["HOME_TEAM_ID"],
            expected_row["AWAY_TEAM_ID"],
            expected_row["GAME_DATE"],
            games_final_df,
        )

        for column in FEATURE_COLUMNS:
            got, want = actual[column].iloc[0], expected_row[column]

            if pd.isna(got) and pd.isna(want):
                continue
            if pd.isna(got) != pd.isna(want):
                mismatches.append((expected_row["GAME_ID"], column, got, want))
                continue

            difference = abs(float(got) - float(want))
            max_difference = max(max_difference, difference)
            if not np.isclose(got, want, rtol=1e-9, atol=1e-9):
                mismatches.append((expected_row["GAME_ID"], column, got, want))

    print(f"Checked {len(sample)} games x {len(FEATURE_COLUMNS)} features "
          f"= {len(sample) * len(FEATURE_COLUMNS)} values")
    print(f"Largest absolute difference: {max_difference:.2e}")

    if mismatches:
        print(f"\nFAIL - {len(mismatches)} mismatches. First 10:")
        for game_id, column, got, want in mismatches[:10]:
            print(f"  game {game_id} {column}: got {got}, pipeline had {want}")
        return False

    print("\nPASS - every value reproduces the pipeline exactly.")
    return True


def main():
    print("=" * 78)
    print("LIVE FEATURE VERIFICATION")
    print("=" * 78)
    print("Rebuilding features for historical games from team ids + date alone,")
    print("then diffing against the rows the pipeline computed for those games.\n")

    ok = verify_against_training_data()

    print("\n" + "=" * 78)
    print("EXAMPLE OUTPUT")
    print("=" * 78)
    games_final_df = load_games_final()
    latest = pd.Timestamp(games_final_df["GAME_DATE"].max())
    print(f"games_final.csv newest game: {latest.date()} "
          f"- features are only as current as this.\n")

    # A plausible next-day matchup between the two teams that played most
    # recently, standing in for a real upcoming fixture.
    last_game = games_final_df[games_final_df["GAME_DATE"] == latest]
    home_id = last_game.loc[last_game["IS_HOME"], "TEAM_ID"].iloc[0]
    away_id = last_game.loc[~last_game["IS_HOME"], "TEAM_ID"].iloc[0]
    # DateOffset rather than Timedelta: pandas 2.3 / numpy 2.5 raise a
    # generic-unit DeprecationWarning on Timestamp + Timedelta.
    tip_off = latest + pd.DateOffset(days=1)

    print(f"Hypothetical: team {home_id} hosting team {away_id} on {tip_off.date()}")
    example = get_live_features(home_id, away_id, tip_off, games_final_df)
    print(example.T.to_string(header=["value"]))

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
