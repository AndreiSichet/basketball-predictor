"""
Build a model-ready Q1/first-half feature row for a game that hasn't been
played yet.

A SIBLING of live_features.py, not an extension of it. These serve a
different model family (linear pipelines, not XGBoost) from a different
historical source (BoxScoreSummaryV3 line scores, not LeagueGameFinder),
and they fail in a different way when history is short. Keeping them apart
mirrors build_quarter_half_rolling.py being kept apart from
build_rolling_features.py rather than folded into it.

NOTHING IS RECOMPUTED THAT ALREADY EXISTS. The opponent pairing and the six
derived metrics come from build_quarter_half_rolling.py itself; the team
context (Elo, rest days, back-to-back) comes from live_features.py, which is
already verified against 200 replayed games. This module contributes exactly
one new thing: the trailing-window arithmetic for six quarter/half metrics,
in the shape one unplayed matchup needs.

THE HISTORY FRAME MUST BE REINDEXED, and this is the subtle part. The
pipeline rolls over the FULL 26,398 team-game universe, in which the three
un-fetchable 2025-11-19 games sit as NaN rows so their absence propagates.
quarter_half_raw.csv has no rows for them at all. Rolling the raw file
directly would quietly close that gap - a team's "last five games" would
reach one game further back than the pipeline's did, and the live features
would disagree with the training features for a handful of fixtures with
nothing to signal it. load_quarter_half_history() reindexes first, exactly
as the pipeline does, so the windows are the same windows.

AN INCOMPLETE WINDOW RAISES, IT DOES NOT RETURN NaN. This is the sharpest
difference from live_features.py, and it follows from the model class.
XGBoost learns a default direction for missing values, so the team-level
service can hand it a NaN and get a degraded but real answer. A
LinearRegression pipeline cannot: it raises deep inside sklearn, on input
validation, with a message that says nothing about basketball. So the check
happens here, naming the team and the shortfall - the same reasoning that
made NoReportAvailable an exception rather than an empty return.

Results are only as current as the frames passed in. A serving process
loads both once and reuses them; that is why they are parameters rather
than module-level loads.
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The 30 columns and their order, from the script that defined them. The
# manifest is a serialized copy of this list, and verify_manifest_agrees()
# checks the two have not drifted.
from train_quarter_half_baseline import (
    CONTEXT_FEATURES,
    FEATURE_COLUMNS,
    QUARTER_HALF_FEATURES,
)

# Team context, already verified against 200 replayed games. Reused whole.
from live_features import current_elo, rest_features, season_of, team_history

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_rolling_features import WINDOWS, derive_season  # noqa: E402
from build_quarter_half_rolling import (  # noqa: E402
    ROLLING_METRICS,
    attach_opponent,
    derive_metrics,
    load_raw,
    load_universe,
    reindex_to_universe,
)

MODELS_DIR = Path(__file__).resolve().parent / "models_quarter_half"
MANIFEST_PATH = MODELS_DIR / "manifest.json"


class InsufficientQuarterHalfHistory(RuntimeError):
    """A team has too few games this season for a complete trailing window.

    An exception rather than a NaN-filled row, because the models this feeds
    cannot consume NaN at all. Carries the team and the shortfall so the
    caller can say which side of the matchup is unscoreable and why.
    """

    def __init__(self, team_id: int, side: str, missing: list, played: int):
        self.team_id = team_id
        self.side = side
        self.missing = missing
        self.games_played = played
        super().__init__(
            f"{side} team {team_id} has played {played} game(s) with quarter "
            f"data this season - not enough for a complete window. "
            f"{len(missing)} feature(s) would be NaN, e.g. {missing[:3]}. "
            f"The Q1/1H models are linear and cannot score a partial row; "
            f"refuse the fixture rather than imputing."
        )


def load_quarter_half_history(quiet: bool = True) -> pd.DataFrame:
    """One row per team-game: the six derived metrics, with date and season.

    Everything here is imported. See the module docstring on why the
    reindex matters. Load once per process, then pass the result in.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer) if quiet else contextlib.nullcontext():
        raw = load_raw()
        paired = attach_opponent(raw)
        metrics = derive_metrics(paired)
        universe = load_universe()
        history = reindex_to_universe(universe, metrics)

    history["SEASON"] = derive_season(history["GAME_DATE"])
    return history.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(
        drop=True
    )


def team_quarter_half_rolling(history: pd.DataFrame, team_id: int,
                              game_date: pd.Timestamp, season: int) -> tuple:
    """Trailing Q1/1H means for one team. Returns (features, games_played).

    Season-scoped and strictly-before, matching the pipeline's groupby plus
    shift(1). A window shorter than `window`, or containing any NaN, yields
    NaN - which is what rolling(window).mean() does at its default
    min_periods, so the two agree by construction rather than by luck.
    """
    rows = history[
        (history["TEAM_ID"] == team_id)
        & (history["GAME_DATE"] < pd.Timestamp(game_date))
        & (history["SEASON"] == season)
    ].sort_values(["GAME_DATE", "GAME_ID"])

    features = {}
    for window in WINDOWS:
        recent = rows.tail(window)
        for metric in ROLLING_METRICS:
            complete = len(recent) == window and bool(recent[metric].notna().all())
            features[f"ROLL{window}_{metric}"] = (
                float(recent[metric].mean()) if complete else np.nan
            )
    return features, len(rows)


def get_live_quarter_half_features(
    home_team_id: int,
    away_team_id: int,
    game_date,
    quarter_half_history_df: pd.DataFrame,
    games_final_df: pd.DataFrame,
) -> pd.DataFrame:
    """The 30-column row the six Q1/1H models expect, for one unplayed game.

    Takes BOTH history frames rather than loading either. The quarter/half
    metrics and the team context genuinely come from different sources -
    line scores and LeagueGameFinder - so one frame cannot supply both, and
    a serving process should hold each open exactly once.

    Raises InsufficientQuarterHalfHistory if either team's trailing window
    is incomplete. See the module docstring.
    """
    game_date = pd.Timestamp(game_date)
    season = season_of(game_date)

    row = {}
    for side, team_id in (("HOME", home_team_id), ("AWAY", away_team_id)):
        rolling, played = team_quarter_half_rolling(
            quarter_half_history_df, team_id, game_date, season
        )

        missing = [f"{side}_{name}" for name, value in rolling.items()
                   if pd.isna(value)]
        if missing:
            raise InsufficientQuarterHalfHistory(team_id, side, missing, played)

        for name, value in rolling.items():
            row[f"{side}_{name}"] = value

        # Reused from live_features, not recomputed. Only the three context
        # values these models actually consume are taken.
        history = team_history(games_final_df, team_id, game_date)
        rest = rest_features(history, game_date)
        row[f"{side}_TEAM_ELO"] = current_elo(history, season)
        row[f"{side}_REST_DAYS"] = rest["REST_DAYS"]
        row[f"{side}_IS_BACK_TO_BACK"] = rest["IS_BACK_TO_BACK"]

    frame = pd.DataFrame([row])[FEATURE_COLUMNS]

    # The context columns have their own missingness (a team's first game
    # ever has no REST_DAYS), and it is just as fatal to a linear model.
    still_missing = [c for c in FEATURE_COLUMNS if frame[c].isna().any()]
    if still_missing:
        raise InsufficientQuarterHalfHistory(
            home_team_id, "context for", still_missing, -1
        )

    return frame


def verify_manifest_agrees() -> bool:
    """The shipped manifest's column list must match the code's.

    A serving layer reads the manifest; this module reads the module. If
    they ever disagree, a row would be assembled in one order and consumed
    in another - silently, since both are 30 floats.
    """
    import json

    if not MANIFEST_PATH.exists():
        print(f"  manifest not found at {MANIFEST_PATH} - skipped")
        return True

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    shipped = manifest["feature_columns"]
    if shipped != FEATURE_COLUMNS:
        raise RuntimeError(
            "manifest.json's feature_columns disagrees with "
            "train_quarter_half_baseline.FEATURE_COLUMNS.\n"
            f"  manifest: {len(shipped)} columns\n"
            f"  code    : {len(FEATURE_COLUMNS)} columns\n"
            f"  first difference at index "
            f"{next(i for i, (a, b) in enumerate(zip(shipped, FEATURE_COLUMNS)) if a != b)}"
        )
    print(f"  manifest and code agree on all {len(FEATURE_COLUMNS)} columns, "
          f"in order.")
    return True


def load_games_final() -> pd.DataFrame:
    """Convenience loader, matching live_features.load_games_final."""
    from live_features import load_games_final as _load

    return _load()


if __name__ == "__main__":
    print("Quarter/half live features")
    print(f"  {len(QUARTER_HALF_FEATURES)} rolling + {len(CONTEXT_FEATURES)} "
          f"context = {len(FEATURE_COLUMNS)} columns")
    verify_manifest_agrees()
    history = load_quarter_half_history()
    print(f"  history loaded: {len(history):,} team-game rows, "
          f"{history['GAME_DATE'].min().date()} to "
          f"{history['GAME_DATE'].max().date()}")
