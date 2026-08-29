"""
Build model-ready player-prop feature rows for a team's roster, for a game
that hasn't been played yet.

Third sibling to live_features.py and live_quarter_half_features.py, and the
hardest of the three - because it is the only one that has to answer a
question the historical pipeline never had to ask: WHO IS PLAYING?

  Team and quarter/half features describe two teams that are certainly
  going to be on the floor. A player prop describes one person who might
  not be. The training data never faced this: every historical row already
  knew who appeared, because the box score said so.

THE ROSTER DEFINITION, chosen deliberately and scoped down on purpose:

    every player whose most recent appearance for this team falls in the
    season the fixture belongs to AND has a computable ROLL10_MIN, minus
    anyone the live injury report marks Out or Doubtful

The season clause is load-bearing, and it was added after the first version
shipped without it. Without it "roster" means everyone who ever wore the
jersey across eleven seasons - 111 players for Atlanta, retirees included -
and that stayed invisible because the endpoint's top-10 display cut happened
to select real current players anyway. See current_roster().

That is NOT a rotation prediction, and it is not trying to be. Guessing
tonight's 8-9 man rotation is a separate modelling problem with its own
data requirements. This is a filter over infrastructure that already exists
and is already verified: the forward-filled ROLL10_MIN from
build_player_rolling_minutes.py answers "is this a real current
contributor", and injury_availability.py already reconciles report names to
PLAYER_IDs. Both were built and checked for other reasons; this composes
them rather than adding a third source of truth.

  Consequence worth stating: a player with no ROLL10_MIN anywhere - a
  genuine debut, a two-way call-up in his first week - is not on the
  returned roster at all. He is not excluded because he is unavailable; he
  is excluded because nothing is known about him. The two are different and
  the caller is told which is which.

NO REPORT IS NOT AN EMPTY REPORT. If injury_report is None - offseason, a
failed fetch, an unpublished hour - the full roster comes back with
availability_known=False on every row, and the summary says so. Silently
returning the same roster as a clean report would assert that nobody is
injured, which is the unknown-as-zero substitution this project has now
rejected five times (empty MIN strings, V3's zero-filled absent players,
NOT YET SUBMITTED teams, tied periods, and here).

ROUTING IS READ FROM THE MANIFEST, NOT RE-DERIVED. models_player_props/
manifest.json carries the rule - linear when all 12 rolling features are
present, xgb otherwise - and this module applies that list rather than
hardcoding a second copy of it. Different players on the SAME roster route
differently in the same call, which is the whole point of the hybrid: a
ten-year veteran and a rookie called up last week both get a number.

Every history frame is a parameter, not a module-level load. The player
history alone is 77 MB; a serving process holds it open once.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from live_features import current_elo, rest_features, season_of, team_history

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_player_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    PLAYER_FEATURE_COLUMNS,
    TEAM_CONTEXT_COLUMNS,
)
from build_player_rolling_minutes import ROLLING_SOURCES, WINDOWS  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent / "models_player_props"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

ROSTER_COLUMN = "ROLL10_MIN"

# Only what the roster and the rolling maths need; the full file is 46
# columns wide and most of them are box-score detail nothing here reads.
HISTORY_COLUMNS = (
    ["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME",
     "MIN_NUMERIC", ROSTER_COLUMN]
    + list(dict.fromkeys(ROLLING_SOURCES))
)

OUTPUT_ID_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "ROUTE",
                     "AVAILABILITY_KNOWN", "APPEARANCES_THIS_SEASON"]


def load_player_history() -> pd.DataFrame:
    """The player-game history, narrowed to what this module reads."""
    path = (Path(__file__).resolve().parents[1] / "data-pipeline" / "data"
            / "processed" / "player_boxscores_with_rolling.csv")
    history = pd.read_csv(path, usecols=HISTORY_COLUMNS, low_memory=False)
    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"])
    return history.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(
        drop=True
    )


def load_routing_rule() -> list:
    """The columns the manifest says decide linear-vs-xgb.

    Read, never re-derived. A second copy of this list here could drift
    from the shipped one and route rows to a model that cannot take them.
    """
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found - run finalize_player_models.py first. "
            f"Routing cannot be guessed."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["routing"]["decide_on"]


def route_for(features: dict, decide_on: list) -> str:
    """THE ROUTING RULE, applied to one player's feature dict."""
    complete = all(not pd.isna(features[column]) for column in decide_on)
    return "linear" if complete else "xgb"


def current_roster(player_history_df: pd.DataFrame, team_id: int,
                   game_date: pd.Timestamp, season: int) -> pd.DataFrame:
    """Players currently on this team with real playing-time history.

    TWO conditions, and the season one is not optional. Without it this
    returns everyone who ever wore the jersey across eleven seasons - 111
    players for Atlanta, including retirees and players traded away years
    ago. That is not a roster, and it was only invisible because a top-10
    display cut happened to hide it. "Who is on the team" is a
    data-correctness question and belongs here; "how many to show" is a
    presentation question and belongs to the caller.

      1. The player's most recent APPEARANCE for this team falls inside the
         season the fixture belongs to. Appearance, not row: absence rows
         exist too, and a player can be listed without ever dressing.
      2. That appearance has a computable ROLL10_MIN - he is an established
         contributor rather than someone with two games to his name.

    "As of game_date", so replaying a historical fixture sees the roster as
    it stood then rather than as it stands now.

    KNOWN GAP, accepted rather than solved: a player genuinely on the
    roster who has not yet appeared FOR THIS TEAM this season is excluded -
    a trade completed yesterday, or the opening night or two of a new
    season before anyone has logged a minute. Narrow, real, and the same
    class of accepted limitation as the three un-fetchable quarter-score
    games. Fixing it needs a live roster feed, which is the same blocked
    dependency as injury availability.
    """
    appearances = player_history_df[
        (player_history_df["TEAM_ID"] == team_id)
        & (player_history_df["GAME_DATE"] < pd.Timestamp(game_date))
        & player_history_df["MIN_NUMERIC"].notna()
    ]
    if appearances.empty:
        return appearances

    # drop_duplicates, NOT groupby().last(). GroupBy.last() takes each
    # column's last NON-NULL value independently, so it returns a row that
    # never existed: for Taurean Prince it paired SEASON from his 2025-26
    # last appearance with a ROLL10_MIN of 29.38 carried over from 2024-25,
    # when his real 2025-26 value is NaN. That admitted players to the
    # roster on the strength of a number from a different season.
    latest = appearances.sort_values(["GAME_DATE", "GAME_ID"]).drop_duplicates(
        subset="PLAYER_ID", keep="last")
    latest = latest[latest["SEASON"] == season]
    return latest[latest[ROSTER_COLUMN].notna()].reset_index(drop=True)


def absent_player_ids(injury_report, team_id: int) -> set:
    """PLAYER_IDs the report marks Out or Doubtful for this team.

    Delegates to injury_availability.reconcile, which owns the name join,
    the accent folding and the traded-player fallback. Importing it here
    rather than reimplementing any of that is the whole point.
    """
    from injury_availability import reconcile

    if injury_report is None or injury_report.players.empty:
        return set()

    reconciled = reconcile(injury_report.players)
    absent = reconciled[
        (reconciled["TEAM_ID"] == team_id)
        & reconciled["IS_ABSENT"]
        & reconciled["PLAYER_ID"].notna()
    ]
    return {int(pid) for pid in absent["PLAYER_ID"]}


def player_rolling(player_history_df: pd.DataFrame, player_id: int,
                   season: int, game_date: pd.Timestamp) -> tuple:
    """Trailing means over the player's last N APPEARANCES this season.

    Appearances, not calendar games: build_player_rolling_minutes.py rolls
    over played rows only so a window is never diluted by games he sat out.
    Reproduced here for one player rather than recomputed differently -
    fewer than `window` appearances gives NaN, exactly as the pipeline's
    shift(1).rolling(window) does at its default min_periods.
    """
    played = player_history_df[
        (player_history_df["PLAYER_ID"] == player_id)
        & (player_history_df["SEASON"] == season)
        & (player_history_df["GAME_DATE"] < pd.Timestamp(game_date))
        & player_history_df["MIN_NUMERIC"].notna()
    ].sort_values(["GAME_DATE", "GAME_ID"])

    features = {}
    for window in WINDOWS:
        recent = played.tail(window)
        complete = len(recent) == window
        for source, suffix in ROLLING_SOURCES.items():
            features[f"ROLL{window}_{suffix}"] = (
                float(recent[source].mean()) if complete else np.nan
            )
    return features, len(played)


def get_live_player_features(
    team_id: int,
    opponent_team_id: int,
    game_date,
    is_home: bool,
    player_history_df: pd.DataFrame,
    games_final_df: pd.DataFrame,
    injury_report=None,
    decide_on: list = None,
) -> pd.DataFrame:
    """One feature row per rostered, available player, tagged with its model.

    Returns a frame carrying the 17 model inputs plus PLAYER_ID,
    PLAYER_NAME, TEAM_ID, ROUTE ("linear" or "xgb"), AVAILABILITY_KNOWN and
    APPEARANCES_THIS_SEASON. Empty frame if nobody qualifies - that is a
    real answer for a team with no history, not an error.

    injury_report is the InjuryReport from fetch_current_injury_report, or
    None. None does NOT mean "nobody is out" - see the module docstring.
    """
    game_date = pd.Timestamp(game_date)
    season = season_of(game_date)
    decide_on = decide_on if decide_on is not None else load_routing_rule()

    roster = current_roster(player_history_df, team_id, game_date, season)
    availability_known = injury_report is not None
    excluded = absent_player_ids(injury_report, team_id) if availability_known else set()

    # Team context is identical for every player on the roster, so it is
    # computed once rather than per player.
    own = team_history(games_final_df, team_id, game_date)
    opponent = team_history(games_final_df, opponent_team_id, game_date)
    rest = rest_features(own, game_date)
    context = {
        "IS_HOME": int(bool(is_home)),
        "REST_DAYS": rest["REST_DAYS"],
        "IS_BACK_TO_BACK": rest["IS_BACK_TO_BACK"],
        "TEAM_ELO": current_elo(own, season),
        "OPPONENT_ELO": current_elo(opponent, season),
    }

    rows = []
    for entry in roster.itertuples():
        player_id = int(entry.PLAYER_ID)
        if player_id in excluded:
            continue

        features, appearances = player_rolling(
            player_history_df, player_id, season, game_date
        )
        features.update(context)

        rows.append({
            "PLAYER_ID": player_id,
            "PLAYER_NAME": entry.PLAYER_NAME,
            "TEAM_ID": team_id,
            "ROUTE": route_for(features, decide_on),
            "AVAILABILITY_KNOWN": availability_known,
            "APPEARANCES_THIS_SEASON": appearances,
            **features,
        })

    if not rows:
        return pd.DataFrame(columns=OUTPUT_ID_COLUMNS + FEATURE_COLUMNS)

    frame = pd.DataFrame(rows)[OUTPUT_ID_COLUMNS + FEATURE_COLUMNS]
    return frame.sort_values(ROSTER_COLUMN if ROSTER_COLUMN in frame
                             else "ROLL10_MIN", ascending=False).reset_index(drop=True)


def describe(frame: pd.DataFrame) -> str:
    """One line a caller can surface, including the honesty caveat."""
    if frame.empty:
        return "no players with usable history"

    linear = int((frame["ROUTE"] == "linear").sum())
    xgb = int((frame["ROUTE"] == "xgb").sum())
    known = bool(frame["AVAILABILITY_KNOWN"].iloc[0])
    caveat = ("" if known else
              "  AVAILABILITY UNKNOWN - no injury report was available, so "
              "nobody has been excluded. This is not a clean bill of health.")
    return (f"{len(frame)} players ({linear} via linear, {xgb} via xgb)." + caveat)


if __name__ == "__main__":
    print("Live player-prop features")
    rule = load_routing_rule()
    print(f"  routing reads {len(rule)} columns: {rule[:3]} ...")
    print(f"  feature row is {len(FEATURE_COLUMNS)} columns "
          f"({len(PLAYER_FEATURE_COLUMNS)} rolling + "
          f"{len(TEAM_CONTEXT_COLUMNS)} context)")
