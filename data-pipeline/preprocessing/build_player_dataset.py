"""
Assemble the player-prop training table: one row per player per game.

  input:  data/processed/player_boxscores_with_rolling.csv  (339,841 rows)
          data/processed/games_final.csv                    (team context)
  output: data/processed/player_dataset.csv

A NEW GRAIN. Everything else in this pipeline works at one row per team
per game, or one row per game. This is the first table at player-game
level, and it is what player-prop models train on.

DELIBERATELY REUSES EXISTING FEATURES, invents none. The player side is
the trailing averages already built and hand-verified in
build_player_rolling_minutes.py; the team side is five columns lifted
straight from games_final.csv, each already trusted in production for
months. The only new work here is the join and the feature/label split.

WHAT COUNTS AS A FEATURE, and why MIN_NUMERIC is not one. Every feature is
either a trailing average over games already played, or a pre-game team
fact (rest, home/away, Elo). This game's own MIN_NUMERIC is a post-game
outcome: knowing a player logged 38 minutes tells you most of what you
need to guess his points, so using it would leak the answer. It is kept in
the output, grouped with the labels, because minutes is itself a real prop
market - but it can never be an input.

ROWS WITHOUT MINUTES ARE DROPPED. A player who did not appear has no line
to predict, so those rows are not training examples. They were essential
for the availability features, which is why they exist upstream, but they
are noise here. Expected survivors: 280,943, the "rows with minutes" count
the rolling build reports.

GAME_ID DTYPE. The player file stores GAME_ID zero-padded ("0021500003")
because the box-score endpoints require that form, while games_final.csv
has always used a plain integer. Note that a default pd.read_csv() infers
the padded column as int64 anyway - so the two would appear to merge
cleanly by luck. This script reads it as text and converts deliberately,
because relying on inference for a merge key is exactly what broke the
regression guard in the rolling build: one side declared, the other
inferred, and pandas refused to join them.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PLAYER_ROLLING_PATH = PROCESSED_DATA_DIR / "player_boxscores_with_rolling.csv"
GAMES_FINAL_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "player_dataset.csv"

MERGE_KEYS = ["GAME_ID", "TEAM_ID"]

# Pre-game team facts, already built and trusted. OPPONENT_ELO is the
# closest thing available to opponent strength, which matters more for a
# player's line than for a team total.
TEAM_CONTEXT_COLUMNS = [
    "IS_HOME",
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
    "OPPONENT_ELO",
]

# Four of the five are never NaN in games_final.csv. REST_DAYS is, for
# exactly 30 team-games: each team's first appearance in the dataset
# (2015-10-27/28), where build_rest_days.py's diff() has no prior game to
# measure against. That is legitimate "insufficient history", the same
# convention as an incomplete rolling window - NOT a failed merge.
#
# The distinction matters because the two are indistinguishable if you only
# count NaN. Merge success is proved by the indicator in
# attach_team_context(); NaN is reported separately below. Conflating them
# would either mask a broken join or fail the run over 30 correct rows.
CONTEXT_ALWAYS_PRESENT = [
    "IS_HOME",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
    "OPPONENT_ELO",
]
CONTEXT_MAY_BE_NAN = ["REST_DAYS"]

ROLLING_STATS = ["MIN", "PTS", "REB", "AST", "FG3M", "PRA"]
WINDOWS = [5, 10]

# Trailing averages over prior appearances only - shift(1) applied upstream.
PLAYER_FEATURE_COLUMNS = [
    f"ROLL{window}_{stat}" for window in WINDOWS for stat in ROLLING_STATS
]
FEATURE_COLUMNS = PLAYER_FEATURE_COLUMNS + TEAM_CONTEXT_COLUMNS

# This game's actual production. Never inputs. MIN_NUMERIC sits here rather
# than with the features for the leakage reason in the module docstring.
LABEL_COLUMNS = ["MIN_NUMERIC", "PTS", "REB", "AST", "FG3M", "PRA"]

# Identifiers: neither features nor labels, kept so a row can be traced
# back to a real player and game.
ID_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
]

EXPECTED_ROWS = 280_943


def load_players() -> pd.DataFrame:
    """Player-game rows, with merge keys converted deliberately."""
    needed = ID_COLUMNS + LABEL_COLUMNS + PLAYER_FEATURE_COLUMNS
    players = pd.read_csv(
        PLAYER_ROLLING_PATH,
        usecols=needed,
        dtype={"GAME_ID": str, "TEAM_ID": str, "PLAYER_ID": str},
        low_memory=False,
    )

    # Read as text, then converted here. See the module docstring: a
    # default read happens to infer int64 and would merge by luck.
    players["GAME_ID"] = players["GAME_ID"].astype(int)
    players["TEAM_ID"] = players["TEAM_ID"].astype(int)
    players["PLAYER_ID"] = players["PLAYER_ID"].astype(int)
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])

    print(f"Loaded {len(players):,} player-game rows from "
          f"{PLAYER_ROLLING_PATH.name}")
    return players


def load_team_context() -> pd.DataFrame:
    """One row per team-game: the pre-game context a player inherits."""
    games = pd.read_csv(
        GAMES_FINAL_PATH, usecols=MERGE_KEYS + TEAM_CONTEXT_COLUMNS
    )

    duplicates = int(games.duplicated(subset=MERGE_KEYS).sum())
    if duplicates:
        raise RuntimeError(
            f"games_final.csv has {duplicates:,} duplicate (GAME_ID, TEAM_ID) "
            f"pairs; the many-to-one merge below assumes it is unique."
        )

    print(f"Loaded {len(games):,} team-game context rows from "
          f"{GAMES_FINAL_PATH.name}")
    return games


def attach_team_context(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Give every player the pre-game context of the team he played for.

    many_to_one, not one_to_one: a dozen or more players legitimately share
    the same team-game row. That is the whole point of this grain.
    """
    before = len(players)
    merged = players.merge(
        games,
        on=MERGE_KEYS,
        how="left",
        validate="many_to_one",
        indicator="_context_merge",
    )

    if len(merged) != before:
        raise RuntimeError(f"merge changed rows: {before:,} -> {len(merged):,}")

    unmatched = merged[merged["_context_merge"] != "both"]
    print(f"\nTeam-context merge: {len(merged):,} rows, "
          f"{len(unmatched):,} without a match")
    if not unmatched.empty:
        sample = unmatched[MERGE_KEYS].drop_duplicates().head(5)
        raise RuntimeError(
            f"{len(unmatched):,} player rows found no team-game context. Every "
            f"player row comes from a game in games_final.csv, so this is a "
            f"failed merge - check the GAME_ID/TEAM_ID dtypes. First few:\n"
            f"{sample.to_string(index=False)}"
        )

    return merged.drop(columns=["_context_merge"])


def drop_players_who_sat(players: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a real line to predict."""
    played = players["MIN_NUMERIC"].notna()
    dropped = int((~played).sum())
    kept = players[played].reset_index(drop=True)

    print(f"\nDropped {dropped:,} rows where the player did not appear "
          f"({dropped / len(players) * 100:.1f}%), leaving {len(kept):,}.")
    return kept


def check_column_split(dataset: pd.DataFrame) -> None:
    """Every column is exactly one of: id, feature, label.

    The same discipline build_final_dataset.py applies. An unclassified
    column is how a post-game outcome quietly becomes a model input.
    """
    overlap = set(FEATURE_COLUMNS) & set(LABEL_COLUMNS)
    if overlap:
        raise RuntimeError(f"columns are both feature and label: {sorted(overlap)}")

    classified = set(ID_COLUMNS) | set(FEATURE_COLUMNS) | set(LABEL_COLUMNS)
    unclassified = set(dataset.columns) - classified
    missing = classified - set(dataset.columns)
    if unclassified or missing:
        raise RuntimeError(
            "column groups are out of sync with the dataset.\n"
            f"  in the frame but unclassified: {sorted(unclassified)}\n"
            f"  expected but absent: {sorted(missing)}"
        )


def main():
    players = load_players()
    games = load_team_context()

    dataset = attach_team_context(players, games)
    dataset = drop_players_who_sat(dataset)

    dataset = dataset[ID_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS]
    check_column_split(dataset)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows     : {len(dataset):,}  (expected {EXPECTED_ROWS:,} -> "
          f"{'MATCH' if len(dataset) == EXPECTED_ROWS else 'MISMATCH'})")
    print(f"  columns  : {len(dataset.columns)} "
          f"({len(ID_COLUMNS)} id + {len(FEATURE_COLUMNS)} feature "
          f"+ {len(LABEL_COLUMNS)} label)")
    print(f"  players  : {dataset['PLAYER_ID'].nunique():,}")
    print(f"  games    : {dataset['GAME_ID'].nunique():,}")
    print(f"  seasons  : {dataset['SEASON'].min()} - {dataset['SEASON'].max()}")

    # Merge success was already proved by the indicator in
    # attach_team_context(). These two blocks are about value availability,
    # which is a different question - see the note by CONTEXT_MAY_BE_NAN.
    print("\n  NaN in team-context columns that must never be NaN:")
    strict_nan = dataset[CONTEXT_ALWAYS_PRESENT].isna().sum()
    print(strict_nan.to_string())
    print(f"  context check: {'PASS' if strict_nan.sum() == 0 else 'FAIL'}")
    if strict_nan.sum():
        raise RuntimeError(
            "These columns are never NaN in games_final.csv, so a NaN here "
            "means the join produced rows it should not have."
        )

    print("\n  NaN in REST_DAYS (expected: each team's first game in the data):")
    rest_nan = int(dataset["REST_DAYS"].isna().sum())
    affected = dataset.loc[dataset["REST_DAYS"].isna(), "TEAM_ID"].nunique()
    print(f"    {rest_nan:,} player-rows across {affected} teams "
          f"- legitimate, not a merge failure")

    # These DO have legitimate NaN, from the rolling warm-up - a different
    # thing entirely from a failed merge, so they are reported separately.
    print("\n  NaN in the player rolling features (early-season warm-up, expected):")
    for column in PLAYER_FEATURE_COLUMNS:
        print(f"    {column:<16} {int(dataset[column].isna().sum()):>7,}")

    print("\n  Labels:")
    print(dataset[LABEL_COLUMNS].describe().to_string())

    print("\n  Preview:")
    preview = dataset[dataset["ROLL10_PRA"].notna()].head(5)
    print(preview[["GAME_DATE", "PLAYER_NAME", "TEAM_ABBREVIATION", "IS_HOME",
                   "TEAM_ELO", "ROLL10_PTS", "ROLL10_PRA", "PTS", "PRA"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
