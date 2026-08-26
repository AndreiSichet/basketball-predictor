"""
Combine every player box score into one table and add each player's
trailing 10-game average minutes.

  input:  data/raw/player_boxscores/*.csv  (13,199 files)
          data/processed/games_final.csv   (for GAME_DATE)
  output: data/processed/player_boxscores_with_rolling.csv

This is the foundation for the availability features: before you can say "a
normally-important player is missing", you need to know how important that
player normally was, measured strictly from games already played.

Same shift-then-roll discipline as build_rolling_features.py, and the same
season reset, for the same reasons: shift(1) so a game's own minutes never
leak into its own feature, and a reset because a player's role genuinely
changes between seasons - trades, new teams, new depth charts.

TWO COLUMNS, TWO DIFFERENT QUESTIONS - the distinction matters:

  MIN_NUMERIC is NaN when a player did not appear. It records what
  happened, and "did not play" is not "played zero minutes", the same
  distinction the raw files are careful about.

  ROLL10_MIN answers "when this player is available, how many minutes does
  he play". It is computed over PLAYED GAMES ONLY - the window means the
  last 10 games he actually appeared in, not the last 10 calendar games -
  and then carried forward across the games he missed, so every row has a
  current known role to look up.

WHY PLAYED-GAMES-ONLY, and not "count an absence as zero minutes":

  Take a starter who missed 5 of his last 10 team games. Counting absences
  as zero averages him to ~15 minutes; over his last 10 played games he
  averages ~30. If he is out again today, the team-level absence feature
  would credit the team with missing 15 minutes instead of 30 - and that
  runs backwards. A chronically injured star being out again is a bigger
  problem, not a shrinking one. Counting absences as zero lets a player's
  own injury history quietly deflate the measured cost of his next absence.

  Rolling over MIN_NUMERIC directly is not an option either: pandas'
  default min_periods equals the window, so a single NaN inside it yields
  NaN, and 17.3% of player-rows are absences. The column would be NaN for
  most rows, and specifically NaN right after a player had been missing.

NaN CONVENTION: a row is NaN until the player has 11 played games in that
season - the first ten build the window, and the eleventh is the first game
with a full trailing window behind it. Absence rows before that point are
NaN too, since there is no known role to carry forward yet. Same
"insufficient history" convention as every other rolling feature here.

GAME_ID stays a zero-padded 10-character string in the output. Read this
file back with dtype={"GAME_ID": str} - a default read turns it into an
integer and silently drops the padding.
"""

import sys
from pathlib import Path

import pandas as pd

# has_played and the schema come from the script that writes the raw files.
INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
from fetch_player_boxscores import (  # noqa: E402
    OUTPUT_DIR as BOXSCORE_DIR,
    TARGET_COLUMNS,
    has_played,
)

# derive_season is a sibling in this directory: the August boundary stays
# defined in exactly one place, as it already is for team features.
from build_rolling_features import derive_season  # noqa: E402

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
GAMES_FINAL_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "player_boxscores_with_rolling.csv"

ROLLING_WINDOW = 10
ROLLING_COLUMN = f"ROLL{ROLLING_WINDOW}_MIN"
PLAYER_SEASON_KEYS = ["PLAYER_ID", "SEASON"]
SORT_KEYS = ["PLAYER_ID", "SEASON", "GAME_DATE", "GAME_ID"]

PROGRESS_EVERY = 2000


def load_all_boxscores() -> pd.DataFrame:
    """Read every per-game file into one frame.

    dtype=str with keep_default_na=False keeps zero-padded GAME_IDs intact
    and preserves the empty-string-means-absent convention the raw files
    use, which has_played() depends on.
    """
    paths = sorted(BOXSCORE_DIR.glob("*.csv"))
    print(f"Reading {len(paths):,} box score files...")

    frames = []
    for i, path in enumerate(paths, start=1):
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
        if i % PROGRESS_EVERY == 0:
            print(f"  {i:,} / {len(paths):,}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  combined: {len(combined):,} player-rows\n")
    return combined


def attach_game_date(players: pd.DataFrame) -> pd.DataFrame:
    """Join GAME_DATE in from the already-trusted game table.

    games_final.csv holds one row per team per game, so it has two rows for
    every GAME_ID. Merging it as-is would silently double every player row.
    Deduping first is the fix; validate="m:1" is the seatbelt, turning a
    regression here into an exception instead of 679,682 rows.
    """
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "GAME_DATE"])
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    before = len(games)
    games = games.drop_duplicates(subset="GAME_ID")
    print(f"games_final.csv: {before:,} rows -> {len(games):,} unique GAME_IDs")

    merged = players.merge(games, on="GAME_ID", how="left", validate="m:1")

    if len(merged) != len(players):
        raise RuntimeError(
            f"join changed the row count: {len(players):,} -> {len(merged):,}"
        )

    unmatched = merged["GAME_DATE"].isna().sum()
    if unmatched:
        raise RuntimeError(f"{unmatched:,} player-rows have no matching GAME_DATE")

    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])
    return merged


def parse_minutes(players: pd.DataFrame) -> pd.DataFrame:
    """MIN ("MM:SS") -> MIN_NUMERIC (float), NaN where the player sat.

    A parse failure would look identical to an absence, so anything that
    claims to have played but will not parse is raised, not shrugged off.
    """
    played = has_played(players["MIN"])

    parts = players.loc[played, "MIN"].str.split(":", expand=True)
    minutes = pd.to_numeric(parts[0], errors="coerce")
    seconds = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0

    players["MIN_NUMERIC"] = float("nan")
    players.loc[played, "MIN_NUMERIC"] = minutes + seconds / 60.0

    unparsed = played & players["MIN_NUMERIC"].isna()
    if unparsed.any():
        examples = players.loc[unparsed, "MIN"].head(5).tolist()
        raise RuntimeError(
            f"{int(unparsed.sum()):,} rows have minutes that would not parse, "
            f"e.g. {examples}"
        )

    return players


def add_rolling_minutes(players: pd.DataFrame) -> pd.DataFrame:
    """Trailing average over played games, carried across absences."""
    players = players.sort_values(SORT_KEYS).reset_index(drop=True)
    played = players["MIN_NUMERIC"].notna()

    # Roll over the played rows only, so the window is "his last 10
    # appearances", never diluted by games he sat out.
    played_only = players.loc[played]
    rolled = played_only.groupby(PLAYER_SEASON_KEYS)["MIN_NUMERIC"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW).mean()
    )

    players[ROLLING_COLUMN] = float("nan")
    players.loc[played, ROLLING_COLUMN] = rolled

    # Carry the last known role across the games he missed, so an absence
    # row still says what the team is missing. Rows are already sorted by
    # date within each player-season, which is what makes ffill correct.
    players[ROLLING_COLUMN] = players.groupby(PLAYER_SEASON_KEYS)[
        ROLLING_COLUMN
    ].ffill()

    return players


def expected_nan_count(players: pd.DataFrame) -> int:
    """How many rows must be NaN, derived rather than eyeballed.

    After the forward fill, a row's value is the value of the most recent
    played row at or before it, and a played row only has a value once it
    is that player-season's 11th appearance. So a row is NaN exactly when
    ten or fewer played games have occurred up to and including it -
    which covers absence rows and short player-seasons without any special
    casing.
    """
    played = players["MIN_NUMERIC"].notna().astype(int)
    appearances_so_far = played.groupby(
        [players["PLAYER_ID"], players["SEASON"]]
    ).cumsum()
    return int((appearances_so_far <= ROLLING_WINDOW).sum())


def report_staleness(players: pd.DataFrame):
    """How far back does a carried-forward value actually reach?

    Informational only. A player on a long-term injury could inherit a role
    measured before the injury started; this measures whether that is a
    real pattern in the data or a theoretical worry.
    """
    played = players["MIN_NUMERIC"].notna()

    last_played_date = players["GAME_DATE"].where(played)
    last_played_date = last_played_date.groupby(
        [players["PLAYER_ID"], players["SEASON"]]
    ).ffill()

    position = players.groupby(PLAYER_SEASON_KEYS).cumcount()
    last_played_position = position.where(played).groupby(
        [players["PLAYER_ID"], players["SEASON"]]
    ).ffill()

    carried = (~played) & players[ROLLING_COLUMN].notna()
    gap_days = (players["GAME_DATE"] - last_played_date).dt.days[carried]
    gap_games = (position - last_played_position)[carried]

    print("\n" + "=" * 66)
    print("STALENESS OF CARRIED-FORWARD VALUES (informational)")
    print("=" * 66)
    print(f"  absence rows with a carried value: {int(carried.sum()):,}")
    print("\n  gap back to the last game the player actually played:")
    print(f"    {'':<10}{'days':>10}{'team games':>14}")
    for label, q in (("median", 0.50), ("75th", 0.75), ("90th", 0.90),
                     ("99th", 0.99), ("max", 1.00)):
        print(f"    {label:<10}{gap_days.quantile(q):>10.0f}{gap_games.quantile(q):>14.0f}")

    for threshold in (10, 20, 40):
        share = (gap_games > threshold).mean() * 100
        print(f"    carried across more than {threshold:>2} team games: {share:5.2f}%")


def main():
    players = load_all_boxscores()
    players = attach_game_date(players)
    players["SEASON"] = derive_season(players["GAME_DATE"])
    players = parse_minutes(players)
    players = add_rolling_minutes(players)

    output_columns = TARGET_COLUMNS + [
        "GAME_DATE", "SEASON", "MIN_NUMERIC", ROLLING_COLUMN
    ]
    players = players[output_columns]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    players.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    expected = expected_nan_count(players)
    actual = int(players[ROLLING_COLUMN].isna().sum())

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows              : {len(players):,}")
    print(f"  columns           : {len(players.columns)}")
    print(f"  date range        : {players['GAME_DATE'].min().date()} "
          f"-> {players['GAME_DATE'].max().date()}")
    print(f"  unique players    : {players['PLAYER_ID'].nunique():,}")
    print(f"  player-seasons    : {players.groupby(PLAYER_SEASON_KEYS).ngroups:,}")
    print(f"  rows with minutes : {int(players['MIN_NUMERIC'].notna().sum()):,}")
    print(f"\n  {ROLLING_COLUMN} NaN : {actual:,}")
    print(f"  expected          : {expected:,}  -> "
          f"{'MATCH' if actual == expected else 'MISMATCH'}")

    report_staleness(players)


if __name__ == "__main__":
    main()
