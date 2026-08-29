"""
Combine every player box score into one table and add each player's
trailing per-game averages.

  input:  data/raw/player_boxscores/*.csv  (13,199 files)
          data/processed/games_final.csv   (for GAME_DATE)
  output: data/processed/player_boxscores_with_rolling.csv

Originally built for ROLL10_MIN alone, to weight the team-level absence
features. Now generalised to the stats player props are priced on:
minutes, points, rebounds, assists, threes, and PRA. The rolling logic is
one code path applied to every column - not five near-copies - because the
decisions it encodes were argued out once and should not be re-litigated
per stat.

Same shift-then-roll discipline as build_rolling_features.py, and the same
season reset, for the same reasons: shift(1) so a game's own line never
leaks into its own feature, and a reset because a player's role genuinely
changes between seasons - trades, new teams, new depth charts.

TWO KINDS OF COLUMN, TWO DIFFERENT QUESTIONS:

  MIN_NUMERIC, PTS, REB, AST, FG3M and PRA are NaN when a player did not
  appear. They record what happened, and "did not play" is not "scored
  zero" - the same distinction the raw files are careful about.

  The ROLL5_/ROLL10_ columns answer "when this player is available, what
  does he produce". They are computed over PLAYED GAMES ONLY - the window
  means his last N appearances, not his last N calendar games - and then
  carried forward across the games he missed, so every row has a current
  known level to look up.

WHY PLAYED-GAMES-ONLY, and not "count an absence as zero": a starter who
missed five of his last ten team games would average out to half his real
output, so a team missing him would be measured as missing half a player.
A chronically injured starter being out again is a bigger problem, not a
shrinking one. Rolling over NaN directly is not an option either - pandas'
default min_periods equals the window, so a single NaN inside it yields
NaN, and 17.3% of player-rows are absences.

PRA IS ROLLED, NOT RECONSTRUCTED. PRA = PTS + REB + AST is computed
per game, before rolling, and then rolled like any other column. Averaging
is linear, so ROLL10_PRA must equal ROLL10_PTS + ROLL10_REB + ROLL10_AST
exactly - the run checks that on every row. It is not a redundant check:
the three parts and the whole travel through the same shift, the same
window and the same forward fill, so a disagreement means the rolling
logic itself is wrong, not that this one derived column is off.

NaN CONVENTION: a row is NaN until the player has (window + 1) appearances
that season - the first N build the window, and the next is the first game
with a full trailing window behind it. Same "insufficient history" rule as
every other rolling feature here.

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

# WINDOWS and the season boundary come from the team-level rolling script,
# so both stay defined in exactly one place.
from build_rolling_features import WINDOWS, derive_season  # noqa: E402

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
GAMES_FINAL_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "player_boxscores_with_rolling.csv"

# Raw counting stats taken straight from the box score. FG3M is threes
# made; the raw files also carry FG3A, which props are not priced on.
COUNTING_STATS = ["PTS", "REB", "AST", "FG3M"]

# Derived per game, before any rolling. See the module docstring.
PRA_COLUMN = "PRA"

# Source column -> the suffix used in ROLL5_/ROLL10_ names. MIN_NUMERIC is
# the parsed form of the "MM:SS" string, and keeps the shorter MIN suffix
# so ROLL10_MIN stays exactly the name the availability features already
# depend on.
ROLLING_SOURCES = {
    "MIN_NUMERIC": "MIN",
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "FG3M": "FG3M",
    PRA_COLUMN: "PRA",
}

PLAYER_SEASON_KEYS = ["PLAYER_ID", "SEASON"]
SORT_KEYS = ["PLAYER_ID", "SEASON", "GAME_DATE", "GAME_ID"]

# ROLL10_MIN feeds team_availability.csv and, through it, the shipped
# 38-feature models. If a rewrite of this script ever changes it, that is a
# silent break in production, so the previous output is diffed before this
# one replaces it.
REGRESSION_GUARD_COLUMN = "ROLL10_MIN"

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


def parse_counting_stats(players: pd.DataFrame) -> pd.DataFrame:
    """Numeric PTS/REB/AST/FG3M plus PRA, all NaN where the player sat.

    The raw files store these as text and blank them for absent players, so
    to_numeric already yields NaN in the right places. The mask is applied
    anyway and then cross-checked: a row that claims minutes but has no
    points is a data problem worth stopping on, not rounding past.
    """
    played = has_played(players["MIN"])

    for column in COUNTING_STATS:
        players[column] = pd.to_numeric(players[column], errors="coerce")
        players.loc[~played, column] = float("nan")

        unparsed = played & players[column].isna()
        if unparsed.any():
            raise RuntimeError(
                f"{int(unparsed.sum()):,} rows played but have no parseable "
                f"{column}. MIN and the counting stats must agree about whether "
                f"a player appeared."
            )

    # Computed per game, before rolling - see the module docstring.
    players[PRA_COLUMN] = players["PTS"] + players["REB"] + players["AST"]

    return players


def add_rolling(players: pd.DataFrame) -> pd.DataFrame:
    """Trailing averages over played games, carried across absences.

    One code path for every stat and every window. The played mask is
    identical across stats (it comes from MIN), so each column's warm-up
    and forward fill line up exactly - which is what makes the PRA
    linearity check below meaningful rather than trivially true.
    """
    players = players.sort_values(SORT_KEYS).reset_index(drop=True)
    played = players["MIN_NUMERIC"].notna()
    played_only = players.loc[played]

    for window in WINDOWS:
        for source, suffix in ROLLING_SOURCES.items():
            column = f"ROLL{window}_{suffix}"

            # Roll over played rows only, so the window is "his last N
            # appearances", never diluted by games he sat out.
            rolled = played_only.groupby(PLAYER_SEASON_KEYS)[source].transform(
                lambda s, w=window: s.shift(1).rolling(w).mean()
            )

            players[column] = float("nan")
            players.loc[played, column] = rolled

            # Carry the last known level across games he missed. Rows are
            # already sorted by date within each player-season, which is
            # what makes ffill correct here.
            players[column] = players.groupby(PLAYER_SEASON_KEYS)[column].ffill()

    return players


def expected_nan_count(players: pd.DataFrame, window: int) -> int:
    """How many rows must be NaN, derived rather than eyeballed.

    After the forward fill, a row's value is that of the most recent played
    row at or before it, and a played row only has a value once it is that
    player-season's (window + 1)th appearance. So a row is NaN exactly when
    `window` or fewer appearances have occurred up to and including it -
    which covers absence rows and short player-seasons without any special
    casing.
    """
    played = players["MIN_NUMERIC"].notna().astype(int)
    appearances_so_far = played.groupby(
        [players["PLAYER_ID"], players["SEASON"]]
    ).cumsum()
    return int((appearances_so_far <= window).sum())


def check_pra_linearity(players: pd.DataFrame) -> bool:
    """ROLL{w}_PRA must equal ROLL{w}_PTS + ROLL{w}_REB + ROLL{w}_AST.

    Exhaustive, not sampled. Averaging is linear and all four columns share
    the same window, the same played rows and the same forward fill, so any
    disagreement points at the rolling machinery rather than at PRA.
    """
    print("\n  PRA linearity (mean is linear, so this must hold exactly):")
    all_ok = True

    for window in WINDOWS:
        direct = players[f"ROLL{window}_PRA"]
        summed = (players[f"ROLL{window}_PTS"]
                  + players[f"ROLL{window}_REB"]
                  + players[f"ROLL{window}_AST"])

        both_nan = direct.isna() & summed.isna()
        difference = (direct - summed).abs()
        worst = float(difference.max(skipna=True))
        disagreeing = int((~both_nan & ~(difference < 1e-9)).sum())

        all_ok &= disagreeing == 0
        print(f"    ROLL{window}_PRA vs sum of parts: {disagreeing:,} disagreeing rows, "
              f"largest difference {worst:.2e}  {'OK' if disagreeing == 0 else 'FAIL'}")

    return all_ok


def check_minutes_regression(players: pd.DataFrame) -> None:
    """ROLL10_MIN must be byte-identical to the previous run's.

    team_availability.csv and the shipped 38-feature models were built from
    it. Generalising this script must not move it, so the old output is
    diffed before it is replaced.
    """
    if not OUTPUT_PATH.exists():
        print(f"\n  {REGRESSION_GUARD_COLUMN} regression check: no previous output "
              f"to compare against (first run).")
        return

    # Both keys are read as text, matching the freshly-computed side, which
    # comes from load_all_boxscores() and is therefore str throughout.
    # Reading PLAYER_ID as Int64 here is what broke the first run: pandas
    # refuses to merge an object key against an Int64 one. Same root cause
    # as the GAME_ID zero-padding trap - a merge key whose dtype was
    # inferred on one side and declared on the other.
    previous = pd.read_csv(
        OUTPUT_PATH,
        usecols=["GAME_ID", "PLAYER_ID", REGRESSION_GUARD_COLUMN],
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )
    merged = players[["GAME_ID", "PLAYER_ID", REGRESSION_GUARD_COLUMN]].merge(
        previous, on=["GAME_ID", "PLAYER_ID"], how="inner",
        suffixes=("_new", "_old"), validate="one_to_one",
    )

    new = merged[f"{REGRESSION_GUARD_COLUMN}_new"]
    old = merged[f"{REGRESSION_GUARD_COLUMN}_old"]
    both_nan = new.isna() & old.isna()
    differing = int((~both_nan & ~((new - old).abs() < 1e-9)).sum())

    print(f"\n  {REGRESSION_GUARD_COLUMN} regression check "
          f"(availability features depend on this):")
    print(f"    rows compared: {len(merged):,}   differing: {differing:,}  "
          f"{'UNCHANGED' if differing == 0 else 'CHANGED - investigate'}")


def main():
    players = load_all_boxscores()
    players = attach_game_date(players)
    players["SEASON"] = derive_season(players["GAME_DATE"])
    players = parse_minutes(players)
    players = parse_counting_stats(players)
    players = add_rolling(players)

    rolling_columns = [
        f"ROLL{window}_{suffix}"
        for window in WINDOWS
        for suffix in ROLLING_SOURCES.values()
    ]
    output_columns = (
        TARGET_COLUMNS
        + ["GAME_DATE", "SEASON", "MIN_NUMERIC", PRA_COLUMN]
        + rolling_columns
    )
    players = players[output_columns]

    check_minutes_regression(players)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    players.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows              : {len(players):,}")
    print(f"  columns           : {len(players.columns)} "
          f"({len(rolling_columns)} rolling)")
    print(f"  date range        : {players['GAME_DATE'].min().date()} "
          f"-> {players['GAME_DATE'].max().date()}")
    print(f"  unique players    : {players['PLAYER_ID'].nunique():,}")
    print(f"  player-seasons    : {players.groupby(PLAYER_SEASON_KEYS).ngroups:,}")
    print(f"  rows with minutes : {int(players['MIN_NUMERIC'].notna().sum()):,}")

    print("\n  NaN counts (early-season warm-up, by design):")
    all_match = True
    for window in WINDOWS:
        expected = expected_nan_count(players, window)
        for suffix in ROLLING_SOURCES.values():
            column = f"ROLL{window}_{suffix}"
            actual = int(players[column].isna().sum())
            ok = actual == expected
            all_match &= ok
            print(f"    {column:<16} {actual:>7,}  expected {expected:>7,}  "
                  f"{'OK' if ok else 'MISMATCH'}")
    print(f"\n  all NaN counts as predicted: {all_match}")

    linear_ok = check_pra_linearity(players)

    print("\n  Preview (a player past the ROLL10 warm-up):")
    preview = players[players["ROLL10_PRA"].notna()].head(6)
    print(preview[["GAME_DATE", "PLAYER_NAME", "MIN", "PTS", "REB", "AST", "PRA",
                   "ROLL5_PTS", "ROLL10_PTS", "ROLL10_PRA"]].to_string(index=False))

    if not (all_match and linear_ok):
        raise SystemExit("Checks failed - see above.")


if __name__ == "__main__":
    main()
