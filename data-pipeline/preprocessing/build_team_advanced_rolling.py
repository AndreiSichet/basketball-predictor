"""
Trailing averages of each team's advanced ratings, one row per team-game.

  input:  data/raw/team_advanced_stats/*.csv  (13,199 files)
          data/processed/games_final.csv      (for GAME_DATE)
  output: data/processed/team_advanced_rolling.csv  (26,398 rows)

Same grain and same discipline as build_rolling_features.py, which produces
ROLL5_PTS and friends: shift(1) so a game's own result never leaks into its
own feature, and a season reset because a team's identity changes between
seasons. WINDOWS and derive_season are imported from that module rather
than restated, so the windows and the August season boundary stay defined
in exactly one place.

WHY THIS PHASE IS SIMPLE, unlike the availability work: these are outcomes
of games already played, at the grain the pipeline already uses. There is
no live-source gap - a team's trailing average of its own past
offensiveRating is exactly as computable for tomorrow's game as ROLL10_PTS
already is, because both depend only on games that have happened. No
serving-time fetch, no NaN-degradation path, no packaging question.

FIVE METRICS, NOT ALL 30. offensiveRating, defensiveRating, netRating, pace
and trueShootingPercentage are the interpretable core. The rest are either
derivable from the box scores the pipeline already has (rebound and assist
percentages) or known-unreliable: every estimated* column disagrees with
its real counterpart by roughly 5x (estimatedOffensiveRating 576.2 against
offensiveRating 115.2 in the same row), because it uses a different
internal formula. Same "ship the defensible core first" call as the
injury-status policy.

GAME_ID IS WRITTEN ZERO-PADDED, matching team_availability.csv, so both
merge into build_final_dataset.py the same way: read with
dtype={"GAME_ID": str}, then .astype(int) to meet games_final.csv's plain
integer convention. A default read silently strips the padding.
"""

import sys
from pathlib import Path

import pandas as pd

# WINDOWS and the season boundary come from the team-level rolling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rolling_features import WINDOWS, derive_season  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_DIR = DATA_DIR / "raw" / "team_advanced_stats"
PROCESSED_DIR = DATA_DIR / "processed"
GAMES_FINAL_PATH = PROCESSED_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DIR / "team_advanced_rolling.csv"

# Source column -> the pipeline's naming convention, matching the existing
# ROLL5_FG_PCT / ROLL10_TOV style rather than the endpoint's camelCase.
METRICS = {
    "offensiveRating": "OFF_RATING",
    "defensiveRating": "DEF_RATING",
    "netRating": "NET_RATING",
    "pace": "PACE",
    "trueShootingPercentage": "TS_PCT",
}

GROUP_KEYS = ["TEAM_ID", "SEASON"]
SORT_KEYS = ["TEAM_ID", "SEASON", "GAME_DATE", "GAME_ID"]

EXPECTED_ROWS = 26_398
PROGRESS_EVERY = 2000


def load_all_advanced() -> pd.DataFrame:
    """Read every per-game file into one frame.

    gameId is read as text so the zero padding survives; it is the join key
    and losing the padding would silently break the merge.
    """
    paths = sorted(INPUT_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit(f"{INPUT_DIR} is empty - run the ingestion first.")

    print(f"Reading {len(paths):,} advanced box score files...")

    keep = ["gameId", "teamId"] + list(METRICS)
    frames = []
    for i, path in enumerate(paths, start=1):
        frames.append(pd.read_csv(path, usecols=keep, dtype={"gameId": str}))
        if i % PROGRESS_EVERY == 0:
            print(f"  {i:,} / {len(paths):,}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"gameId": "GAME_ID", "teamId": "TEAM_ID"})
    print(f"  combined: {len(combined):,} team-game rows\n")
    return combined


def attach_game_date(teams: pd.DataFrame) -> pd.DataFrame:
    """Join GAME_DATE from the already-trusted game table.

    games_final.csv holds one row per team per game, so it has two rows for
    every GAME_ID. Merging it as-is would double every row. Deduping first
    is the fix; validate="m:1" is the seatbelt that turns a regression here
    into an exception rather than a silently doubled dataset.
    """
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "GAME_DATE"])
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    before = len(games)
    games = games.drop_duplicates(subset="GAME_ID")
    print(f"games_final.csv: {before:,} rows -> {len(games):,} unique GAME_IDs")

    merged = teams.merge(games, on="GAME_ID", how="left", validate="m:1")

    if len(merged) != len(teams):
        raise RuntimeError(f"join changed rows: {len(teams):,} -> {len(merged):,}")

    unmatched = int(merged["GAME_DATE"].isna().sum())
    if unmatched:
        raise RuntimeError(f"{unmatched:,} rows have no matching GAME_DATE")

    merged["GAME_DATE"] = pd.to_datetime(merged["GAME_DATE"])
    return merged


def add_rolling(teams: pd.DataFrame) -> pd.DataFrame:
    """ROLL5/ROLL10 trailing means per team, reset each season."""
    teams = teams.sort_values(SORT_KEYS).reset_index(drop=True)
    grouped = teams.groupby(GROUP_KEYS)

    for window in WINDOWS:
        for source_column, feature_name in METRICS.items():
            teams[f"ROLL{window}_{feature_name}"] = grouped[source_column].transform(
                lambda s, w=window: s.shift(1).rolling(w).mean()
            )

    return teams


def expected_nan_counts(teams: pd.DataFrame) -> dict:
    """How many NaN each window must produce, derived rather than eyeballed.

    A team-season's first `window` games have no complete trailing window,
    so they are NaN by design. Capped at the season's length, which matters
    for any team-season shorter than the window.
    """
    sizes = teams.groupby(GROUP_KEYS).size()
    return {window: int(sizes.clip(upper=window).sum()) for window in WINDOWS}


def main():
    teams = load_all_advanced()
    teams = attach_game_date(teams)
    teams["SEASON"] = derive_season(teams["GAME_DATE"])
    teams = add_rolling(teams)

    rolling_columns = [
        f"ROLL{window}_{name}" for window in WINDOWS for name in METRICS.values()
    ]
    # Minimal output, like team_availability.csv: GAME_ID + TEAM_ID is the
    # merge key, and carrying GAME_DATE would collide with games_master's
    # own column on merge.
    output = teams[["GAME_ID", "TEAM_ID"] + rolling_columns]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"  rows    : {len(output):,}  (expected {EXPECTED_ROWS:,} -> "
          f"{'MATCH' if len(output) == EXPECTED_ROWS else 'MISMATCH'})")
    print(f"  columns : {len(output.columns)}  "
          f"({len(rolling_columns)} rolling + GAME_ID + TEAM_ID)")
    print(f"  teams   : {output['TEAM_ID'].nunique()}")
    print(f"  team-seasons: {teams.groupby(GROUP_KEYS).ngroups:,}")

    expected = expected_nan_counts(teams)
    print("\n  NaN counts (early-season warm-up, by design):")
    all_match = True
    for window in WINDOWS:
        for name in METRICS.values():
            column = f"ROLL{window}_{name}"
            actual = int(output[column].isna().sum())
            ok = actual == expected[window]
            all_match &= ok
            print(f"    {column:<22} {actual:>6,}  expected {expected[window]:>6,}  "
                  f"{'OK' if ok else 'MISMATCH'}")
    print(f"\n  all NaN counts as predicted: {all_match}")

    print("\n  Preview (first rows past the ROLL10 warm-up):")
    preview = output[output["ROLL10_OFF_RATING"].notna()].head(5)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
