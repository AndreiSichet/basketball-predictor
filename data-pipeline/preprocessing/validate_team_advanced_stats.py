"""
Read-only quality gate for the team advanced stats in
data-pipeline/data/raw/team_advanced_stats/.

Structural checks plus three independent consistency checks, in ascending
order of how much they prove:

  1. PACE agrees between the two teams in a game. Possessions are shared -
     every possession belongs to exactly one team - so both rows describe
     the same number. Internal to one file.
  2. netRating == offensiveRating - defensiveRating. Also internal, but
     across three separately-reported columns.
  3. **The centrepiece: PTS reconstructed from the rating.** Offensive
     rating is points per 100 possessions, so
     offensiveRating * possessions / 100 must reproduce that team's actual
     final score in games_final.csv - a table trusted in production for
     months. This is the direct parallel to the player box scores' PTS-sum
     check, and it is the only one of the three that validates this data
     against an independent source rather than against itself.

Checks 1 and 2 were confirmed by hand on two sample games during the
endpoint investigation (PACE differing by 0.0000, netRating matching
exactly). Running them across all 13,199 is what turns two samples into a
property of the dataset.

TOLERANCE. Check 3 will not be exact: the endpoint reports offensiveRating
rounded to one decimal, and possessions is itself a derived quantity, so a
game with ~100 possessions inherits roughly +/- 0.05 * 100 / 100 points of
slack from the rounding alone. PTS_TOLERANCE starts at 2.0 and the run
prints the full distribution of differences, so the threshold can be tuned
to what the data actually does rather than to a guess.

SCHEMA IS DERIVED, NOT HARDCODED. nba_api declares 29 TeamStats columns
while the live response carries 30, so a hardcoded list would encode
whichever number happened to be right on the day. Instead the first file
read defines the canonical order and every other file is checked against
it - which tests the invariant that actually matters (all files agree)
rather than a constant that could drift.

Does not modify or write anything.

Run:  python data-pipeline/preprocessing/validate_team_advanced_stats.py
"""

import sys
from pathlib import Path

import pandas as pd

INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
from fetch_team_advanced_stats import (  # noqa: E402
    EXPECTED_ROWS_PER_GAME,
    OUTPUT_DIR,
    REQUIRED_COLUMNS,
)

GAMES_FINAL_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "games_final.csv"

# Both derived quantities are reported rounded, so exact equality is not
# expected. See the tolerance note in the module docstring.
PACE_TOLERANCE = 0.05
NET_RATING_TOLERANCE = 0.15
PTS_TOLERANCE = 2.0

EXAMPLES_TO_PRINT = 5
PROGRESS_EVERY = 2500


def load_expected_game_ids() -> set:
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"])
    return {str(gid).zfill(10) for gid in games["GAME_ID"].unique()}


def load_team_facts() -> dict:
    """(game_id, team_id) -> real final PTS, from the already-trusted table."""
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "PTS"])
    return {
        (str(row.GAME_ID).zfill(10), int(row.TEAM_ID)): float(row.PTS)
        for row in games.itertuples()
    }


def check_structure(game_id, frame, canonical_columns, issues):
    """Per-file structural invariants. Returns False if further checks are moot."""
    if list(frame.columns) != canonical_columns:
        issues["column_mismatch"].append(game_id)
        return False

    if len(frame) != EXPECTED_ROWS_PER_GAME:
        issues["wrong_row_count"].append(f"{game_id} ({len(frame)} rows)")
        return False

    if not (frame["gameId"].astype(str).str.zfill(10) == game_id).all():
        issues["game_id_mismatch"].append(game_id)

    if frame["teamId"].nunique() != EXPECTED_ROWS_PER_GAME:
        issues["duplicate_team"].append(game_id)

    return True


def check_pace(game_id, frame, issues):
    """Possessions are shared, so both teams must report the same pace."""
    values = pd.to_numeric(frame["pace"], errors="coerce")
    if values.isna().any():
        issues["pace_unparseable"].append(game_id)
        return

    spread = float(values.max() - values.min())
    if spread > PACE_TOLERANCE:
        issues["pace_mismatch"].append(
            f"{game_id}: {values.tolist()} differ by {spread:.4f}"
        )


def check_net_rating(game_id, frame, issues):
    """netRating must equal offensiveRating - defensiveRating."""
    offensive = pd.to_numeric(frame["offensiveRating"], errors="coerce")
    defensive = pd.to_numeric(frame["defensiveRating"], errors="coerce")
    net = pd.to_numeric(frame["netRating"], errors="coerce")

    if offensive.isna().any() or defensive.isna().any() or net.isna().any():
        issues["rating_unparseable"].append(game_id)
        return

    difference = (net - (offensive - defensive)).abs().max()
    if difference > NET_RATING_TOLERANCE:
        issues["net_rating_mismatch"].append(
            f"{game_id}: off-def vs net differ by {difference:.4f}"
        )


def check_points(game_id, frame, team_points, issues, differences):
    """The centrepiece: rebuild PTS from the rating and compare to reality.

    offensiveRating is points per 100 possessions, so
    rating * possessions / 100 must be that team's actual final score.
    """
    for row in frame.itertuples():
        team_id = int(row.teamId)
        actual = team_points.get((game_id, team_id))

        if actual is None:
            issues["team_not_in_games_final"].append(f"{game_id} team {team_id}")
            continue

        rating = pd.to_numeric(row.offensiveRating, errors="coerce")
        possessions = pd.to_numeric(row.possessions, errors="coerce")
        if pd.isna(rating) or pd.isna(possessions):
            issues["points_unparseable"].append(f"{game_id} team {team_id}")
            continue

        implied = float(rating) * float(possessions) / 100.0
        difference = implied - actual
        differences.append(difference)

        if abs(difference) > PTS_TOLERANCE:
            issues["points_mismatch"].append(
                f"{game_id} team {team_id}: rating {rating:.1f} x {possessions:.0f} poss "
                f"/ 100 = {implied:.2f} vs actual {actual:.0f} (diff {difference:+.2f})"
            )


def describe(differences):
    """The distribution the tolerance should actually be set from."""
    if not differences:
        print("  no comparable rows")
        return

    series = pd.Series(differences)
    absolute = series.abs()
    print(f"  rows compared      : {len(series):,}")
    print(f"  mean difference    : {series.mean():+.4f}")
    print(f"  median |difference|: {absolute.median():.4f}")
    print("  |difference| percentiles:")
    for q in (0.50, 0.90, 0.99, 0.999, 1.00):
        print(f"    {q * 100:>6.1f}th : {absolute.quantile(q):.4f}")
    for threshold in (0.5, 1.0, 2.0, 5.0):
        share = (absolute > threshold).mean() * 100
        print(f"  beyond +/-{threshold:<4}    : {share:6.3f}%")


def main():
    if not OUTPUT_DIR.exists():
        raise SystemExit(f"{OUTPUT_DIR} does not exist - run the ingestion first.")

    expected_ids = load_expected_game_ids()
    team_points = load_team_facts()

    paths = sorted(OUTPUT_DIR.glob("*.csv"))
    found_ids = {p.stem for p in paths}

    print(f"Validating {len(paths):,} files in {OUTPUT_DIR}")
    print(f"games_final.csv lists {len(expected_ids):,} unique games.\n")

    print("=" * 70)
    print("COVERAGE")
    print("=" * 70)
    missing, extra = expected_ids - found_ids, found_ids - expected_ids
    print(f"  files found : {len(found_ids):,}")
    print(f"  missing     : {len(missing):,}")
    print(f"  extra       : {len(extra):,}")
    for label, group in (("missing", missing), ("extra", extra)):
        if group:
            print(f"    first {label}: {sorted(group)[:EXAMPLES_TO_PRINT]}")

    issues = {k: [] for k in (
        "unreadable", "column_mismatch", "wrong_row_count", "game_id_mismatch",
        "duplicate_team", "pace_unparseable", "pace_mismatch",
        "rating_unparseable", "net_rating_mismatch",
        "team_not_in_games_final", "points_unparseable", "points_mismatch",
    )}
    differences = []
    canonical_columns = None

    for i, path in enumerate(paths, start=1):
        game_id = path.stem
        try:
            frame = pd.read_csv(path, dtype={"gameId": str}, keep_default_na=False)
        except Exception as exc:
            issues["unreadable"].append(f"{game_id} ({type(exc).__name__})")
            continue

        if canonical_columns is None:
            canonical_columns = list(frame.columns)
            print(f"\nCanonical schema taken from {game_id}: "
                  f"{len(canonical_columns)} columns")
            absent = [c for c in REQUIRED_COLUMNS if c not in canonical_columns]
            if absent:
                raise SystemExit(
                    f"The first file is missing columns this feature needs: {absent}"
                )

        if not check_structure(game_id, frame, canonical_columns, issues):
            continue

        check_pace(game_id, frame, issues)
        check_net_rating(game_id, frame, issues)
        check_points(game_id, frame, team_points, issues, differences)

        if i % PROGRESS_EVERY == 0:
            print(f"  ...{i:,} / {len(paths):,} files checked")

    print("\n" + "=" * 70)
    print("CHECKS")
    print("=" * 70)
    labels = {
        "unreadable": "files that could not be read",
        "column_mismatch": "columns differ from the canonical order",
        "wrong_row_count": f"not exactly {EXPECTED_ROWS_PER_GAME} team rows",
        "game_id_mismatch": "gameId inside file != filename",
        "duplicate_team": "the two rows share a teamId",
        "pace_unparseable": "pace not numeric",
        "pace_mismatch": f"teams' pace differs by > {PACE_TOLERANCE}",
        "rating_unparseable": "ratings not numeric",
        "net_rating_mismatch": f"net != off - def by > {NET_RATING_TOLERANCE}",
        "team_not_in_games_final": "team missing from games_final",
        "points_unparseable": "rating/possessions not numeric",
        "points_mismatch": f"implied PTS off by > {PTS_TOLERANCE}",
    }
    for key, label in labels.items():
        found = issues[key]
        print(f"  [{'OK  ' if not found else 'FAIL'}] {label:<48} {len(found):,}")
        for example in found[:EXAMPLES_TO_PRINT]:
            print(f"           {example}")
        if len(found) > EXAMPLES_TO_PRINT:
            print(f"           ... and {len(found) - EXAMPLES_TO_PRINT:,} more")

    print("\n" + "=" * 70)
    print("IMPLIED-PTS DIFFERENCE DISTRIBUTION")
    print("=" * 70)
    print("  (rating x possessions / 100) - actual PTS, per team-game.")
    print(f"  Tolerance is currently +/-{PTS_TOLERANCE}; tune it from this.\n")
    describe(differences)

    total_failures = sum(len(v) for v in issues.values()) + len(missing) + len(extra)
    print("\n" + "=" * 70)
    print("PASS" if total_failures == 0 else f"FAIL - {total_failures:,} issue(s)")
    print("=" * 70)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
