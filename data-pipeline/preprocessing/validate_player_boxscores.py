"""
Read-only quality gate for the player box scores in
data-pipeline/data/raw/player_boxscores/.

Structural checks (shape, ids, duplicates, participation consistency) plus
one cross-validation that is worth more than all of them together: each
team's players' PTS summed from the box score must equal that team's actual
final PTS in games_final.csv. That compares a brand-new data domain against
an independent source already trusted and used in production for months.
A team's individual scoring has no legitimate way to sum to anything other
than its real final score, so any mismatch is a genuine defect - reported
by GAME_ID, team and margin, never averaged away.

Does not modify or write anything.

Every read is dtype=str with keep_default_na=False, for two reasons: it
keeps zero-padded GAME_IDs intact, and it preserves the distinction between
an empty MIN ("") and a genuinely absent value. A default read turns both
into NaN and would quietly hide the participation check this file exists
to perform.

Run:  python data-pipeline/preprocessing/validate_player_boxscores.py
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Import the schema and the participation rule rather than restating them,
# so this cannot drift from the script that writes the files.
INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
from fetch_player_boxscores import (  # noqa: E402
    OUTPUT_DIR,
    TARGET_COLUMNS,
    has_played,
)

GAMES_FINAL_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "games_final.csv"

EXPECTED_TEAMS_PER_GAME = 2
MIN_PLAUSIBLE_ROWS = 15
MAX_PLAUSIBLE_ROWS = 40

# Counting stats that must be present exactly when the player appeared.
COUNTING_STATS = ["PTS", "REB", "AST", "FGA", "MIN"]

EXAMPLES_TO_PRINT = 5


def read_lossless(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_expected_game_ids() -> set:
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"])
    return {str(gid).zfill(10) for gid in games["GAME_ID"].unique()}


def load_team_points() -> dict:
    """(game_id, team_id) -> final PTS, from the already-trusted table."""
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "PTS"])
    return {
        (str(row.GAME_ID).zfill(10), str(row.TEAM_ID)): float(row.PTS)
        for row in games.itertuples()
    }


def check_structure(game_id: str, frame: pd.DataFrame, issues: dict):
    """Per-file structural invariants."""
    if list(frame.columns) != TARGET_COLUMNS:
        issues["bad_columns"].append(game_id)
        # Everything below indexes by column name; bail out for this file.
        return False

    if not (frame["GAME_ID"] == game_id).all():
        issues["game_id_mismatch"].append(game_id)

    if frame["TEAM_ID"].nunique() != EXPECTED_TEAMS_PER_GAME:
        issues["wrong_team_count"].append(game_id)

    if frame.duplicated(subset=["TEAM_ID", "PLAYER_ID"]).any():
        issues["duplicate_players"].append(game_id)

    if not MIN_PLAUSIBLE_ROWS <= len(frame) <= MAX_PLAUSIBLE_ROWS:
        issues["implausible_row_count"].append(f"{game_id} ({len(frame)} rows)")

    return True


def check_participation(game_id: str, frame: pd.DataFrame, issues: dict):
    """MIN and the counting stats must agree about whether a player played.

    A row with minutes but no points column, or no minutes but a populated
    stat line, would mean the empty-string handling missed something.
    """
    played = has_played(frame["MIN"])

    for stat in COUNTING_STATS:
        stat_present = frame[stat].astype(str).str.strip() != ""
        if not stat_present.equals(played):
            mismatched = frame.loc[stat_present != played]
            issues["participation_mismatch"].append(
                f"{game_id} ({stat}, {len(mismatched)} row(s))"
            )
            break


def check_points_against_games_final(
    game_id: str, frame: pd.DataFrame, team_points: dict, issues: dict
):
    """The centrepiece: box score PTS must sum to the real final score."""
    points = pd.to_numeric(frame["PTS"], errors="coerce").fillna(0.0)

    for team_id, team_total in points.groupby(frame["TEAM_ID"]).sum().items():
        expected = team_points.get((game_id, str(team_id)))

        if expected is None:
            issues["team_not_in_games_final"].append(f"{game_id} team {team_id}")
            continue

        if float(team_total) != expected:
            issues["points_mismatch"].append(
                f"{game_id} team {team_id}: box score {team_total:.0f} "
                f"vs games_final {expected:.0f} (diff {team_total - expected:+.0f})"
            )


def main():
    if not OUTPUT_DIR.exists():
        raise SystemExit(f"{OUTPUT_DIR} does not exist.")

    expected_ids = load_expected_game_ids()
    team_points = load_team_points()

    paths = sorted(OUTPUT_DIR.glob("*.csv"))
    found_ids = {p.stem for p in paths}

    print(f"Validating {len(paths):,} files in {OUTPUT_DIR}")
    print(f"games_final.csv lists {len(expected_ids):,} unique games.\n")

    print("=" * 68)
    print("COVERAGE")
    print("=" * 68)
    missing = expected_ids - found_ids
    extra = found_ids - expected_ids
    print(f"  files found            : {len(found_ids):,}")
    print(f"  missing (in games_final, no file): {len(missing):,}")
    print(f"  extra (file, not in games_final) : {len(extra):,}")
    for label, group in (("missing", missing), ("extra", extra)):
        if group:
            print(f"    first {label}: {sorted(group)[:EXAMPLES_TO_PRINT]}")

    issues = {
        "bad_columns": [],
        "game_id_mismatch": [],
        "wrong_team_count": [],
        "duplicate_players": [],
        "implausible_row_count": [],
        "participation_mismatch": [],
        "points_mismatch": [],
        "team_not_in_games_final": [],
        "unreadable": [],
    }
    comment_counter = Counter()
    rows_total = 0
    absent_total = 0

    for i, path in enumerate(paths, start=1):
        game_id = path.stem

        try:
            frame = read_lossless(path)
        except Exception as exc:
            issues["unreadable"].append(f"{game_id} ({type(exc).__name__})")
            continue

        rows_total += len(frame)

        if not check_structure(game_id, frame, issues):
            continue

        check_participation(game_id, frame, issues)
        check_points_against_games_final(game_id, frame, team_points, issues)

        absent = frame.loc[~has_played(frame["MIN"])]
        absent_total += len(absent)
        comment_counter.update(
            value.strip() if value.strip() else "<empty COMMENT>"
            for value in absent["COMMENT"]
        )

        if i % 2500 == 0:
            print(f"  ...{i:,} / {len(paths):,} files checked")

    print("\n" + "=" * 68)
    print("CHECKS")
    print("=" * 68)
    labels = {
        "unreadable": "files that could not be read",
        "bad_columns": "wrong columns or column order",
        "game_id_mismatch": "GAME_ID inside file != filename",
        "wrong_team_count": f"not exactly {EXPECTED_TEAMS_PER_GAME} distinct TEAM_IDs",
        "duplicate_players": "duplicate (TEAM_ID, PLAYER_ID)",
        "implausible_row_count": f"row count outside {MIN_PLAUSIBLE_ROWS}-{MAX_PLAUSIBLE_ROWS}",
        "participation_mismatch": "MIN and counting stats disagree",
        "team_not_in_games_final": "team missing from games_final",
        "points_mismatch": "PTS sum != games_final final score",
    }
    for key, label in labels.items():
        found = issues[key]
        status = "OK  " if not found else "FAIL"
        print(f"  [{status}] {label:<48} {len(found):,}")
        for example in found[:EXAMPLES_TO_PRINT]:
            print(f"           {example}")
        if len(found) > EXAMPLES_TO_PRINT:
            print(f"           ... and {len(found) - EXAMPLES_TO_PRINT:,} more")

    total_failures = sum(len(v) for v in issues.values()) + len(missing) + len(extra)

    print("\n" + "=" * 68)
    print("ABSENCES BY COMMENT (informational, not a check)")
    print("=" * 68)
    print(f"  {rows_total:,} player-rows total, {absent_total:,} with no minutes "
          f"({absent_total / rows_total * 100:.1f}%)\n")
    for comment, count in comment_counter.most_common():
        print(f"    {count:>7,}  {comment}")

    print("\n" + "=" * 68)
    print("PASS" if total_failures == 0 else f"FAIL - {total_failures:,} issue(s)")
    print("=" * 68)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
