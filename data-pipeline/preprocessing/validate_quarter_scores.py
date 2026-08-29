"""
Read-only quality gate for the quarter scores in
data-pipeline/data/raw/quarter_scores/.

THE CENTREPIECE CHECK IS DIFFERENT HERE, and deliberately so. Every prior
domain could assert a clean identity against an independent source - player
PTS summed to the team's final score, offensiveRating x possessions
reproduced it. The obvious equivalent, "quarters sum to the final", would
be WRONG here: BoxScoreSummaryV3's LineScore carries no overtime data, so
on an overtime game the four periods sum to the regulation-only total while
the true result is higher. That check would fail on roughly 5% of games and
report a structural gap as a data-quality problem.

So the check is split by population, the same way NOT YET SUBMITTED injury
rows and REST_DAYS's legitimate first-game NaNs are handled: strict
equality where it must hold, a bounded plausibility check where a different
shape is expected.

  Regulation games : sum(Q1..Q4) == the final score, exactly.
  Overtime games   : sum(Q1..Q4) <  the final score, by a plausible margin.

  Neither population may ever have sum(Q1..Q4) > the final score. That
  direction has no innocent explanation - overtime only adds points - so it
  is a hard failure rather than a reported statistic.

"THE FINAL SCORE" MEANS games_final.csv, NOT THE LINESCORE'S OWN `score`.
That distinction is not pedantic; a real defect forced it. On game
0021700732 (PHX at HOU, 2018-01-28) the LineScore reports Phoenix's periods
as 24+29+24+25 = 102 while its own `score` field says 103. games_final.csv,
trusted in production for months, says 102 - agreeing with the period
breakdown, not with the total sitting beside it. Classified against `score`
that phantom point reads as a one-point overtime, and the two overtime
signals below then disagree on exactly that one game (711 vs 710).
Classified against games_final.csv it disappears and the signals agree
perfectly, 710 each way.

  So the endpoint's periods are reliable and its `score` is not. FINAL_PTS
  is still ingested and still compared, but as a DIAGNOSTIC rather than a
  gate - see the split between `issues` and `diagnostics` below. Nothing
  downstream reads it: Q1 and 1H need periods 1 and 2, and the full-game
  markets already have games_final.csv.

TWO INDEPENDENT OVERTIME SIGNALS, cross-checked. The shortfall above says a
game went to overtime. So does the raw season files' MIN column: a
regulation team-game records about 240 minutes, one overtime 265, two 290.

  Honest caveat on "independent". games_final.csv and the MIN column both
  derive from the LeagueGameFinder season pulls, so measuring the shortfall
  against games_final.csv does couple the two signals more tightly than
  measuring it against the LineScore's own `score` did. That is a
  deliberate trade, and it is still worth making. What actually decides
  overtime under the shortfall rule is the PERIOD scores, which come wholly
  from BoxScoreSummaryV3 and are untouched by MIN; the two shared inputs
  that remain are different columns measuring different physical quantities
  (points scored vs. time elapsed), so a fault would have to corrupt both
  in the same games to manufacture agreement. Set against one demonstrated
  false positive from the alternative, the coupling is the smaller cost -
  but it is a weaker claim than "fully independent" and is recorded as such.

Note the MIN threshold is 255, not 240. A naive >240 test flags 694 games
in a single season because regulation games record 238-242 from rounding;
the real count at >=255 is about 54 per season, 710 across all eleven. That
calibration was done before this script existed and is why it is not 240.

Does not modify or write anything.

Run:  python data-pipeline/preprocessing/validate_quarter_scores.py
"""

import glob
import sys
from pathlib import Path

import pandas as pd

INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
from fetch_quarter_scores import (  # noqa: E402
    EXPECTED_ROWS_PER_GAME,
    OUTPUT_DIR,
    TARGET_COLUMNS,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
RAW_SEASONS_GLOB = str(DATA_DIR / "raw" / "games_*.csv")

QUARTER_COLUMNS = ["PTS_Q1", "PTS_Q2", "PTS_Q3", "PTS_Q4"]

# See the module docstring: 240 is not usable, 238-242 is regulation noise.
OVERTIME_MIN_THRESHOLD = 255

# One overtime period is five minutes; a team cannot plausibly score more
# than this in the overtimes of a single game. Loose on purpose - it is a
# sanity bound, not a model of scoring.
MAX_PLAUSIBLE_OVERTIME_POINTS = 60

# Roughly 5-7% of NBA games go to overtime.
EXPECTED_OT_SHARE_RANGE = (0.02, 0.12)

EXAMPLES_TO_PRINT = 5
PROGRESS_EVERY = 2500

# Hard failures: each one means a file cannot be trusted as it stands.
ISSUE_LABELS = {
    "unreadable": "files that could not be read",
    "column_mismatch": "columns differ from the canonical set",
    "wrong_row_count": f"not exactly {EXPECTED_ROWS_PER_GAME} team rows",
    "game_id_mismatch": "GAME_ID inside file != filename",
    "duplicate_team": "the two rows share a TEAM_ID",
    "unparseable": "quarter or final score not numeric",
    "quarters_exceed_final": "quarters sum to MORE than games_final PTS",
    "implausible_overtime": f"shortfall > {MAX_PLAUSIBLE_OVERTIME_POINTS} points",
    "team_not_in_games_final": "team missing from games_final (cannot classify)",
}

# Reported, but not fatal. FINAL_PTS is ingested and is no longer
# authoritative for anything - see the module docstring on 0021700732. A
# disagreement is a fact about the endpoint, not a defect in what this
# pipeline consumes, so it must stay VISIBLE without failing a run.
# Downgrading it silently would be the mistake; naming it as a diagnostic
# and printing it every time is the point.
DIAGNOSTIC_LABELS = {
    "final_disagrees": "LineScore FINAL_PTS != games_final PTS",
}


def load_expected_game_ids() -> set:
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"])
    return {str(gid).zfill(10) for gid in games["GAME_ID"].unique()}


def load_final_points() -> dict:
    """(game_id, team_id) -> final PTS, from the already-trusted table.

    This is the reference the regulation/overtime split is measured
    against, not merely a cross-check. See the module docstring.
    """
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID", "TEAM_ID", "PTS"])
    return {
        (str(row.GAME_ID).zfill(10), int(row.TEAM_ID)): float(row.PTS)
        for row in games.itertuples()
    }


def load_overtime_by_minutes() -> set:
    """Game ids that went to overtime, judged only by elapsed minutes.

    Independent of the period scores: this reads the original season pulls,
    which carry a per-team MIN column the pipeline otherwise drops.
    """
    overtime = set()
    for path in glob.glob(RAW_SEASONS_GLOB):
        season = pd.read_csv(path, usecols=["GAME_ID", "MIN"])
        hits = season.loc[season["MIN"] >= OVERTIME_MIN_THRESHOLD, "GAME_ID"]
        overtime |= {str(gid).zfill(10) for gid in hits}
    return overtime


def check_structure(game_id, frame, issues) -> bool:
    if list(frame.columns) != TARGET_COLUMNS:
        issues["column_mismatch"].append(game_id)
        return False
    if len(frame) != EXPECTED_ROWS_PER_GAME:
        issues["wrong_row_count"].append(f"{game_id} ({len(frame)} rows)")
        return False
    if not (frame["GAME_ID"].astype(str).str.zfill(10) == game_id).all():
        issues["game_id_mismatch"].append(game_id)
    if frame["TEAM_ID"].nunique() != EXPECTED_ROWS_PER_GAME:
        issues["duplicate_team"].append(game_id)
    return True


def check_quarters(game_id, frame, final_points, issues, diagnostics, gaps) -> bool:
    """Returns True if this game looks like overtime by the score shortfall.

    The shortfall is measured against games_final.csv, so a team with no row
    there cannot be classified at all - it is recorded and skipped rather
    than quietly falling back to the LineScore's own total, which is the
    field this check exists to stop depending on.
    """
    is_overtime = False

    for row in frame.itertuples():
        team_id = int(row.TEAM_ID)
        quarters = [getattr(row, c) for c in QUARTER_COLUMNS]

        if any(pd.isna(q) for q in quarters) or pd.isna(row.FINAL_PTS):
            issues["unparseable"].append(f"{game_id} team {team_id}")
            continue

        actual = final_points.get((game_id, team_id))
        if actual is None:
            issues["team_not_in_games_final"].append(f"{game_id} team {team_id}")
            continue

        summed = float(sum(quarters))
        gap = actual - summed

        if gap < 0:
            # No innocent explanation: overtime only adds points.
            issues["quarters_exceed_final"].append(
                f"{game_id} team {team_id}: quarters {summed:.0f} > "
                f"games_final {actual:.0f}"
            )
        elif gap > 0:
            is_overtime = True
            gaps.append(gap)
            if gap > MAX_PLAUSIBLE_OVERTIME_POINTS:
                issues["implausible_overtime"].append(
                    f"{game_id} team {team_id}: {gap:.0f} points beyond regulation"
                )

        # Diagnostic only. The endpoint's own total is known to be wrong on
        # at least one game, so it decides nothing here - but it is still
        # worth knowing when and how far it drifts.
        reported = float(row.FINAL_PTS)
        if abs(reported - actual) > 1e-9:
            diagnostics["final_disagrees"].append(
                f"{game_id} team {team_id}: quarters {summed:.0f}, "
                f"LineScore {reported:.0f} vs games_final {actual:.0f}"
            )

    return is_overtime


def report(title, labels, found_map, flag="FAIL"):
    """Print one block. `flag` is the marker for a non-empty list.

    Diagnostics pass "NOTE": printing [FAIL] beside a line the run does not
    fail on is exactly the kind of contradiction that teaches a reader to
    stop trusting the output.
    """
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, label in labels.items():
        found = found_map[key]
        print(f"  [{'OK  ' if not found else flag}] {label:<48} {len(found):,}")
        for example in found[:EXAMPLES_TO_PRINT]:
            print(f"           {example}")
        if len(found) > EXAMPLES_TO_PRINT:
            print(f"           ... and {len(found) - EXAMPLES_TO_PRINT:,} more")


def main():
    if not OUTPUT_DIR.exists():
        raise SystemExit(f"{OUTPUT_DIR} does not exist - run the ingestion first.")

    expected_ids = load_expected_game_ids()
    final_points = load_final_points()

    paths = sorted(OUTPUT_DIR.glob("*.csv"))
    found_ids = {p.stem for p in paths}

    print(f"Validating {len(paths):,} files in {OUTPUT_DIR}")
    print(f"games_final.csv lists {len(expected_ids):,} unique games.\n")

    print("=" * 72)
    print("COVERAGE")
    print("=" * 72)
    missing, extra = expected_ids - found_ids, found_ids - expected_ids
    print(f"  files found : {len(found_ids):,}")
    print(f"  missing     : {len(missing):,}")
    print(f"  extra       : {len(extra):,}")
    for label, group in (("missing", missing), ("extra", extra)):
        if group:
            print(f"    {label}: {sorted(group)[:EXAMPLES_TO_PRINT]}")

    issues = {key: [] for key in ISSUE_LABELS}
    diagnostics = {key: [] for key in DIAGNOSTIC_LABELS}
    gaps = []
    overtime_by_score = set()

    for i, path in enumerate(paths, start=1):
        game_id = path.stem
        try:
            frame = pd.read_csv(path, dtype={"GAME_ID": str})
        except Exception as exc:
            issues["unreadable"].append(f"{game_id} ({type(exc).__name__})")
            continue

        if not check_structure(game_id, frame, issues):
            continue

        if check_quarters(game_id, frame, final_points, issues, diagnostics, gaps):
            overtime_by_score.add(game_id)

        if i % PROGRESS_EVERY == 0:
            print(f"  ...{i:,} / {len(paths):,} files checked")

    report("CHECKS (any failure here blocks the run)", ISSUE_LABELS, issues)
    report("DIAGNOSTICS (reported, deliberately not fatal - see docstring)",
           DIAGNOSTIC_LABELS, diagnostics, flag="NOTE")

    print("\n" + "=" * 72)
    print("REGULATION vs OVERTIME (the split the sum check needs)")
    print("=" * 72)
    checked = len(paths) - len(issues["unreadable"]) - len(issues["column_mismatch"])
    regulation = checked - len(overtime_by_score)
    share = len(overtime_by_score) / checked if checked else 0

    print(f"  games checked       : {checked:,}")
    print(f"  regulation (exact)  : {regulation:,}  ({regulation / checked:.1%})")
    print(f"  overtime (shortfall): {len(overtime_by_score):,}  ({share:.1%})")
    print(f"  measured against    : games_final.csv PTS")

    low, high = EXPECTED_OT_SHARE_RANGE
    print(f"  expected OT share   : {low:.0%}-{high:.0%}  -> "
          f"{'in range' if low <= share <= high else 'OUTSIDE RANGE - investigate'}")

    if gaps:
        gap_series = pd.Series(gaps)
        print(f"\n  overtime shortfall per team-game (points beyond regulation):")
        print(f"    min {gap_series.min():.0f}   median {gap_series.median():.0f}   "
              f"max {gap_series.max():.0f}")
        print(f"    all strictly positive: {bool((gap_series > 0).all())}")

    print("\n" + "=" * 72)
    print("TWO INDEPENDENT OVERTIME SIGNALS - do they agree?")
    print("=" * 72)
    overtime_by_minutes = load_overtime_by_minutes() & found_ids
    only_score = overtime_by_score - overtime_by_minutes
    only_minutes = overtime_by_minutes - overtime_by_score
    both = overtime_by_score & overtime_by_minutes

    print(f"  by score shortfall  : {len(overtime_by_score):,}")
    print(f"  by MIN >= {OVERTIME_MIN_THRESHOLD}      : {len(overtime_by_minutes):,}")
    print(f"  agreed by both      : {len(both):,}")
    print(f"  score only          : {len(only_score):,}")
    print(f"  minutes only        : {len(only_minutes):,}")

    if not only_score and not only_minutes:
        print("\n  Perfect agreement between two independently-derived signals.")
    else:
        print("\n  DISAGREEMENT - worth investigating before trusting either:")
        for game_id in sorted(only_score)[:EXAMPLES_TO_PRINT]:
            print(f"    shortfall but not minutes: {game_id}")
        for game_id in sorted(only_minutes)[:EXAMPLES_TO_PRINT]:
            print(f"    minutes but not shortfall: {game_id}")

    total_failures = sum(len(v) for v in issues.values()) + len(missing) + len(extra)
    total_diagnostics = sum(len(v) for v in diagnostics.values())

    print("\n" + "=" * 72)
    print("PASS" if total_failures == 0 else f"FAIL - {total_failures:,} issue(s)")
    if total_diagnostics:
        print(f"({total_diagnostics:,} diagnostic(s) reported above, not counted)")
    print("=" * 72)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
