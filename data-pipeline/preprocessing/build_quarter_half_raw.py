"""
Collapse the per-game quarter score files into one team-game table.

  input:  data/raw/quarter_scores/<game_id>.csv   (13,196 files, 2 rows each)
  output: data/processed/quarter_half_raw.csv     (26,392 rows)

Deliberately the thinnest possible step. Everything hard about this domain -
the overtime split, the LineScore `score` defect, the two cross-checked
signals - was settled in validate_quarter_scores.py. All that remains is a
concatenation and one addition, so this script stays boring on purpose and
does not re-litigate any of it.

ONLY PERIODS 1 AND 2 ARE READ, and that is what makes this safe. The whole
overtime problem lives at the end of a game: the four period columns sum to
the regulation-only total, so a full-game reconstruction from them is
impossible. Q1 and 1H are untouched by that - a first quarter is a first
quarter whether the game ends in regulation or triple overtime. PTS_Q3,
PTS_Q4 and FINAL_PTS are read past and dropped.

26,392 ROWS, NOT 26,398. Three games (0022500259/260/261, all 2025-11-19)
could not be fetched: BoxScoreSummaryV3 raises an internal AttributeError on
them, reproducibly, across three separate runs. That is 0.02% of the corpus
and it is handled the way every other legitimately-missing value in this
project is handled - a left join downstream yields NaN for those six
team-game rows, and the existing "some rows lack this feature" machinery
takes it from there. No special-casing, no placeholder, no zero.

GAME_ID IS WRITTEN ZERO-PADDED, matching team_availability.csv and
team_advanced_rolling.csv. The consumer converts to int on the way in, which
is the convention build_final_dataset.py already applies to both of those.

Run:  python data-pipeline/preprocessing/build_quarter_half_raw.py
"""

import sys
from pathlib import Path

import pandas as pd

INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))
from fetch_quarter_scores import (  # noqa: E402
    EXPECTED_ROWS_PER_GAME,
    OUTPUT_DIR as QUARTER_SCORES_DIR,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
GAMES_FINAL_PATH = PROCESSED_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DIR / "quarter_half_raw.csv"

SOURCE_COLUMNS = ["GAME_ID", "TEAM_ID", "PTS_Q1", "PTS_Q2"]
OUTPUT_COLUMNS = ["GAME_ID", "TEAM_ID", "Q1_PTS", "HALF1_PTS"]

# Loose sanity bounds. A team has never scored close to these in a quarter
# or a half; the check exists to catch a column that silently became
# something else, not to model scoring.
MAX_PLAUSIBLE_Q1 = 60
MAX_PLAUSIBLE_HALF1 = 100


def load_all() -> pd.DataFrame:
    """One frame from every per-game file, reading only what is needed."""
    paths = sorted(QUARTER_SCORES_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit(
            f"no files in {QUARTER_SCORES_DIR} - run the ingestion first."
        )

    frames = [
        pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype={"GAME_ID": str})
        for path in paths
    ]
    combined = pd.concat(frames, ignore_index=True)

    print(f"Read {len(paths):,} files -> {len(combined):,} team-game rows.")
    expected = len(paths) * EXPECTED_ROWS_PER_GAME
    if len(combined) != expected:
        raise RuntimeError(
            f"expected {expected:,} rows ({len(paths):,} files x "
            f"{EXPECTED_ROWS_PER_GAME}), got {len(combined):,}. The validator "
            f"passes only when every file has exactly two team rows, so this "
            f"means the files changed after it last ran."
        )
    return combined


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "GAME_ID": raw["GAME_ID"].str.zfill(10),
        "TEAM_ID": raw["TEAM_ID"].astype(int),
        "Q1_PTS": raw["PTS_Q1"].astype(int),
        "HALF1_PTS": (raw["PTS_Q1"] + raw["PTS_Q2"]).astype(int),
    })
    return out[OUTPUT_COLUMNS]


def check(out: pd.DataFrame) -> None:
    """Cheap structural guards. Nothing here should ever fire."""
    problems = []

    nulls = int(out.isna().sum().sum())
    if nulls:
        problems.append(f"{nulls:,} NaN values in the output")

    duplicates = int(out.duplicated(subset=["GAME_ID", "TEAM_ID"]).sum())
    if duplicates:
        problems.append(f"{duplicates:,} duplicate (GAME_ID, TEAM_ID) pairs")

    # A half contains its own first quarter, so this cannot be violated
    # unless the two columns were built from different rows.
    inconsistent = int((out["HALF1_PTS"] < out["Q1_PTS"]).sum())
    if inconsistent:
        problems.append(f"{inconsistent:,} rows where HALF1_PTS < Q1_PTS")

    negative = int(((out["Q1_PTS"] < 0) | (out["HALF1_PTS"] < 0)).sum())
    if negative:
        problems.append(f"{negative:,} rows with a negative score")

    implausible = int(((out["Q1_PTS"] > MAX_PLAUSIBLE_Q1)
                       | (out["HALF1_PTS"] > MAX_PLAUSIBLE_HALF1)).sum())
    if implausible:
        problems.append(f"{implausible:,} rows beyond the plausibility bounds")

    if problems:
        raise RuntimeError("output failed its own checks: " + "; ".join(problems))

    print("Structural checks: PASS "
          "(no NaN, no duplicates, HALF1 >= Q1 everywhere, values in range).")


def report_coverage(out: pd.DataFrame) -> None:
    """Name the gap explicitly rather than letting a row count imply it."""
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"])
    expected_ids = {str(gid).zfill(10) for gid in games["GAME_ID"].unique()}
    covered_ids = set(out["GAME_ID"])
    missing = sorted(expected_ids - covered_ids)

    print(f"\nCoverage: {len(covered_ids):,} of {len(expected_ids):,} games "
          f"({len(covered_ids) / len(expected_ids):.2%}).")
    if missing:
        print(f"  {len(missing)} game(s) absent, "
              f"-> {len(missing) * EXPECTED_ROWS_PER_GAME} team-game rows will be "
              f"NaN after the merge: {missing}")
        print("  Expected. See the module docstring - not special-cased.")


def main():
    raw = load_all()
    out = derive(raw)
    check(out)
    report_coverage(out)

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out):,} rows x {len(out.columns)} columns to {OUTPUT_PATH}")

    print("\nDistributions:")
    print(out[["Q1_PTS", "HALF1_PTS"]].describe().to_string())

    print("\nPreview:")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
