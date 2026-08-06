"""
Read-only quality gate for the raw game data in data-pipeline/data/raw/.

Loads every games_<season>.csv, checks it against a handful of invariants,
and prints a per-season summary plus a final overall pass/fail line. Does
not modify or write anything.
"""

import re
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

CRITICAL_COLUMNS = ["TEAM_ID", "GAME_ID", "GAME_DATE", "WL", "PTS"]
VALID_WL_VALUES = {"W", "L"}

SEASON_FILE_RE = re.compile(r"games_(\d{4}-\d{2})\.csv$")


def find_season_files():
    files = sorted(RAW_DATA_DIR.glob("games_*.csv"))
    seasons = []
    for f in files:
        match = SEASON_FILE_RE.search(f.name)
        if match:
            seasons.append((match.group(1), f))
    return seasons


def season_date_bounds(season: str):
    start_year = int(season[:4])
    end_year = start_year + 1
    lower = pd.Timestamp(f"{start_year}-09-01")
    upper = pd.Timestamp(f"{end_year}-08-31")
    return lower, upper


def validate_file(season: str, path: Path, reference_columns):
    issues = []
    df = pd.read_csv(path)

    row_count = len(df)
    col_count = len(df.columns)

    schema_ok = True
    if reference_columns is not None and list(df.columns) != list(reference_columns):
        schema_ok = False
        missing = set(reference_columns) - set(df.columns)
        extra = set(df.columns) - set(reference_columns)
        issues.append(
            f"schema mismatch (missing={sorted(missing)}, extra={sorted(extra)})"
        )

    missing_counts = {}
    for col in CRITICAL_COLUMNS:
        if col in df.columns:
            missing_counts[col] = int(df[col].isna().sum())
            if missing_counts[col] > 0:
                issues.append(f"{missing_counts[col]} missing values in {col}")
        else:
            missing_counts[col] = None
            issues.append(f"critical column {col} not present")

    date_parse_failures = 0
    date_out_of_range = 0
    if "GAME_DATE" in df.columns:
        parsed_dates = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        date_parse_failures = int(parsed_dates.isna().sum() - df["GAME_DATE"].isna().sum())
        lower, upper = season_date_bounds(season)
        in_range = parsed_dates.between(lower, upper)
        date_out_of_range = int((~in_range & parsed_dates.notna()).sum())
        if date_parse_failures > 0:
            issues.append(f"{date_parse_failures} GAME_DATE values failed to parse")
        if date_out_of_range > 0:
            issues.append(f"{date_out_of_range} GAME_DATE values outside expected season range")

    invalid_wl_count = 0
    if "WL" in df.columns:
        invalid_wl = df.loc[df["WL"].notna() & ~df["WL"].isin(VALID_WL_VALUES)]
        invalid_wl_count = len(invalid_wl)
        if invalid_wl_count > 0:
            issues.append(f"{invalid_wl_count} WL values outside {{'W', 'L'}}")

    bad_game_id_count = 0
    if "GAME_ID" in df.columns:
        game_id_counts = df["GAME_ID"].value_counts()
        bad_game_ids = game_id_counts[game_id_counts != 2]
        bad_game_id_count = len(bad_game_ids)
        if bad_game_id_count > 0:
            issues.append(f"{bad_game_id_count} GAME_IDs do not appear exactly twice")

    duplicate_row_count = int(df.duplicated().sum())
    if duplicate_row_count > 0:
        issues.append(f"{duplicate_row_count} exact duplicate rows")

    passed = len(issues) == 0

    print(f"\nSeason {season} ({path.name})")
    print(f"  rows: {row_count}, columns: {col_count}, schema_ok: {schema_ok}")
    print(f"  missing values (critical columns): {missing_counts}")
    print(f"  GAME_DATE parse failures: {date_parse_failures}, out-of-range: {date_out_of_range}")
    print(f"  invalid WL values: {invalid_wl_count}")
    print(f"  GAME_IDs not paired exactly twice: {bad_game_id_count}")
    print(f"  exact duplicate rows: {duplicate_row_count}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}" + (f" ({'; '.join(issues)})" if issues else ""))

    return passed, list(df.columns)


def main():
    season_files = find_season_files()

    if not season_files:
        print(f"No games_<season>.csv files found in {RAW_DATA_DIR}")
        return

    reference_columns = None
    results = []

    for season, path in season_files:
        passed, columns = validate_file(season, path, reference_columns)
        if reference_columns is None:
            reference_columns = columns
        results.append((season, passed))

    print("\n" + "=" * 40)
    failed = [season for season, passed in results if not passed]
    if failed:
        print(f"{len(results) - len(failed)}/{len(results)} seasons passed validation.")
        print(f"Failed seasons: {', '.join(failed)}")
    else:
        print(f"All {len(results)} seasons passed validation.")


if __name__ == "__main__":
    main()
