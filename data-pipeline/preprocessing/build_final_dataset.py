"""
Reshape games_final.csv from one-row-per-team-per-game into one-row-per-game,
with home and away features side by side and a single win/loss label.

Input:  data-pipeline/data/processed/games_final.csv
Output: data-pipeline/data/processed/model_dataset.csv

Early-season NaN rows (incomplete rolling windows) are left as-is. Whether
to drop or impute them is a training decision, not a pipeline one.

Features vs labels: anything derived from WL, PTS, REB or AST is a
post-game outcome and must be kept out of the feature matrix, or it leaks
the answer. LABEL_COLUMNS below is the authoritative list.

The ROLL5_/ROLL10_ versions of REB and AST are different: those average
prior games and are valid features. Only the raw single-game values are not.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset.csv"

# Known before tip-off, safe to train on.
PREGAME_FEATURE_COLUMNS = ["REST_DAYS", "IS_BACK_TO_BACK", "TEAM_ELO"]

# Known only after the game. Carried through to build targets, never inputs.
POSTGAME_OUTCOME_COLUMNS = ["WL", "PTS", "REB", "AST"]

# Identifiers: not features, not labels.
ID_COLUMNS = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME"]

# Identical on both team rows, so kept unprefixed instead of HOME_/AWAY_.
GAME_LEVEL_COLUMNS = ["GAME_ID", "GAME_DATE"]

# Labels in the saved dataset. Drop these from X when training.
LABEL_COLUMNS = [
    "HOME_WIN",
    "HOME_PTS",
    "AWAY_PTS",
    "HOME_MARGIN",
    "TOTAL_PTS",
    "HOME_REB",
    "AWAY_REB",
    "REB_MARGIN",
    "TOTAL_REB",
    "HOME_AST",
    "AWAY_AST",
    "AST_MARGIN",
    "TOTAL_AST",
]


def build_keep_columns(df: pd.DataFrame) -> list:
    rolling_cols = [c for c in df.columns if c.startswith("ROLL5_") or c.startswith("ROLL10_")]
    return ID_COLUMNS + POSTGAME_OUTCOME_COLUMNS + rolling_cols + PREGAME_FEATURE_COLUMNS


def split_and_prefix(df: pd.DataFrame, keep_cols: list, prefix: str) -> pd.DataFrame:
    subset = df.loc[df["IS_HOME"] == (prefix == "HOME"), keep_cols]
    rename_map = {c: f"{prefix}_{c}" for c in keep_cols if c not in GAME_LEVEL_COLUMNS}
    return subset.rename(columns=rename_map)


def main():
    df = pd.read_csv(INPUT_PATH)

    keep_cols = build_keep_columns(df)

    home = split_and_prefix(df, keep_cols, "HOME")
    away = split_and_prefix(df, keep_cols, "AWAY")

    merged = home.merge(away, on=GAME_LEVEL_COLUMNS, how="inner", validate="one_to_one")

    # Labels. HOME_PTS/AWAY_PTS stay for debugging and for checking actual
    # totals against over/under lines once odds data exists.
    merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)
    merged["HOME_MARGIN"] = merged["HOME_PTS"] - merged["AWAY_PTS"]
    merged["TOTAL_PTS"] = merged["HOME_PTS"] + merged["AWAY_PTS"]
    merged["REB_MARGIN"] = merged["HOME_REB"] - merged["AWAY_REB"]
    merged["TOTAL_REB"] = merged["HOME_REB"] + merged["AWAY_REB"]
    merged["AST_MARGIN"] = merged["HOME_AST"] - merged["AWAY_AST"]
    merged["TOTAL_AST"] = merged["HOME_AST"] + merged["AWAY_AST"]
    merged = merged.drop(columns=["HOME_WL", "AWAY_WL"])

    merged.to_csv(OUTPUT_PATH, index=False)

    expected_rows = len(df) // 2
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(merged)} (expected {expected_rows})")
    print(
        "Merge check: "
        + ("PASS, no games dropped or duplicated." if len(merged) == expected_rows else "FAIL, mismatch.")
    )

    print(f"\nSaved to {OUTPUT_PATH}")

    print("\nNaN counts (early-season rolling windows still to be decided later):")
    print(merged[["HOME_ROLL5_PTS", "AWAY_ROLL5_PTS"]].isna().sum())

    print(f"\nLabel columns (exclude from training features): {LABEL_COLUMNS}")
    print(merged[LABEL_COLUMNS].describe())

    print("\nPreview:")
    print(merged.head())


if __name__ == "__main__":
    main()
