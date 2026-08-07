"""
Reshape games_final.csv from one-row-per-team-per-game into one-row-per-game,
with home and away features side by side and a single win/loss label.

Input:  data-pipeline/data/processed/games_final.csv
Output: data-pipeline/data/processed/model_dataset.csv — the model-ready table.

Early-season NaN rows (incomplete rolling windows) are left as-is; whether
to drop/impute them belongs to the training phase, not here.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "games_final.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset.csv"


def build_keep_columns(df: pd.DataFrame) -> list:
    rolling_cols = [c for c in df.columns if c.startswith("ROLL5_") or c.startswith("ROLL10_")]
    return ["GAME_ID", "TEAM_ID", "TEAM_NAME", "WL"] + rolling_cols + [
        "REST_DAYS",
        "IS_BACK_TO_BACK",
        "TEAM_ELO",
    ]


def split_and_prefix(df: pd.DataFrame, keep_cols: list, prefix: str) -> pd.DataFrame:
    subset = df.loc[df["IS_HOME"] == (prefix == "HOME"), keep_cols]
    rename_map = {c: f"{prefix}_{c}" for c in keep_cols if c != "GAME_ID"}
    return subset.rename(columns=rename_map)


def main():
    df = pd.read_csv(INPUT_PATH)

    keep_cols = build_keep_columns(df)

    home = split_and_prefix(df, keep_cols, "HOME")
    away = split_and_prefix(df, keep_cols, "AWAY")

    merged = home.merge(away, on="GAME_ID", how="inner", validate="one_to_one")

    merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)
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

    print("\nPreview:")
    print(merged.head())


if __name__ == "__main__":
    main()
