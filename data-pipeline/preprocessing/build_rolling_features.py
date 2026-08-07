"""
Add trailing rolling-window features to the canonical games table.

Input:  data-pipeline/data/processed/games_master.csv
Output: data-pipeline/data/processed/games_with_rolling.csv

Rolling windows are computed per team-season (so they reset at season
boundaries, not blended across the offseason) and use shift(1) before
rolling so a game's features only ever reflect games strictly before it.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "games_master.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "games_with_rolling.csv"

TEAM_KEY = "TEAM_ID"
WINDOWS = [5, 10]

# source column -> feature name used in ROLL{n}_{name} columns
METRICS = {
    "WIN": "WIN_PCT",
    "PTS": "PTS",
    "PLUS_MINUS": "PLUS_MINUS",
    "FG_PCT": "FG_PCT",
    "REB": "REB",
    "AST": "AST",
    "TOV": "TOV",
}


def derive_season(game_date: pd.Series) -> pd.Series:
    return game_date.dt.year.where(game_date.dt.month >= 8, game_date.dt.year - 1)


def main():
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df["SEASON"] = derive_season(df["GAME_DATE"])
    df["WIN"] = (df["WL"] == "W").astype(int)

    df = df.sort_values([TEAM_KEY, "SEASON", "GAME_DATE"]).reset_index(drop=True)

    grouped = df.groupby([TEAM_KEY, "SEASON"])
    for window in WINDOWS:
        for source_col, feature_name in METRICS.items():
            df[f"ROLL{window}_{feature_name}"] = grouped[source_col].transform(
                lambda s, w=window: s.shift(1).rolling(w).mean()
            )

    df = df.drop(columns=["WIN"])

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Saved to {OUTPUT_PATH}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
