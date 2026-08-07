"""
Add rest-day features to the feature table.

Input:  data-pipeline/data/processed/games_with_rolling.csv
Output: data-pipeline/data/processed/games_with_features.csv — the main
        working feature file going forward.

REST_DAYS is computed across the full team history (not season-scoped) so
it reflects real calendar gaps.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "games_with_rolling.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "games_with_features.csv"

TEAM_KEY = "TEAM_ID"

# Offseason gaps between a season's last game and the next season's first
# (100+ days) are calendar-accurate but not basketball-meaningful — a team
# isn't more "rested" from 150 days off than from 10. Cap at a week, past
# which extra rest stops being informative.
REST_DAYS_CAP = 7


def main():
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = df.sort_values([TEAM_KEY, "GAME_DATE"]).reset_index(drop=True)

    days_since_prev = df.groupby(TEAM_KEY)["GAME_DATE"].diff().dt.days
    df["REST_DAYS"] = days_since_prev.clip(upper=REST_DAYS_CAP)

    df["IS_BACK_TO_BACK"] = df["REST_DAYS"] == 1

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(df)}")

    print("\nREST_DAYS distribution:")
    print(df["REST_DAYS"].describe())

    print(f"\nBack-to-back games: {int(df['IS_BACK_TO_BACK'].sum())}")

    sample_team_id = df[TEAM_KEY].iloc[0]
    print(f"\nPreview for TEAM_ID {sample_team_id}:")
    print(
        df.loc[df[TEAM_KEY] == sample_team_id, ["GAME_DATE", "REST_DAYS"]]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
