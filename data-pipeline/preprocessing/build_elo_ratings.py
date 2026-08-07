"""
Add team Elo ratings to the feature table.

Input:  data-pipeline/data/processed/games_with_features.csv
Output: data-pipeline/data/processed/games_final.csv — the model-ready table.

Elo is shared, sequential state (each game updates both teams' ratings),
unlike the per-team rolling/rest features built so far. Games are
processed in true league-wide chronological order with a plain loop —
slower than a vectorized pandas op, but the update logic is inherently
sequential and correctness matters more than speed here.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "games_with_features.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "games_final.csv"

TEAM_KEY = "TEAM_ID"

BASELINE_RATING = 1500.0
K_FACTOR = 20

# Roster turnover over the offseason means a team's rating shouldn't carry
# over fully intact. Move each team 1/3 of the way back toward baseline at
# the start of a new season.
SEASON_REGRESSION_FRACTION = 1 / 3


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    ratings = {}
    last_season = {}

    team_elo = pd.Series(index=df.index, dtype=float)
    opponent_elo = pd.Series(index=df.index, dtype=float)

    def rating_for(team_id, season) -> float:
        if team_id not in ratings:
            ratings[team_id] = BASELINE_RATING
        elif last_season[team_id] != season:
            ratings[team_id] = BASELINE_RATING + (ratings[team_id] - BASELINE_RATING) * (
                1 - SEASON_REGRESSION_FRACTION
            )
        last_season[team_id] = season
        return ratings[team_id]

    for _, group in df.groupby("GAME_ID", sort=False):
        idx_a, idx_b = group.index[0], group.index[1]
        row_a, row_b = df.loc[idx_a], df.loc[idx_b]
        season = row_a["SEASON"]

        rating_a = rating_for(row_a[TEAM_KEY], season)
        rating_b = rating_for(row_b[TEAM_KEY], season)

        expected_a = expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a

        actual_a = 1.0 if row_a["WL"] == "W" else 0.0
        actual_b = 1.0 - actual_a

        team_elo[idx_a], opponent_elo[idx_a] = rating_a, rating_b
        team_elo[idx_b], opponent_elo[idx_b] = rating_b, rating_a

        ratings[row_a[TEAM_KEY]] = rating_a + K_FACTOR * (actual_a - expected_a)
        ratings[row_b[TEAM_KEY]] = rating_b + K_FACTOR * (actual_b - expected_b)

    df["TEAM_ELO"] = team_elo
    df["OPPONENT_ELO"] = opponent_elo
    return df, ratings


def main():
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    df, final_ratings = compute_elo(df)

    df = df.sort_values([TEAM_KEY, "GAME_DATE"]).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Saved to {OUTPUT_PATH}")

    print(
        f"\nFinal Elo range across {len(final_ratings)} teams: "
        f"{min(final_ratings.values()):.1f} to {max(final_ratings.values()):.1f}"
    )

    sample_team_id = df[TEAM_KEY].iloc[0]
    print(f"\nTEAM_ELO progression for TEAM_ID {sample_team_id}:")
    print(
        df.loc[df[TEAM_KEY] == sample_team_id, ["GAME_DATE", "TEAM_ELO"]]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
