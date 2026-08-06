"""
Build the canonical games table from the raw per-season CSVs.

Concatenates all seasons, derives home/away (from MATCHUP) and opponent
(from GAME_ID pairing) columns, drops non-predictive columns, and sorts
chronologically per team so downstream feature scripts (rolling averages,
rest days, etc.) can rely on ordering. Games where IS_HOME can't be
trusted (source data doesn't give exactly one home team per GAME_ID) are
dropped and reported. Output: data-pipeline/data/processed/games_master.csv
"""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "games_master.csv"

# TEAM_ID is the stable key (abbreviations can change on relocation), so
# it's the canonical team identifier; TEAM_ABBREVIATION is dropped in
# favor of it. TEAM_NAME is kept purely for human readability.
COLUMNS_TO_DROP = ["SEASON_ID", "MIN", "TEAM_ABBREVIATION", "MATCHUP"]

TEAM_KEY = "TEAM_ID"


def load_all_seasons():
    season_files = sorted(RAW_DATA_DIR.glob("games_*.csv"))
    frames = [pd.read_csv(f) for f in season_files]
    return pd.concat(frames, ignore_index=True)


def derive_opponent(df: pd.DataFrame) -> pd.Series:
    """Opponent abbreviation, taken from the other row sharing the same
    GAME_ID rather than parsed out of MATCHUP text — robust to whatever
    formatting quirks MATCHUP might have."""

    def other_team(group: pd.DataFrame) -> pd.Series:
        if len(group) != 2:
            return pd.Series(pd.NA, index=group.index)
        return pd.Series(group["TEAM_ABBREVIATION"].values[::-1], index=group.index)

    return df.groupby("GAME_ID", group_keys=False).apply(other_team)


def drop_unreliable_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Every GAME_ID should have exactly one home team. Where that's not
    true, the source data can't be trusted for IS_HOME — drop those games
    rather than guess, and document exactly what was dropped."""

    home_counts = df.groupby("GAME_ID")["IS_HOME"].sum()
    bad_game_ids = home_counts[home_counts != 1].index

    if len(bad_game_ids) == 0:
        print("IS_HOME check: every GAME_ID has exactly one home team. Nothing dropped.")
        return df

    dropped = df[df["GAME_ID"].isin(bad_game_ids)]
    print(
        f"IS_HOME check: dropping {len(bad_game_ids)} games ({len(dropped)} rows) "
        f"with an unreliable home/away signal (not exactly one home team per GAME_ID):"
    )
    print(
        dropped[["GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "IS_HOME"]]
        .sort_values(["GAME_ID", "GAME_DATE"])
        .to_string(index=False)
    )

    return df[~df["GAME_ID"].isin(bad_game_ids)]


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_seasons()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    df["OPPONENT"] = derive_opponent(df)

    df = drop_unreliable_home_away(df)

    df = df.drop(columns=COLUMNS_TO_DROP)

    df = df.sort_values([TEAM_KEY, "GAME_DATE"], ascending=True).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")
    print(f"\nSaved to {OUTPUT_PATH}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
