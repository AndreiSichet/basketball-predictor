"""
Pull raw game data from nba_api's LeagueGameFinder, one season at a time,
and save each season to its own untouched CSV in data-pipeline/data/raw/.

No cleaning/filtering/transforming here — that belongs in a later script.
"""

import time
from pathlib import Path

from nba_api.stats.endpoints import leaguegamefinder

SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

DELAY_SECONDS = 1
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch_season(season: str):
    # Regular season only: playoff/preseason/All-Star games behave
    # differently (rest patterns, elimination pressure, exhibition rosters)
    # and would just add noise to a regular-season prediction model.
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, season_type_nullable="Regular Season"
    )
    return finder.get_data_frames()[0]


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = []

    for season in SEASONS:
        print(f"Fetching season {season}...")

        try:
            games_df = fetch_season(season)
        except Exception as exc:
            print(f"  FAILED: {season} -> {exc}")
            failures.append(season)
            time.sleep(DELAY_SECONDS)
            continue

        out_path = RAW_DATA_DIR / f"games_{season}.csv"
        games_df.to_csv(out_path, index=False)

        print(f"  OK: {len(games_df)} rows -> {out_path.name}")
        successes += 1

        time.sleep(DELAY_SECONDS)

    print(f"\nDone. {successes}/{len(SEASONS)} seasons saved.")
    if failures:
        print(f"Failed seasons: {', '.join(failures)}")


if __name__ == "__main__":
    main()
