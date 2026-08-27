"""
Pull per-game team-level advanced stats from BoxScoreAdvancedV3, one file
per game, into data-pipeline/data/raw/team_advanced_stats/.

Feeds the pace/possession-adjusted rolling features (roadmap item 2). Same
architecture as fetch_player_boxscores.py - shared rate limiter, resume by
file existence, atomic write-then-rename, retries with linear backoff - and
that module's RateLimiter is imported rather than reimplemented.

SIMPLER THAN THE PLAYER BOX SCORES, in three ways worth knowing:

  * **V3 only, no fallback.** BoxScoreAdvancedV2 is dead for 2025-26 the
    same way Traditional V2 was, and V3's 30-column TeamStats schema is
    identical - same columns, same order - in both 2015-16 and 2025-26.
    So unlike the traditional box scores, where V2 and V3 files coexisted
    and had to be merged into a canonical schema (the NICKNAME /
    START_POSITION / POSITION reconciliation), V3 alone covers all 13,199
    games cleanly. There is no per-era handling here at all.
  * **TeamStats only.** This feature needs team-level rating and pace, not
    player-level advanced rows. Saving PlayerStats too would mean 27-37
    extra rows per game for data nothing reads.
  * **An empty frame is unambiguously a failure.** Every real game has
    exactly two teams. Unlike a player frame, where "empty" could in
    principle be a strange-but-real game, two rows is a hard invariant, so
    anything else is logged and not written.

THE estimated* COLUMNS ARE KNOWN-UNRELIABLE AND DELIBERATELY UNUSED.
estimatedOffensiveRating came back at 576.2 against a real offensiveRating
of 115.2 in the same row, and estimatedPace at 20.9 against a real pace of
104.5 - both roughly 5x out. They use a different internal formula and are
not a "more precise" version of the real columns. The five metrics this
feature will use are all the plain, non-estimated ones: offensiveRating,
defensiveRating, netRating, pace, trueShootingPercentage. They are saved
anyway, since the whole frame is stored raw, but do not reach for them.

SCHEMA IS NOT HARDCODED HERE. nba_api 1.11.4 declares 29 TeamStats columns
while the live response carries 30, so the declared shape lags reality.
Whatever the endpoint returns is written as-is; validate_team_advanced_stats
.py then checks that every file agrees with every other, which tests the
real invariant (consistency) rather than a number guessed in advance.

SCALE: one call per game, 13,199 games. Hours, not minutes. Run it in the
background; it is safe to interrupt and rerun.
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoreadvancedv3

# The rate limiter is imported, not reimplemented: one definition of "how
# hard do we hit this API" for the whole project.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_player_boxscores import RateLimiter  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_DIR = DATA_DIR / "raw" / "team_advanced_stats"
# Kept out of OUTPUT_DIR so that directory stays purely <game_id>.csv files.
FAILURES_LOG_PATH = DATA_DIR / "raw" / "team_advanced_stats_failures.log"

# Same global ceiling as the player box-score pull. Workers overlap network
# latency; they do not raise the request rate. Only REQUESTS_PER_SECOND
# does that.
REQUESTS_PER_SECOND = 1.0
MAX_WORKERS = 6

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30

PROGRESS_EVERY = 100

# Every game has exactly two teams. Not a heuristic.
EXPECTED_ROWS_PER_GAME = 2

# Checked on arrival so a silent upstream schema change fails loudly here
# rather than in feature engineering. Deliberately only what this feature
# depends on, not the full column list - see the module docstring.
REQUIRED_COLUMNS = (
    "gameId",
    "teamId",
    "offensiveRating",
    "defensiveRating",
    "netRating",
    "pace",
    "trueShootingPercentage",
    "possessions",
)

rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
stop_requested = threading.Event()
print_lock = threading.Lock()
failures_lock = threading.Lock()


def load_game_ids() -> list:
    """Every game id in the history table, zero-padded to 10 characters.

    The padding is not cosmetic: the endpoint wants the full 10-digit
    string, and str(21500001) drops the leading zeros, which it answers
    with an empty result rather than an error.
    """
    games = pd.read_csv(GAMES_FINAL_PATH, usecols=["GAME_ID"])
    return sorted(str(game_id).zfill(10) for game_id in games["GAME_ID"].unique())


def record_failure(game_id: str, reason: str):
    with failures_lock:
        with FAILURES_LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(f"{game_id}\t{reason}\n")


def fetch_one(game_id: str) -> str:
    """Fetch and save one game. Returns "skipped", "fetched" or "failed"."""
    out_path = OUTPUT_DIR / f"{game_id}.csv"
    if out_path.exists():
        return "skipped"

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if stop_requested.is_set():
            return "failed"

        rate_limiter.acquire()

        try:
            box_score = boxscoreadvancedv3.BoxScoreAdvancedV3(
                game_id=game_id, timeout=REQUEST_TIMEOUT_SECONDS
            )
            # By name, not get_data_frames()[0]: this endpoint also returns
            # PlayerStats, and positional access would break silently if
            # that ordering ever changed.
            teams = box_score.team_stats.get_data_frame()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            problem = validate_frame(teams)
            if problem:
                # Never write a bad file: it would satisfy the resume check
                # forever and bake a hole into the dataset.
                last_error = problem
            else:
                # Write-then-rename, so an interrupted write cannot leave a
                # truncated CSV a later run would skip as done.
                temp_path = out_path.with_suffix(".csv.partial")
                teams.to_csv(temp_path, index=False, encoding="utf-8")
                temp_path.replace(out_path)
                return "fetched"

        if attempt < MAX_ATTEMPTS and not stop_requested.is_set():
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    record_failure(game_id, last_error or "unknown error")
    return "failed"


def validate_frame(teams: pd.DataFrame) -> str:
    """Reasons this frame must not be written, or "" if it is fine."""
    if teams.empty:
        return "empty TeamStats frame"
    if len(teams) != EXPECTED_ROWS_PER_GAME:
        return f"expected {EXPECTED_ROWS_PER_GAME} team rows, got {len(teams)}"

    missing = [c for c in REQUIRED_COLUMNS if c not in teams.columns]
    if missing:
        return f"missing expected columns: {missing}"

    return ""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    game_ids = load_game_ids()
    already_present = sum(1 for gid in game_ids if (OUTPUT_DIR / f"{gid}.csv").exists())
    remaining = len(game_ids) - already_present

    print(f"{len(game_ids):,} games in {GAMES_FINAL_PATH.name}.")
    print(f"{already_present:,} already downloaded, {remaining:,} to fetch via V3.")
    print(
        f"Rate limit {REQUESTS_PER_SECOND}/s across {MAX_WORKERS} workers "
        f"-> roughly {remaining / REQUESTS_PER_SECOND / 3600:.1f}h if nothing fails.\n"
    )

    counts = {"skipped": 0, "fetched": 0, "failed": 0}
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_one, gid): gid for gid in game_ids}

        try:
            for future in as_completed(futures):
                counts[future.result()] += 1
                processed += 1

                if processed % PROGRESS_EVERY == 0:
                    with print_lock:
                        print(
                            f"{processed:>6,} / {len(game_ids):,}  "
                            f"(fetched {counts['fetched']:,}, "
                            f"skipped {counts['skipped']:,}, "
                            f"failed {counts['failed']:,})"
                        )
        except KeyboardInterrupt:
            print("\nInterrupted - finishing in-flight requests, then stopping.")
            print("Rerun this script to continue from where it left off.")
            stop_requested.set()
            for future in futures:
                future.cancel()
            raise SystemExit(1)

    print(
        f"\nDone. {processed:,} games processed: "
        f"{counts['fetched']:,} fetched, "
        f"{counts['skipped']:,} already present, "
        f"{counts['failed']:,} failed."
    )

    if counts["failed"]:
        print(f"Failed game ids logged to {FAILURES_LOG_PATH}.")
        print("Rerunning this script retries only those, since the rest are on disk.")


if __name__ == "__main__":
    main()
