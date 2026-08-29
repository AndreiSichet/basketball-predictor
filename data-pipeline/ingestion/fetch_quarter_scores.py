"""
Pull per-game quarter scoring from BoxScoreSummaryV3's LineScore, one file
per game, into data-pipeline/data/raw/quarter_scores/.

Feeds the Q1 and 1H markets (roadmap item 4). Same architecture as the last
two per-game pulls - shared RateLimiter, resume by file existence, atomic
write-then-rename, retries with linear backoff - and that module's
RateLimiter is imported rather than reimplemented.

V3 ONLY, no V2 fallback. BoxScoreSummaryV2 carries the same
broken-after-April-2025 warning its Traditional and Advanced siblings did.
Building a dual path for an endpoint already on its way out is work this
project has now paid for twice.

OVERTIME IS NOT IN THIS RESPONSE, and that is confirmed rather than
assumed. A four-game investigation - regulation and overtime, from both
2015-16 and 2025-26 - found the same four period columns every time: no
period5Score, and no overtime points folded into period4Score. On an
overtime game the four periods sum to the regulation-only total while
`score` carries the true final, so the two genuinely disagree.

That does not affect what is being built: Q1 and 1H need periods 1 and 2,
which are complete regardless of what happens later in a game. But it does
mean FINAL_PTS is worth keeping even though no market uses it directly -
`FINAL_PTS - (Q1+Q2+Q3+Q4) > 0` becomes an independent way to identify
overtime games, cross-checkable against the raw season files' MIN column.
validate_quarter_scores.py does exactly that.

SCHEMA IS VERIFIED ON ARRIVAL. The 13-column LineScore shape was confirmed
identical across both eras, so a response that does not match it is
rejected rather than written - the same guard the advanced-stats pull uses.
A file is never written for a bad response, because an existing file is
what the resume check trusts.

SCALE: one call per game, 13,199 games. Hours, not minutes. Safe to
interrupt and rerun.
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoresummaryv3

# One definition of how hard this project hits the API.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_player_boxscores import RateLimiter  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_DIR = DATA_DIR / "raw" / "quarter_scores"
# Kept out of OUTPUT_DIR so that directory stays purely <game_id>.csv files.
FAILURES_LOG_PATH = DATA_DIR / "raw" / "quarter_scores_failures.log"

REQUESTS_PER_SECOND = 1.0
MAX_WORKERS = 6

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30

PROGRESS_EVERY = 100

EXPECTED_ROWS_PER_GAME = 2

# The exact LineScore shape confirmed on four games across both eras. A
# response that differs is rejected, not written.
EXPECTED_SOURCE_COLUMNS = [
    "gameId", "teamId", "teamCity", "teamName", "teamTricode", "teamSlug",
    "teamWins", "teamLosses",
    "period1Score", "period2Score", "period3Score", "period4Score", "score",
]

# Renamed into this project's existing conventions rather than left in the
# endpoint's camelCase, exactly as the traditional box scores are.
COLUMN_RENAMES = {
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamTricode": "TEAM_ABBREVIATION",
    "period1Score": "PTS_Q1",
    "period2Score": "PTS_Q2",
    "period3Score": "PTS_Q3",
    "period4Score": "PTS_Q4",
    "score": "FINAL_PTS",
}

TARGET_COLUMNS = [
    "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION",
    "PTS_Q1", "PTS_Q2", "PTS_Q3", "PTS_Q4",
    "FINAL_PTS",
]

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


def validate_frame(line_score: pd.DataFrame) -> str:
    """Reasons this frame must not be written, or "" if it is fine."""
    if line_score.empty:
        return "empty LineScore frame"
    if len(line_score) != EXPECTED_ROWS_PER_GAME:
        return f"expected {EXPECTED_ROWS_PER_GAME} team rows, got {len(line_score)}"

    if list(line_score.columns) != EXPECTED_SOURCE_COLUMNS:
        unexpected = [c for c in line_score.columns if c not in EXPECTED_SOURCE_COLUMNS]
        missing = [c for c in EXPECTED_SOURCE_COLUMNS if c not in line_score.columns]
        return (f"LineScore schema changed - missing {missing}, "
                f"unexpected {unexpected}")

    return ""


def normalize(line_score: pd.DataFrame) -> pd.DataFrame:
    """Rename into the project's conventions and keep only what is used."""
    return line_score.rename(columns=COLUMN_RENAMES)[TARGET_COLUMNS]


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
            summary = boxscoresummaryv3.BoxScoreSummaryV3(
                game_id=game_id, timeout=REQUEST_TIMEOUT_SECONDS
            )
            # By name, not get_data_frames()[n]: this endpoint returns nine
            # data sets, and positional access would break silently if that
            # ordering ever changed.
            line_score = summary.line_score.get_data_frame()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            problem = validate_frame(line_score)
            if problem:
                last_error = problem
            else:
                # Write-then-rename, so an interrupted write cannot leave a
                # truncated CSV a later run would skip as done.
                temp_path = out_path.with_suffix(".csv.partial")
                normalize(line_score).to_csv(temp_path, index=False, encoding="utf-8")
                temp_path.replace(out_path)
                return "fetched"

        if attempt < MAX_ATTEMPTS and not stop_requested.is_set():
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    record_failure(game_id, last_error or "unknown error")
    return "failed"


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
