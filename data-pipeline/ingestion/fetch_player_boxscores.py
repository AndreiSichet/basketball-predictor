"""
Pull per-game player box scores from nba_api's BoxScoreTraditionalV3, one
file per game, into data-pipeline/data/raw/player_boxscores/.

V3, not V2. V2 is deprecated and warns that it will be removed, so there is
no dual path here - the resume logic below leaves already-downloaded V2
files alone and fetches only what is missing, all through V3.

V3 returns camelCase columns with different names for everything, so the
frame is normalized into the existing schema before it is written. Nothing
downstream should have to know which endpoint produced a given file.

Two things about V3 that are not obvious and were confirmed by inspecting
real games from both 2015-16 and 2025-26:

  * A player who did not appear has MIN == "" - an empty string, NOT NaN.
    A .isna() check finds none of them, so the availability signal would
    silently come back "nobody was ever missing". Use has_played() below.
  * V3's `position` is NOT V2's START_POSITION. In a 2025-26 game it
    happened to be non-empty for exactly the five starters; in a 2015-16
    game it was non-empty for nine and ten players, bench included. It is
    the listed position, populated inconsistently. It is therefore mapped
    to POSITION, and START_POSITION is written empty rather than filled
    from it - calling it START_POSITION would be a lie the schema tells
    silently. See TARGET_COLUMNS for the full canonical shape.

The 34-column V3 response was byte-identical across those two seasons, so
one normalization layer covers all 11 seasons with no era special-casing.

SCALE - read before starting. One call per game, 13,199 games. Hours, not
minutes, so:

  * Run it in the background. It is not an interactive script.
  * It is safe to interrupt. Each game is a separate file and an existing
    file is skipped, so Ctrl+C and a later rerun continue where this left
    off rather than starting over.
  * Some calls will fail. At this volume transient timeouts are normal.
    Failures are logged and skipped, never fatal, and a rerun retries them.

THREADING - what it does and does not change. Workers exist to overlap
network latency, not to hit the API harder: every request passes through
one shared rate limiter, so the request rate is a global ceiling no matter
how many workers run. Raising REQUESTS_PER_SECOND is the only change that
makes this more aggressive - MAX_WORKERS alone does not.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GAMES_FINAL_PATH = DATA_DIR / "processed" / "games_final.csv"
OUTPUT_DIR = DATA_DIR / "raw" / "player_boxscores"
# Kept out of OUTPUT_DIR so that directory stays purely <game_id>.csv files.
FAILURES_LOG_PATH = DATA_DIR / "raw" / "player_boxscores_failures.log"

# Global ceiling across all workers, matching the original pull's rate.
REQUESTS_PER_SECOND = 1.0
MAX_WORKERS = 6

# Transient failures are expected over 13,199 calls; a few backed-off
# retries turn most of them into successes within the same run.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT_SECONDS = 30

PROGRESS_EVERY = 100

# V3 column -> the schema kept on disk. PLAYER_NAME is built separately
# from firstName + familyName, which V3 splits and V2 did not.
V3_TO_TARGET = {
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamTricode": "TEAM_ABBREVIATION",
    "teamCity": "TEAM_CITY",
    "personId": "PLAYER_ID",
    "position": "POSITION",
    "comment": "COMMENT",
    "minutes": "MIN",
    "fieldGoalsMade": "FGM",
    "fieldGoalsAttempted": "FGA",
    "fieldGoalsPercentage": "FG_PCT",
    "threePointersMade": "FG3M",
    "threePointersAttempted": "FG3A",
    "threePointersPercentage": "FG3_PCT",
    "freeThrowsMade": "FTM",
    "freeThrowsAttempted": "FTA",
    "freeThrowsPercentage": "FT_PCT",
    "reboundsOffensive": "OREB",
    "reboundsDefensive": "DREB",
    "reboundsTotal": "REB",
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "turnovers": "TO",
    "foulsPersonal": "PF",
    "points": "PTS",
    "plusMinusPoints": "PLUS_MINUS",
}

# The canonical schema: this order, these 30 columns, every file, whichever
# endpoint produced it.
#
# NICKNAME and START_POSITION are written empty on the V3 path because V3
# does not return them - "this source never told us", the same honesty the
# pipeline already applies with NaN for a team's first REST_DAYS or an
# incomplete rolling window. They are NOT derived from row order: box
# scores conventionally list starters first, but that is a convention, not
# a guarantee, and inventing a starter flag from it would be exactly the
# kind of plausible-looking fiction this schema is meant to avoid.
# V2-fetched files already on disk carry the real values for both.
TARGET_COLUMNS = [
    "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY",
    "PLAYER_ID", "PLAYER_NAME",
    "NICKNAME", "START_POSITION", "POSITION", "COMMENT", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS",
]

# Columns the V3 response simply does not contain.
V3_MISSING_COLUMNS = ["NICKNAME", "START_POSITION"]

# Everything after MIN is a performance stat, blanked for a player who did
# not appear. V3 returns 0 for those rows; V2 returned empty. They are
# different claims - "he played and scored nothing" versus "he did not
# play" - and a 0 would be counted as a real performance by any average
# that does not filter on has_played() first. Same principle as not calling
# V3's position START_POSITION: a source being silent is not a value.
STAT_COLUMNS = TARGET_COLUMNS[TARGET_COLUMNS.index("FGM"):]


def has_played(minutes) -> "pd.Series | bool":
    """Did this player actually appear?

    THE definition of that question for this project - import it, never
    restate it. A player who did not appear has MIN == "" (empty string),
    not NaN, so notna() alone silently counts every absent player as
    present and any absence feature built on it computes to zero.

    Accepts a Series (returns a boolean mask) or a single value.
    """
    if isinstance(minutes, pd.Series):
        return minutes.notna() & (minutes.astype(str).str.strip() != "")
    return minutes is not None and not pd.isna(minutes) and str(minutes).strip() != ""


def normalize_player_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a V3 PlayerStats frame into the schema kept on disk.

    Raises if V3 stops returning a column this depends on, so a schema
    change becomes a logged failure for that game rather than a quietly
    malformed file that a later run would skip as already done.
    """
    required = set(V3_TO_TARGET) | {"firstName", "familyName"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"V3 response missing expected columns: {sorted(missing)}")

    normalized = raw.rename(columns=V3_TO_TARGET)
    normalized["PLAYER_NAME"] = (
        raw["firstName"].fillna("").astype(str).str.strip()
        + " "
        + raw["familyName"].fillna("").astype(str).str.strip()
    ).str.strip()

    # Present but empty, so the file matches the canonical shape without
    # claiming a signal V3 never provided.
    for column in V3_MISSING_COLUMNS:
        normalized[column] = ""

    # A player who did not appear has no stat line. V3 fills those rows
    # with 0; blank them so "did not play" reads identically in every file.
    normalized = normalized[TARGET_COLUMNS].copy()
    absent = ~has_played(normalized["MIN"])
    if absent.any():
        normalized.loc[absent, STAT_COLUMNS] = ""

    # MIN stays the raw "MM:SS" string. Parsing it to a number is feature
    # engineering, and every other raw file in this project is kept raw.
    return normalized


class RateLimiter:
    """Allows one request per interval, shared across every worker.

    The point of throttling globally rather than sleeping inside each
    worker: with a per-worker delay the real request rate would scale with
    MAX_WORKERS, so tuning for speed would quietly also tune for
    aggressiveness. Here the two are independent.
    """

    def __init__(self, requests_per_second: float):
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait < 0:
                wait = 0.0
            self._next_allowed = max(now, self._next_allowed) + self._min_interval

        if wait:
            time.sleep(wait)


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
            box_score = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id, timeout=REQUEST_TIMEOUT_SECONDS
            )
            # By name, not get_data_frames()[0]: this endpoint also returns
            # TeamStats and TeamStarterBenchStats, and positional access
            # would break silently if that ordering ever changed.
            players = box_score.player_stats.get_data_frame()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if players.empty:
                # Never write an empty file: it would satisfy the resume
                # check forever and quietly bake a hole into the dataset.
                last_error = "empty PlayerStats frame"
            else:
                try:
                    normalized = normalize_player_stats(players)
                except KeyError as exc:
                    last_error = str(exc)
                else:
                    # Write-then-rename, so an interrupted write cannot
                    # leave a truncated CSV a later run would skip as done.
                    temp_path = out_path.with_suffix(".csv.partial")
                    normalized.to_csv(temp_path, index=False, encoding="utf-8")
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

    print(f"{len(game_ids)} games in {GAMES_FINAL_PATH.name}.")
    print(f"{already_present} already downloaded, {remaining} to fetch via V3.")
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
            # Everything already written stays valid, and a rerun resumes
            # from it, so stopping early costs only the in-flight games.
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
