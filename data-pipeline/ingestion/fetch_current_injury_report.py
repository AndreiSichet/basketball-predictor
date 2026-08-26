"""
Fetch the NBA's official injury report as of right now, parsed into rows.

**This is the odd one out in ingestion/.** Every other script here builds a
fixed historical archive: run it once, get the same answer forever. This
one always answers "right now", and the answer changes hourly. It exists to
serve predictions for games that have not been played, which is the one
thing the box-score pipeline structurally cannot do - tomorrow's game has
no box score to read absences from.

Written as an importable module rather than a script, because
live_features.py will call get_current_injury_status() at serving time.
Same shape as build_elo_ratings.py: functions plus a __main__ block for
standalone checking.

PARSING IS NOT DONE HERE. The report is a PDF whose columns are separated
by position rather than by any character, and whose wrapped "Reason" text
extracts out of visual order - a player's reason can appear before that
player's own name. Hand-rolling that needs coordinate clustering, so this
delegates to `nbainjuries`, which does it correctly. Verified against the
2026-03-14 07:15 ET report: team names come back cleanly separated
("Brooklyn Nets", not "BrooklynNets") and multi-line reasons reassemble
onto the right player.

STATUS POLICY - a deliberate decision, not a default. The report carries
five statuses; the models were trained on a binary played/did-not-play
signal, so the five collapse:

    Out, Doubtful                     -> absent
    Questionable, Probable, Available -> present

Doubtful sits close enough to Out in real outcome rates that grouping it
with "will not play" is the safer approximation. Questionable is closer to
"expected to play, uncertainty noted". This is deliberately a simple,
auditable binary rule rather than a probabilistic weighting - the same
principle applied to ABSENCES_COUNT_AS_ZERO: ship the simplest defensible
policy, and only build something more nuanced once evidence shows it falls
short. An unrecognised status raises rather than defaulting to present.

"NOT YET SUBMITTED" IS NOT "NOBODY IS OUT". Teams file at different times,
so a report routinely contains games where one or both teams have not
reported yet. Those rows carry no player and no status. They are returned
separately, never folded into the player table, because counting them as
zero absences is the same unknown-as-zero bug this project has already
found twice - the empty-string MIN in box scores, and V3's zero-filled
stat lines for absent players. A caller that ignores `pending` will
silently treat an unreported team as fully healthy.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# Reports are published on Eastern time regardless of where this runs.
EASTERN = ZoneInfo("America/New_York")

# Roughly hourly, but the exact minute moves, so probe every quarter hour.
STEP_MINUTES = 15
MAX_STEPS = 48  # 12 hours back - enough to cross an overnight gap
DELAY_BETWEEN_PROBES_SECONDS = 0.5

ABSENT_STATUSES = frozenset({"Out", "Doubtful"})
PRESENT_STATUSES = frozenset({"Questionable", "Probable", "Available"})

PENDING_MARKER = "NOT YET SUBMITTED"

STATUS_COLUMN = "Current Status"
PLAYER_COLUMN = "Player Name"
REASON_COLUMN = "Reason"


class NoReportAvailable(Exception):
    """No injury report exists in the searched window.

    Normal between seasons and during long overnight gaps. Raised rather
    than returning an empty frame so a caller cannot mistake "nothing
    published" for "nobody is injured".
    """


@dataclass(frozen=True)
class InjuryReport:
    """One parsed report."""

    timestamp: datetime
    players: pd.DataFrame   # real rows, with IS_ABSENT
    pending: pd.DataFrame   # teams that have not filed yet

    @property
    def absent(self) -> pd.DataFrame:
        return self.players[self.players["IS_ABSENT"]]


def normalize_player_name(name: str) -> str:
    """"Porter Jr., Michael" -> "Michael Porter Jr."

    The box scores store "First Last Suffix" and never contain a comma
    (verified across the corpus), so the comma is a safe split point. The
    suffix travels with the surname, which is where it already sits.

    Note for whoever wires up the PLAYER_ID join: match on the full name.
    Craig Porter Jr., Michael Porter Jr. and Otto Porter Jr. all exist, so
    a surname-only match would collide.
    """
    if not isinstance(name, str) or "," not in name:
        return name.strip() if isinstance(name, str) else name
    surname, forename = name.split(",", 1)
    return f"{forename.strip()} {surname.strip()}".strip()


def find_latest_report(as_of: datetime = None, max_steps: int = MAX_STEPS) -> datetime:
    """Newest report at or before `as_of`, searching backward.

    Uses nbainjuries' own check_reportvalid rather than probing HTTP status
    codes directly - the library owns the URL format, so it stays right
    when that format changes.
    """
    from nbainjuries import injury

    if as_of is None:
        as_of = datetime.now(EASTERN)

    # The library takes naive datetimes and treats them as ET.
    anchor = as_of.replace(tzinfo=None) if as_of.tzinfo else as_of
    anchor = anchor.replace(
        minute=(anchor.minute // STEP_MINUTES) * STEP_MINUTES, second=0, microsecond=0
    )

    for step in range(max_steps):
        candidate = anchor - timedelta(minutes=STEP_MINUTES * step)
        try:
            if injury.check_reportvalid(candidate):
                return candidate
        except Exception:
            # A transport error is not proof of absence; keep stepping.
            pass
        time.sleep(DELAY_BETWEEN_PROBES_SECONDS)

    raise NoReportAvailable(
        f"No injury report in the {max_steps * STEP_MINUTES / 60:.0f} hours before "
        f"{anchor:%Y-%m-%d %I:%M %p} ET. Expected between seasons - the NBA "
        f"publishes these only around game days."
    )


def apply_status_policy(players: pd.DataFrame) -> pd.DataFrame:
    """Collapse the five statuses into IS_ABSENT."""
    statuses = players[STATUS_COLUMN].astype(str).str.strip()

    unknown = sorted(set(statuses) - ABSENT_STATUSES - PRESENT_STATUSES)
    if unknown:
        raise ValueError(
            f"Unrecognised status value(s): {unknown}. The five documented "
            f"statuses are {sorted(ABSENT_STATUSES | PRESENT_STATUSES)}. Decide "
            f"explicitly how a new one maps rather than letting it default."
        )

    players = players.copy()
    players["IS_ABSENT"] = statuses.isin(ABSENT_STATUSES)
    players["PLAYER_NAME_NORMALIZED"] = players[PLAYER_COLUMN].map(normalize_player_name)
    return players


def get_current_injury_status(as_of: datetime = None) -> InjuryReport:
    """The newest available report, split into real rows and pending teams.

    Raises NoReportAvailable if nothing is published in the search window.
    """
    from nbainjuries import injury

    timestamp = find_latest_report(as_of=as_of)
    frame = injury.get_reportdata(timestamp, return_df=True)

    # A pending row has no player and no status, only the marker in Reason.
    is_pending = (
        frame[PLAYER_COLUMN].isna()
        | (frame[PLAYER_COLUMN].astype(str).str.strip() == "")
        | frame[REASON_COLUMN].astype(str).str.upper().str.contains(PENDING_MARKER)
    )

    pending = frame[is_pending].copy()
    players = apply_status_policy(frame[~is_pending].copy())

    return InjuryReport(timestamp=timestamp, players=players, pending=pending)


def main():
    print("Looking for the most recent injury report...\n")

    try:
        report = get_current_injury_status()
    except NoReportAvailable as error:
        print("NO REPORT CURRENTLY AVAILABLE")
        print(f"  {error}")
        print("\nThis is correct behaviour, not a failure: nothing is published")
        print("between seasons. Nothing stale was returned.")
        return 0

    print(f"Report: {report.timestamp:%Y-%m-%d %I:%M %p} ET")
    print(f"  player rows : {len(report.players):,}")
    print(f"  pending rows: {len(report.pending):,}")

    print("\n  status breakdown:")
    counts = report.players[STATUS_COLUMN].value_counts()
    for status, count in counts.items():
        bucket = "absent" if status in ABSENT_STATUSES else "present"
        print(f"    {status:<14} {count:>4}   -> {bucket}")
    print(f"    {'IS_ABSENT True':<14} {int(report.players['IS_ABSENT'].sum()):>4}")

    if not report.pending.empty:
        print(f"\n  NOT YET SUBMITTED - unknown, NOT zero absences:")
        for matchup, group in report.pending.groupby("Matchup"):
            teams = ", ".join(sorted(group["Team"].astype(str)))
            print(f"    {matchup}: {teams}")

    if not report.players.empty:
        first_matchup = report.players["Matchup"].iloc[0]
        sample = report.players[report.players["Matchup"] == first_matchup]
        print(f"\n  parsed table for {first_matchup}:")
        print(sample[["Team", PLAYER_COLUMN, "PLAYER_NAME_NORMALIZED",
                      STATUS_COLUMN, "IS_ABSENT"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
