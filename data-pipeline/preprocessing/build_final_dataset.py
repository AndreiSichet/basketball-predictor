"""
Reshape games_final.csv from one-row-per-team-per-game into one-row-per-game,
with home and away features side by side and a single win/loss label.

Input:  data-pipeline/data/processed/games_final.csv
        data-pipeline/data/processed/team_availability.csv
        data-pipeline/data/processed/team_advanced_rolling.csv
Output: data-pipeline/data/processed/model_dataset.csv

Early-season NaN rows (incomplete rolling windows) are left as-is. Whether
to drop or impute them is a training decision, not a pipeline one.

Features vs labels: anything derived from WL, PTS, REB or AST is a
post-game outcome and must be kept out of the feature matrix, or it leaks
the answer. LABEL_COLUMNS below is the authoritative list.

The ROLL5_/ROLL10_ versions of REB and AST are different: those average
prior games and are valid features. Only the raw single-game values are not.

GAME_ID DTYPE, and why the availability merge converts rather than the
other way round: the team-level pipeline has always read GAME_ID as a plain
integer (21500001), because it never needed the zero-padded 10-digit form.
The player box-score endpoints require that padding, so team_availability
.csv carries "0021500001" as a string. Merging the two without reconciling
would match nothing - and could do so quietly, producing a clean-looking
left join in which every availability column is NaN. The conversion happens
on the availability side so nothing already established has to change.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "games_final.csv"
AVAILABILITY_PATH = PROCESSED_DATA_DIR / "team_availability.csv"
ADVANCED_ROLLING_PATH = PROCESSED_DATA_DIR / "team_advanced_rolling.csv"
QUARTER_HALF_PATH = PROCESSED_DATA_DIR / "quarter_half_raw.csv"
QUARTER_HALF_ROLLING_PATH = PROCESSED_DATA_DIR / "quarter_half_rolling.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset.csv"
QUARTER_SCORES_FAILURES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw"
    / "quarter_scores_failures.log"
)

# WHICH GAMES ARE ALLOWED TO BE MISSING IS READ FROM THE FETCHER'S OWN
# RECORD, not hardcoded here.
#
# Three games (2025-11-19) cannot be fetched at all: BoxScoreSummaryV3
# raises an internal AttributeError on them, reproducibly. Six team-game
# rows therefore have no quarter data, which is expected and is NOT a merge
# failure. That much has been true since the corpus was built.
#
# It used to be a literal list of those three ids, and that had no expiry
# condition: the first time ANY new game failed to fetch - one nba_api
# hiccup - this merge would see an unfamiliar id and crash the whole
# retrain. Deriving the set from quarter_scores_failures.log instead means
# the expectation tracks reality without anyone editing a constant.
#
# The strictness is unchanged. An unmatched game that is not in the log
# still raises, loudly, which is the entire point of the check.

# Known before tip-off, safe to train on. ABSENT_COUNT and
# WEIGHTED_ABSENT_MIN describe who is unavailable for this game, weighted by
# trailing minutes that are themselves strictly backward-looking, so nothing
# from this game's own result reaches them.
PREGAME_FEATURE_COLUMNS = [
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
    "ABSENT_COUNT",
    "WEIGHTED_ABSENT_MIN",
]

# Unlike the rolling features, these have no legitimate NaN: availability was
# computed for all 26,398 team-games with no gaps, because an absent player
# with no known role contributes 0 rather than nulling the row. A NaN in
# these columns means the merge failed, not that history was short.
AVAILABILITY_COLUMNS = ["ABSENT_COUNT", "WEIGHTED_ABSENT_MIN"]

# The advanced rolling columns are NOT listed in PREGAME_FEATURE_COLUMNS,
# and that is deliberate. build_keep_columns() already globs every column
# starting ROLL5_ or ROLL10_, so these are picked up the moment they are
# merged in - naming them again would put them in keep_cols twice and
# produce duplicate columns downstream. The glob is why adding a rolling
# metric needs no edit here at all.
#
# Unlike the availability columns, these DO carry legitimate NaN: a team's
# first five or ten games of a season have no complete trailing window, the
# same warm-up every other ROLL5_/ROLL10_ column has. So "no NaN" is the
# wrong check for them. What must hold is that the merge matched every row -
# see attach_advanced_rolling(), which uses a merge indicator to separate
# "no match found" from "matched, value legitimately NaN".

# Known only after the game. Carried through to build targets, never inputs.
#
# Q1_PTS and HALF1_PTS join this list rather than getting bespoke handling:
# split_and_prefix() then produces HOME_/AWAY_ versions automatically, which
# is the same mechanism every prior label addition has used. They are
# post-game outcomes in the ordinary sense - a first quarter has to be played
# before it can be scored - so they are labels, never features.
POSTGAME_OUTCOME_COLUMNS = ["WL", "PTS", "REB", "AST", "Q1_PTS", "HALF1_PTS"]

# Identifiers: not features, not labels.
ID_COLUMNS = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_NAME"]

# Identical on both team rows, so kept unprefixed instead of HOME_/AWAY_.
GAME_LEVEL_COLUMNS = ["GAME_ID", "GAME_DATE"]

# Labels in the saved dataset. Drop these from X when training.
LABEL_COLUMNS = [
    "HOME_WIN",
    "HOME_PTS",
    "AWAY_PTS",
    "HOME_MARGIN",
    "TOTAL_PTS",
    "HOME_REB",
    "AWAY_REB",
    "REB_MARGIN",
    "TOTAL_REB",
    "HOME_AST",
    "AWAY_AST",
    "AST_MARGIN",
    "TOTAL_AST",
    # Q1 and 1H markets. Same shape as the full-game trio above.
    "HOME_Q1_PTS",
    "AWAY_Q1_PTS",
    "HOME_Q1_MARGIN",
    "TOTAL_Q1_PTS",
    "HOME_Q1_WIN",
    "HOME_HALF1_PTS",
    "AWAY_HALF1_PTS",
    "HOME_HALF1_MARGIN",
    "TOTAL_HALF1_PTS",
    "HOME_HALF1_WIN",
]

# Every column above that comes from the quarter/half join, so the NaN
# report at the end can address them as a group.
QUARTER_HALF_LABELS = [c for c in LABEL_COLUMNS if "Q1" in c or "HALF1" in c]


def attach_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Join the team-level availability features onto each team-game row."""
    availability = pd.read_csv(AVAILABILITY_PATH, dtype={"GAME_ID": str})
    # Read as text, then convert deliberately: the padding disappears on int
    # conversion, which is exactly what matches games_final.csv.
    availability["GAME_ID"] = availability["GAME_ID"].astype(int)

    before = len(df)
    merged = df.merge(
        availability, on=["GAME_ID", "TEAM_ID"], how="left", validate="one_to_one"
    )

    if len(merged) != before:
        raise RuntimeError(f"availability join changed rows: {before} -> {len(merged)}")

    missing = int(merged[AVAILABILITY_COLUMNS].isna().any(axis=1).sum())
    if missing:
        raise RuntimeError(
            f"{missing:,} team-game rows got no availability data. These columns "
            f"have no legitimate NaN, so this is a failed merge - check that the "
            f"GAME_ID dtypes match on both sides."
        )

    print(f"Availability merged: {len(availability):,} team-games, 0 unmatched.")
    return merged


def attach_advanced_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Join the pace/rating rolling features onto each team-game row.

    Same GAME_ID dtype reconciliation as the availability merge: this file
    stores the zero-padded 10-character form the box-score endpoints
    require, while games_final.csv has always used a plain integer.
    """
    advanced = pd.read_csv(ADVANCED_ROLLING_PATH, dtype={"GAME_ID": str})
    advanced["GAME_ID"] = advanced["GAME_ID"].astype(int)

    before = len(df)
    merged = df.merge(
        advanced,
        on=["GAME_ID", "TEAM_ID"],
        how="left",
        validate="one_to_one",
        indicator="_advanced_merge",
    )

    if len(merged) != before:
        raise RuntimeError(f"advanced join changed rows: {before} -> {len(merged)}")

    # UNMATCHED ROWS ARE REPORTED, NOT FATAL - and this is the ONLY merge in
    # this file with that exemption. Every other one still raises.
    #
    # The reason is that these twenty columns are REJECTED FEATURES. Two
    # experiments measured them against the 38-feature baseline and neither
    # won, so they sit in train_baseline.UNUSED_FEATURE_COLUMNS and
    # influence no shipped prediction (CLAUDE.md section 16). They are kept
    # built because the evidence was worth preserving, not because anything
    # consumes them.
    #
    # A hard assertion here means the first genuinely new game past the end
    # of team_advanced_rolling.csv fails the whole pipeline - and the only
    # way to satisfy it would be re-fetching 13,199 advanced box scores on
    # every retrain, hours of ingestion to keep a merge happy that feeds
    # nothing. A rejected feature set should not be able to block a
    # production pipeline it has no influence over.
    #
    # Unmatched rows simply carry NaN in columns nothing reads. What the
    # count still buys is visibility: a sudden jump to all 26,398 is a
    # dtype mismatch, and looks nothing like the handful of new games this
    # tolerates.
    unmatched = int((merged["_advanced_merge"] != "both").sum())

    merged = merged.drop(columns=["_advanced_merge"])
    added = [c for c in advanced.columns if c not in ("GAME_ID", "TEAM_ID")]
    print(f"Advanced rolling merged: {len(advanced):,} team-games, "
          f"{len(added)} columns, {unmatched:,} unmatched.")
    if unmatched:
        print(f"  {unmatched:,} row(s) have no advanced-stats match and will carry "
              f"NaN in those columns. Not fatal: they are held out of "
              f"FEATURE_COLUMNS and feed no prediction. A count near "
              f"{len(merged):,} would mean a broken merge instead.")
    return merged


def read_known_fetch_failures() -> set:
    """Game ids fetch_quarter_scores.py has ever failed on.

    THE LOG IS APPEND-ONLY - confirmed at the source, `open("a", ...)` with
    no truncation anywhere in that script - so this is a record of every
    failure ever, not of current ones. Two consequences drive the design:

      * The same id appears once per ATTEMPT, so retries produce repeats and
        the result has to be a set. The three permanently-broken games add 9
        lines to this file on every single run (3 games x 3 attempts).
      * A game that failed once and succeeded later stays here forever.
        That is why the caller checks membership in ONE DIRECTION only.

    int(), not the raw string, and this is not redundant with the split.
    The log stores the zero-padded 10-character form ("0022500259") while
    GAME_ID in the dataset is int64 (22500259). Comparing the two as-is
    yields sets that can never intersect, so every unmatched game would look
    unexpected and this merge would raise on every run - a type bug wearing
    a data bug's clothes. attach_quarter_half() already reconciles the same
    way when it reads quarter_half_raw.csv.

    A MALFORMED LINE RAISES rather than being skipped. Silently dropping one
    is how a genuinely unexpected failure slips into the "expected" set
    unnoticed - the same reasoning as everywhere else here.
    """
    if not QUARTER_SCORES_FAILURES_PATH.exists():
        # No log means nothing is known to have failed, so any unmatched
        # game is genuinely unexpected. Letting the caller raise is correct.
        return set()

    failures = set()
    with QUARTER_SCORES_FAILURES_PATH.open(encoding="utf-8") as log:
        for number, line in enumerate(log, start=1):
            # .strip() and not just rstrip("\n"): the file carries CRLF, and
            # a trailing \r inside the id would survive the tab split.
            text = line.strip()
            if not text:
                continue
            game_id, _, reason = text.partition("\t")
            if not reason or not game_id.isdigit():
                raise RuntimeError(
                    f"{QUARTER_SCORES_FAILURES_PATH.name} line {number} is not "
                    f"'<game_id>\\t<reason>': {line!r}. Refusing to guess which "
                    f"games are allowed to be missing."
                )
            failures.add(int(game_id))
    return failures


def attach_quarter_half(df: pd.DataFrame) -> pd.DataFrame:
    """Join Q1 and first-half scoring onto each team-game row.

    Same GAME_ID dtype reconciliation as the two merges above.

    validate="one_to_one" is kept even though three games are missing, and
    the distinction is worth being precise about: that flag checks the merge
    KEYS ARE UNIQUE on both sides, not that every left row finds a match. A
    left join with absent right-side rows passes it and yields NaN, which is
    exactly the intended behaviour here. Dropping the flag to accommodate the
    missing games would give up a real check - duplicate (GAME_ID, TEAM_ID)
    pairs - for nothing.
    """
    quarters = pd.read_csv(QUARTER_HALF_PATH, dtype={"GAME_ID": str})
    quarters["GAME_ID"] = quarters["GAME_ID"].astype(int)

    before = len(df)
    merged = df.merge(
        quarters,
        on=["GAME_ID", "TEAM_ID"],
        how="left",
        validate="one_to_one",
        indicator="_quarter_merge",
    )

    if len(merged) != before:
        raise RuntimeError(f"quarter join changed rows: {before} -> {len(merged)}")

    # Asserted against the known gap, not against zero. A dtype mismatch
    # would strand all 26,398 rows, so this still fails loudly for the
    # reason the indicator exists - it just tolerates the gaps the fetcher
    # itself recorded.
    unmatched = merged.loc[merged["_quarter_merge"] != "both", "GAME_ID"]
    unmatched_games = set(unmatched)
    known_failures = read_known_fetch_failures()

    # ONE DIRECTION ONLY, and the asymmetry is the point. Every unmatched
    # game must be in the log; a logged game that merged fine is not a
    # problem. The log is append-only, so an id that failed once and
    # succeeded on a later retry stays in it forever - an equality check
    # would turn that stale entry into a false alarm.
    unexpected = sorted(unmatched_games - known_failures)
    if unexpected:
        raise RuntimeError(
            f"quarter/half merge left {len(unmatched):,} rows unmatched across "
            f"{len(unmatched_games)} game(s), and {len(unexpected)} of those "
            f"game(s) are NOT in {QUARTER_SCORES_FAILURES_PATH.name}: "
            f"{unexpected[:10]}. Either the fetch genuinely failed without "
            f"being logged, or the GAME_ID dtypes disagree on the two sides."
        )

    # Each missing game strands exactly its two team-rows, never more. This
    # replaces the old hardcoded "exactly 6" and keeps the property that
    # number was really checking, without a constant that expires: a merge
    # that stranded 12 rows across 3 games would pass a pure subset test.
    if len(unmatched) != 2 * len(unmatched_games):
        raise RuntimeError(
            f"quarter/half merge left {len(unmatched):,} unmatched rows across "
            f"{len(unmatched_games)} game(s); expected exactly 2 per game. "
            f"A game should strand its home and away rows together or not at "
            f"all, so this means the merge key is wrong."
        )

    merged = merged.drop(columns=["_quarter_merge"])
    # Both numbers printed rather than one, and the log's own size beside
    # them: if the recorded-failure count drifts far above the number
    # actually unmatched, the log has accumulated stale entries from games
    # that later succeeded. Visible without anyone going looking.
    print(f"Quarter/half merged: {len(quarters):,} team-games, "
          f"{len(unmatched)} unmatched row(s) across {len(unmatched_games)} game(s), "
          f"all present in {QUARTER_SCORES_FAILURES_PATH.name} "
          f"(which records {len(known_failures)} distinct game(s)).")
    return merged


def attach_quarter_half_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Join trailing Q1/1H form onto each team-game row.

    Unlike the raw quarter scores above, this file covers the FULL 26,398
    team-game universe: build_quarter_half_rolling.py reindexes the three
    missing games back in as NaN rows so their absence propagates through
    the rolling windows instead of quietly shortening them. So zero
    unmatched is the right assertion here, exactly as for advanced rolling.

    THIS FUNCTION STAYS STRICT WHILE attach_quarter_half() CONSULTS THE
    FETCHER'S FAILURES LOG, and that inconsistency is deliberate rather than
    an oversight - do not harmonise them. A fetch failure shows up in the
    two files by different mechanisms: quarter_half_raw.csv simply has no
    row for the game, while this file HAS a row that is all NaN. So an
    unmatched row here does not mean "a game could not be fetched", it means
    the reindex-onto-the-full-universe step did not do its job, which no
    failures log can excuse. Same shape as the attach_advanced_rolling()
    asymmetry documented in CLAUDE.md section 12.

    The NaN these columns carry is the ordinary rolling warm-up plus that
    propagation - both legitimate - which is why the indicator does the
    merge check rather than a NaN count.

    build_keep_columns() globs every ROLL5_/ROLL10_ column, so these are
    picked up and HOME_/AWAY_ prefixed automatically; nothing needs naming
    twice. They are held out of training by train_baseline.py's
    UNUSED_FEATURE_COLUMNS, not by being absent from the dataset.
    """
    rolling = pd.read_csv(QUARTER_HALF_ROLLING_PATH, dtype={"GAME_ID": str})
    rolling["GAME_ID"] = rolling["GAME_ID"].astype(int)

    before = len(df)
    merged = df.merge(
        rolling,
        on=["GAME_ID", "TEAM_ID"],
        how="left",
        validate="one_to_one",
        indicator="_qh_rolling_merge",
    )

    if len(merged) != before:
        raise RuntimeError(
            f"quarter/half rolling join changed rows: {before} -> {len(merged)}"
        )

    unmatched = int((merged["_qh_rolling_merge"] != "both").sum())
    if unmatched:
        raise RuntimeError(
            f"{unmatched:,} team-game rows found no quarter/half rolling match. "
            f"That file is built against the full universe, so this is a failed "
            f"merge, not a warm-up gap - check the GAME_ID dtypes."
        )

    merged = merged.drop(columns=["_qh_rolling_merge"])
    added = [c for c in rolling.columns if c not in ("GAME_ID", "TEAM_ID")]
    print(f"Quarter/half rolling merged: {len(rolling):,} team-games, "
          f"{len(added)} columns, 0 unmatched.")
    return merged


def home_win_label(home_points: pd.Series, away_points: pd.Series) -> pd.Series:
    """1 if the home team led this period, 0 if it trailed, NA if tied.

    A TIE IS NOT A HOME LOSS, and this is the one place the quarter/half
    labels genuinely cannot mirror HOME_WIN. A full game cannot end level, so
    HOME_WIN is a clean binary read off WL. A quarter can and often does:
    611 first quarters (4.6%) and 431 first halves (3.3%) are tied. Coding
    those as 0 would assert the away team led in 611 games where nobody did -
    the same unknown-as-zero substitution that empty MIN strings, V3's
    zero-filled absent players and NOT YET SUBMITTED injury rows each turned
    out to be.

    So a tie is NA, and the pipeline stops there. Whether to drop those rows,
    model the market as three-way, or treat it as a push is a training
    decision, exactly as the docstring says about early-season NaN. The
    margin and total labels stay valid for tied periods - only the binary
    is undefined.
    """
    known = home_points.notna() & away_points.notna()
    decided = known & (home_points != away_points)
    return (home_points > away_points).astype("Int64").where(decided)


def add_period_labels(merged: pd.DataFrame, period: str) -> pd.DataFrame:
    """Margin / total / win for one period, mirroring the full-game trio."""
    home, away = f"HOME_{period}_PTS", f"AWAY_{period}_PTS"
    merged[f"HOME_{period}_MARGIN"] = merged[home] - merged[away]
    merged[f"TOTAL_{period}_PTS"] = merged[home] + merged[away]
    merged[f"HOME_{period}_WIN"] = home_win_label(merged[home], merged[away])
    return merged


def build_keep_columns(df: pd.DataFrame) -> list:
    rolling_cols = [c for c in df.columns if c.startswith("ROLL5_") or c.startswith("ROLL10_")]
    return ID_COLUMNS + POSTGAME_OUTCOME_COLUMNS + rolling_cols + PREGAME_FEATURE_COLUMNS


def split_and_prefix(df: pd.DataFrame, keep_cols: list, prefix: str) -> pd.DataFrame:
    subset = df.loc[df["IS_HOME"] == (prefix == "HOME"), keep_cols]
    rename_map = {c: f"{prefix}_{c}" for c in keep_cols if c not in GAME_LEVEL_COLUMNS}
    return subset.rename(columns=rename_map)


def main():
    df = pd.read_csv(INPUT_PATH)
    df = attach_availability(df)
    df = attach_advanced_rolling(df)
    df = attach_quarter_half(df)
    df = attach_quarter_half_rolling(df)

    keep_cols = build_keep_columns(df)

    home = split_and_prefix(df, keep_cols, "HOME")
    away = split_and_prefix(df, keep_cols, "AWAY")

    merged = home.merge(away, on=GAME_LEVEL_COLUMNS, how="inner", validate="one_to_one")

    # Labels. HOME_PTS/AWAY_PTS stay for debugging and for checking actual
    # totals against over/under lines once odds data exists.
    merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)
    merged["HOME_MARGIN"] = merged["HOME_PTS"] - merged["AWAY_PTS"]
    merged["TOTAL_PTS"] = merged["HOME_PTS"] + merged["AWAY_PTS"]
    merged["REB_MARGIN"] = merged["HOME_REB"] - merged["AWAY_REB"]
    merged["TOTAL_REB"] = merged["HOME_REB"] + merged["AWAY_REB"]
    merged["AST_MARGIN"] = merged["HOME_AST"] - merged["AWAY_AST"]
    merged["TOTAL_AST"] = merged["HOME_AST"] + merged["AWAY_AST"]
    for period in ("Q1", "HALF1"):
        merged = add_period_labels(merged, period)
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

    # Different failure signature from the rolling columns above: any NaN
    # here means the merge dropped rows, not that history was insufficient.
    availability_output = [
        f"{side}_{col}" for side in ("HOME", "AWAY") for col in AVAILABILITY_COLUMNS
    ]
    availability_nan = merged[availability_output].isna().sum()
    print("NaN counts in availability features (must all be 0):")
    print(availability_nan.to_string())
    print("Availability check: " + ("PASS" if availability_nan.sum() == 0 else "FAIL"))

    # Two different reasons a quarter/half label can be NaN, and conflating
    # them would hide either one. Missing source data affects all ten
    # columns equally; a tie affects only the two binaries.
    print("\nQuarter/half label NaN, by cause:")
    no_data = merged["HOME_Q1_PTS"].isna()
    print(f"  games with no quarter data : {int(no_data.sum())}  "
          f"(affects all {len(QUARTER_HALF_LABELS)} columns)")
    for period in ("Q1", "HALF1"):
        tied = (~no_data) & (merged[f"HOME_{period}_PTS"]
                             == merged[f"AWAY_{period}_PTS"])
        print(f"  {period + ' tied':<26} : {int(tied.sum()):,}  "
              f"({tied.mean():.1%} - HOME_{period}_WIN only)")

    print("\n  resulting NaN per column:")
    for col in QUARTER_HALF_LABELS:
        print(f"    {col:<20}{int(merged[col].isna().sum()):>6,}")

    print(f"\nLabel columns (exclude from training features): {LABEL_COLUMNS}")
    print(merged[LABEL_COLUMNS].describe())

    print("\nPreview:")
    print(merged.head())


if __name__ == "__main__":
    main()
