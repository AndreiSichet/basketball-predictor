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
OUTPUT_PATH = PROCESSED_DATA_DIR / "model_dataset.csv"

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
POSTGAME_OUTCOME_COLUMNS = ["WL", "PTS", "REB", "AST"]

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
]


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

    # The indicator is the point. These columns have real NaN from the
    # rolling warm-up, so counting NaN cannot tell a failed merge from a
    # team's first game of the season. Only the indicator can.
    unmatched = int((merged["_advanced_merge"] != "both").sum())
    if unmatched:
        raise RuntimeError(
            f"{unmatched:,} team-game rows found no advanced-stats match. That is a "
            f"failed merge, not a rolling warm-up - check that GAME_ID dtypes agree "
            f"on both sides."
        )

    merged = merged.drop(columns=["_advanced_merge"])
    added = [c for c in advanced.columns if c not in ("GAME_ID", "TEAM_ID")]
    print(f"Advanced rolling merged: {len(advanced):,} team-games, "
          f"{len(added)} columns, 0 unmatched.")
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

    print(f"\nLabel columns (exclude from training features): {LABEL_COLUMNS}")
    print(merged[LABEL_COLUMNS].describe())

    print("\nPreview:")
    print(merged.head())


if __name__ == "__main__":
    main()
