"""
Replay verification for live_quarter_half_features.py.

Same bar as live_features.verify_against_training_data(): rebuild features
for real past games and diff them against the pipeline's own rows in
model_dataset.csv, which are what the shipped models were actually fit on.
A mismatch here means live predictions would be computed from different
numbers than training was, silently.

THE SAMPLE IS DELIBERATELY UNEVEN, not random. Three populations, because
they exercise different code:

  1. Mid-season games across every era - the ordinary case, where an exact
     match is the only acceptable result.
  2. Season openers and early-season games - where the correct behaviour is
     to RAISE, not to return a number. A function that silently produced
     something here would be worse than one that crashed.
  3. The games immediately following 2025-11-19 - the three un-fetchable
     games. Their absence must propagate through the trailing window, which
     it only does if the history frame was reindexed. This is the case a
     naive implementation gets wrong while passing every other check.

Run:  python ml-training/verify_live_quarter_half.py
"""

import numpy as np
import pandas as pd

from live_features import load_games_final
from live_quarter_half_features import (
    InsufficientQuarterHalfHistory,
    get_live_quarter_half_features,
    load_quarter_half_history,
    verify_manifest_agrees,
)
from train_baseline import load_dataset
from train_quarter_half_baseline import FEATURE_COLUMNS

MID_SEASON_SAMPLE = 25
TOLERANCE = 1e-9

# The three games with no line score, and the date they fall on.
MISSING_GAME_DATE = pd.Timestamp("2025-11-19")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def compare(expected_row, actual_frame) -> tuple:
    """Largest absolute difference across the 30 columns, and the worst name."""
    worst_name, worst_diff = None, 0.0
    for column in FEATURE_COLUMNS:
        expected = float(expected_row[column])
        actual = float(actual_frame[column].iloc[0])
        diff = abs(expected - actual)
        if diff > worst_diff:
            worst_name, worst_diff = column, diff
    return worst_name, worst_diff


def main():
    section("SETUP")
    verify_manifest_agrees()

    dataset = load_dataset()
    games_final = load_games_final()
    history = load_quarter_half_history()
    print(f"  dataset {len(dataset):,} games | games_final {len(games_final):,} "
          f"team-games | quarter/half history {len(history):,} team-games")

    complete = dataset.dropna(subset=FEATURE_COLUMNS)
    incomplete = dataset[dataset[FEATURE_COLUMNS].isna().any(axis=1)]
    print(f"  {len(complete):,} games have all 30 features; "
          f"{len(incomplete):,} do not (these must raise)")

    # ---------------------------------------------------------------- 1
    section(f"1. MID-SEASON REPLAY - {MID_SEASON_SAMPLE} games across all eras")
    print("Every one must reproduce model_dataset.csv exactly.\n")

    # Spread the sample over seasons rather than sampling uniformly, so no
    # era can be silently absent.
    per_season = max(1, MID_SEASON_SAMPLE // complete["SEASON"].nunique())
    sample = (complete.groupby("SEASON", group_keys=False)
              .apply(lambda g: g.sample(min(per_season, len(g)), random_state=42))
              .head(MID_SEASON_SAMPLE))

    failures, worst_overall, worst_where = 0, 0.0, None
    for _, row in sample.iterrows():
        frame = get_live_quarter_half_features(
            int(row["HOME_TEAM_ID"]), int(row["AWAY_TEAM_ID"]),
            row["GAME_DATE"], history, games_final,
        )
        name, diff = compare(row, frame)
        if diff > TOLERANCE:
            failures += 1
            print(f"  MISMATCH {row['GAME_DATE'].date()} "
                  f"{int(row['GAME_ID'])}: {name} off by {diff:.3e}")
        if diff > worst_overall:
            worst_overall, worst_where = diff, f"{name} on {row['GAME_DATE'].date()}"

    seasons = sorted(sample["SEASON"].unique())
    print(f"  seasons covered : {seasons[0]}-{seasons[-1]} "
          f"({len(seasons)} of {complete['SEASON'].nunique()})")
    print(f"  games checked   : {len(sample)}")
    print(f"  mismatches      : {failures}")
    print(f"  largest diff    : {worst_overall:.3e}  ({worst_where})")
    print(f"  {'PASS' if failures == 0 else 'FAIL'}")

    # ---------------------------------------------------------------- 2
    section("2. EARLY SEASON - the function must RAISE, not return")
    print("A linear model cannot score a partial row, so returning anything")
    print("here would be worse than crashing.\n")

    early = incomplete.sort_values("GAME_DATE").groupby("SEASON").head(1).head(6)
    raised = returned = 0
    for _, row in early.iterrows():
        label = (f"{row['GAME_DATE'].date()} season {int(row['SEASON'])} "
                 f"game {int(row['GAME_ID'])}")
        try:
            get_live_quarter_half_features(
                int(row["HOME_TEAM_ID"]), int(row["AWAY_TEAM_ID"]),
                row["GAME_DATE"], history, games_final,
            )
            returned += 1
            print(f"  {label}: RETURNED a row - should have raised")
        except InsufficientQuarterHalfHistory as error:
            raised += 1
            print(f"  {label}: raised ({error.side} team {error.team_id}, "
                  f"{error.games_played} prior game(s), "
                  f"{len(error.missing)} feature(s) NaN)")

    print(f"\n  raised {raised}, wrongly returned {returned}")
    print(f"  {'PASS' if returned == 0 else 'FAIL'}")

    # ---------------------------------------------------------------- 3
    section("3. THE REINDEX CASE - games after the 3 un-fetchable fixtures")
    print("The pipeline carries 2025-11-19 as NaN rows so the gap propagates.")
    print("If live features skipped them instead, windows would quietly differ")
    print("here and nowhere else - the one case a naive version gets wrong.\n")

    affected_teams = games_final.loc[
        games_final["GAME_ID"].isin([22500259, 22500260, 22500261]), "TEAM_ID"
    ].unique()
    window = dataset[
        (dataset["GAME_DATE"] > MISSING_GAME_DATE)
        & (dataset["GAME_DATE"] <= MISSING_GAME_DATE + pd.Timedelta(days=25))
        & (dataset["HOME_TEAM_ID"].isin(affected_teams)
           | dataset["AWAY_TEAM_ID"].isin(affected_teams))
    ].sort_values("GAME_DATE")

    print(f"  {len(affected_teams)} teams affected, "
          f"{len(window)} of their games in the following 25 days")

    checked = matched = raised_here = 0
    for _, row in window.iterrows():
        expects_nan = row[FEATURE_COLUMNS].isna().any()
        try:
            frame = get_live_quarter_half_features(
                int(row["HOME_TEAM_ID"]), int(row["AWAY_TEAM_ID"]),
                row["GAME_DATE"], history, games_final,
            )
        except InsufficientQuarterHalfHistory:
            raised_here += 1
            checked += 1
            if not expects_nan:
                print(f"  {row['GAME_DATE'].date()}: raised, but the pipeline "
                      f"has a complete row - MISMATCH")
            continue

        checked += 1
        if expects_nan:
            print(f"  {row['GAME_DATE'].date()}: returned a row, but the "
                  f"pipeline has NaN - MISMATCH (the reindex is missing)")
            continue
        _, diff = compare(row, frame)
        if diff <= TOLERANCE:
            matched += 1
        else:
            print(f"  {row['GAME_DATE'].date()}: off by {diff:.3e}")

    agree = matched + raised_here
    print(f"\n  checked {checked}: {matched} reproduced exactly, "
          f"{raised_here} correctly refused (pipeline NaN)")
    print(f"  live and pipeline agree on all {checked}: "
          f"{'PASS' if agree == checked else 'FAIL'}")

    # ---------------------------------------------------------------- 4
    section("4. A REAL PREDICTION, END TO END")
    import joblib
    from live_quarter_half_features import MODELS_DIR

    row = complete.sort_values("GAME_DATE").iloc[-1]
    frame = get_live_quarter_half_features(
        int(row["HOME_TEAM_ID"]), int(row["AWAY_TEAM_ID"]),
        row["GAME_DATE"], history, games_final,
    )
    print(f"  {row['GAME_DATE'].date()}  "
          f"{int(row['AWAY_TEAM_ID'])} @ {int(row['HOME_TEAM_ID'])}\n")
    print(f"  {'MODEL':<12}{'PREDICTED':>11}{'ACTUAL':>9}")
    print("  " + "-" * 32)
    for key, target in (("q1_spread", "HOME_Q1_MARGIN"),
                        ("q1_total", "TOTAL_Q1_PTS"),
                        ("1h_spread", "HOME_HALF1_MARGIN"),
                        ("1h_total", "TOTAL_HALF1_PTS")):
        model = joblib.load(MODELS_DIR / f"{key}.joblib")
        print(f"  {key:<12}{float(model.predict(frame)[0]):>11.2f}"
              f"{row[target]:>9.0f}")
    for key, target in (("q1_winner", "HOME_Q1_WIN"),
                        ("1h_winner", "HOME_HALF1_WIN")):
        model = joblib.load(MODELS_DIR / f"{key}.joblib")
        probability = float(model.predict_proba(frame)[0][1])
        actual = row[target]
        shown = "tied" if pd.isna(actual) else f"{int(actual)}"
        print(f"  {key:<12}{probability:>11.4f}{shown:>9}")

    section("RESULT")
    ok = failures == 0 and returned == 0 and agree == checked
    print("PASS - live features reproduce the pipeline, and refuse where the"
          if ok else "FAIL")
    if ok:
        print("pipeline has no answer to reproduce.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
