"""
Baselines for the five player-prop targets, before any XGBoost is touched.

  input: data-pipeline/data/processed/player_dataset.csv  (280,943 rows)

Same discipline every team-level target has been held to: establish what a
trivially simple method already achieves, so a later model has a real bar
to clear rather than an impressive-sounding number with no reference point.

TWO BASELINES PER TARGET:

  Naive - the player's own ROLL10_<TARGET>, i.e. "he'll do what he's been
  doing." There is no separate league-mean baseline here, unlike the team
  models' always-predict-home. A personalised trailing average is already
  the natural floor for a prop, and a league mean would be a straw man:
  nobody prices Jokic's rebounds off the league average.

  LinearRegression - all 17 features, StandardScaler fit on train only.

MIN_NUMERIC IS NOT A TARGET HERE, deliberately. Predicting playing time is
a different problem from predicting production given playing time: it turns
on coaching decisions, blowout risk and foul trouble, none of which this
feature set represents. Bolting it on as a sixth target would produce a
number that looks like the others and means something else. It deserves its
own pass.

It is also not a FEATURE - see build_player_dataset.py. This game's minutes
are a post-game outcome, and knowing a player logged 38 of them gives away
most of his points. The assertion in check_no_leakage() enforces that here
too, rather than trusting the upstream split to stay correct.

ROWS WITH INCOMPLETE HISTORY ARE DROPPED, mirroring the team-level
baselines exactly: LinearRegression cannot take NaN, and imputing a
player's trailing average would invent the very thing being measured.
Expect a larger fraction than the team-level 13.1% - players enter and
leave the league constantly, and a player-season warm-up recurs for every
rookie, call-up and mid-season signing, not just 30 teams a year.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# The column groups are defined by the script that builds the dataset, and
# TEST_SEASON_COUNT by the team-level baseline. Both are imported rather
# than restated so the split boundary and the feature list cannot drift.
PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
sys.path.insert(0, str(PIPELINE_DIR))
from build_player_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    OUTPUT_PATH as DATASET_PATH,
    PLAYER_FEATURE_COLUMNS,
    TEAM_CONTEXT_COLUMNS,
)

from train_baseline import TEST_SEASON_COUNT  # noqa: E402

TARGETS = ["PTS", "REB", "AST", "FG3M", "PRA"]

# Excluded on purpose - see the module docstring.
NOT_A_TARGET = "MIN_NUMERIC"

TARGET_LABELS = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "FG3M": "Threes made",
    "PRA": "Points+Reb+Ast",
}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check_no_leakage() -> None:
    """No post-game outcome may appear among the inputs.

    Enforced here as well as upstream: this is the file that would actually
    do the damage, and a feature list is easy to extend without noticing
    what it now contains.
    """
    leaked = [c for c in FEATURE_COLUMNS if c in LABEL_COLUMNS]
    if leaked:
        raise ValueError(f"post-game columns present in FEATURE_COLUMNS: {leaked}")

    if NOT_A_TARGET in FEATURE_COLUMNS:
        raise ValueError(
            f"{NOT_A_TARGET} is a post-game outcome and must never be a feature."
        )

    for target in TARGETS:
        if target in FEATURE_COLUMNS:
            raise ValueError(f"target {target} is also listed as a feature.")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    print(f"Loaded {len(df):,} player-games from {DATASET_PATH.name}")
    return df


def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows LinearRegression cannot consume, and say why separately.

    Two distinct causes, reported apart rather than as one number: a
    player's early-season warm-up, and the handful of team-games with no
    REST_DAYS because they are a team's first appearance in the data. Same
    reasoning as the merge checks - "missing" is not one thing.
    """
    before = len(df)

    warmup = df[PLAYER_FEATURE_COLUMNS].isna().any(axis=1)
    context = df[TEAM_CONTEXT_COLUMNS].isna().any(axis=1)

    print(f"\n  incomplete player rolling windows : {int(warmup.sum()):>7,}")
    print(f"  missing team context (REST_DAYS)  : {int(context.sum()):>7,}")
    print(f"  overlap                           : {int((warmup & context).sum()):>7,}")

    kept = df[~(warmup | context)].reset_index(drop=True)
    dropped = before - len(kept)
    print(f"\nDropped {dropped:,} of {before:,} rows ({dropped / before:.1%}), "
          f"leaving {len(kept):,}.")

    residual = kept[FEATURE_COLUMNS].isna().sum()
    residual = residual[residual > 0]
    if not residual.empty:
        raise ValueError(f"NaN remains in the feature matrix:\n{residual}")

    return kept


def split_by_season(df: pd.DataFrame) -> tuple:
    """Chronological hold-out, identical boundary to the team models.

    The last TEST_SEASON_COUNT seasons are the test set. Splitting by time
    rather than at random is what stops a player's future games teaching
    the model about his own past ones.
    """
    seasons = sorted(df["SEASON"].unique())
    test_seasons = seasons[-TEST_SEASON_COUNT:]

    train = df[~df["SEASON"].isin(test_seasons)].reset_index(drop=True)
    test = df[df["SEASON"].isin(test_seasons)].reset_index(drop=True)

    train_seasons = seasons[:-TEST_SEASON_COUNT]
    print(f"\nTrain: seasons {train_seasons[0]}-{train_seasons[-1]} "
          f"({len(train_seasons)} seasons, {len(train):,} player-games)")
    print(f"Test:  seasons {test_seasons[0]}-{test_seasons[-1]} "
          f"({len(test_seasons)} seasons, {len(test):,} player-games)")

    return train, test, test_seasons


def scale_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Standardise, fitting on train only.

    Fitting on everything would let the test seasons' distribution leak
    into the transform - small here, but the same principle the team
    baselines follow.
    """
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train[FEATURE_COLUMNS])
    test_x = scaler.transform(test[FEATURE_COLUMNS])
    print(f"Standardized {len(FEATURE_COLUMNS)} features "
          f"(scaler fit on train only).")
    return train_x, test_x


def evaluate(target: str, train: pd.DataFrame, test: pd.DataFrame,
             train_x, test_x) -> dict:
    section(f"{TARGET_LABELS[target].upper()} (target: {target})")

    naive_column = f"ROLL10_{target}"
    naive_mae = mean_absolute_error(test[target], test[naive_column])

    model = LinearRegression()
    model.fit(train_x, train[target])
    linear_mae = mean_absolute_error(test[target], model.predict(test_x))

    change = (linear_mae - naive_mae) / naive_mae * 100

    print(f"Naive: {naive_column:<14} MAE {naive_mae:7.3f}")
    print(f"LinearRegression              MAE {linear_mae:7.3f}   "
          f"{change:+.1f}% vs naive")
    print(f"Test-set mean {target}: {test[target].mean():.2f}  "
          f"(MAE in context)")

    return {"target": target, "naive": naive_mae,
            "linear": linear_mae, "change": change}


def main():
    check_no_leakage()

    section("DATA PREP")
    df = load_dataset()
    df = drop_incomplete(df)
    train, test, test_seasons = split_by_season(df)
    train_x, test_x = scale_features(train, test)

    results = [evaluate(t, train, test, train_x, test_x) for t in TARGETS]

    section(f"SUMMARY (test: seasons {test_seasons[0]}-{test_seasons[-1]}, "
            f"{len(test):,} player-games)")
    print(f"{'TARGET':<16}{'NAIVE (ROLL10)':>16}{'LINEAR':>10}{'CHANGE':>10}")
    print("-" * 52)
    for row in results:
        print(f"{TARGET_LABELS[row['target']]:<16}{row['naive']:>16.3f}"
              f"{row['linear']:>10.3f}{row['change']:>9.1f}%")

    print(f"\n{NOT_A_TARGET} deliberately not modelled here - predicting playing "
          f"time is a\ndifferent problem and deserves its own pass.")


if __name__ == "__main__":
    main()
