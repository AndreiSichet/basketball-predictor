"""
Baselines for the Q1 and first-half markets.

  input: data-pipeline/data/processed/model_dataset.csv
  output: printed metrics only, nothing saved.

Six targets: spread, total and winner for each of the first quarter and the
first half. Same two-tier shape as every prior baseline here - a naive
construction first, then a linear model - so anything more expensive has a
bar to clear rather than a vacuum to look good in.

THE FEATURE SET IS LOCAL TO THIS SCRIPT, deliberately. common.py's
FEATURE_COLUMNS drives the seven shipped team-level models and the live
inference service; this is a different modelling problem with a different
input set, and folding it in would silently change what those models train
on. The 24 quarter/half rolling columns are merged into model_dataset.csv
but sit in train_baseline.UNUSED_FEATURE_COLUMNS, so the shared guard still
accounts for them while nothing shared consumes them. Reading them here by
name is exactly what that arrangement is for.

SIX CONTEXT COLUMNS ARE REUSED rather than rebuilt: both Elo ratings, both
rest-day counts and both back-to-back flags. Already computed, already
verified, and there is a real basketball reason each should carry signal
even over twelve minutes - a rested team against a tired one, or a far
stronger team against a weaker one, does not wait until the fourth quarter
to show it. Same reasoning that justified reusing team context for the
player-prop table.

TWO ROW FILTERS, FOR TWO DIFFERENT REASONS, reported separately because
conflating them would hide either one:

  1. Incomplete rolling window. Linear and logistic regression cannot take
     NaN, so a row whose trailing Q1/1H form is not yet complete is dropped -
     the same rule the team-level baseline applies, and the same reason.
     This also removes the three games with no quarter data at all, plus the
     games that follow them for one window's length.

  2. Tied period, CLASSIFICATION ONLY. 611 first quarters and 431 first
     halves ended level, and build_final_dataset.py records those as NA
     rather than asserting a winner nobody was. A binary classifier has no
     honest label for them, so they are dropped from the two winner targets.
     They stay in the four regression targets: a margin of 0 and a real
     total are perfectly good data for a tied quarter, and dropping them
     there would throw away the games nearest the decision boundary.

     Note what this does NOT do - it does not alter the shipped dataset. The
     rows keep their real NA label in model_dataset.csv; only this
     classifier's training and test sets exclude them. A sportsbook treats a
     tied period as a push, so the two-way market being modelled here is
     genuinely "conditional on a decision being reached", and the reported
     accuracy should be read that way.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

# Everything shared comes from the script that owns it - the dataset loader
# with its classification guard, the season split, the naive construction
# and the summary formatting. Nothing here is a second copy.
from train_baseline import (
    format_metric,
    load_dataset,
    naive_prediction,
    print_summary,
    section,
    split_by_season,
)

PERIODS = ["Q1", "HALF1"]
ROLLING_METRICS = ["MARGIN", "PTS", "PTS_ALLOWED"]

# The 24 columns built by build_quarter_half_rolling.py.
QUARTER_HALF_FEATURES = [
    f"{side}_{window}_{period}_{metric}"
    for side in ("HOME", "AWAY")
    for window in ("ROLL5", "ROLL10")
    for period in PERIODS
    for metric in ROLLING_METRICS
]

# Already built, already verified, cheap to include. See the docstring.
CONTEXT_FEATURES = [
    "HOME_TEAM_ELO",
    "AWAY_TEAM_ELO",
    "HOME_REST_DAYS",
    "AWAY_REST_DAYS",
    "HOME_IS_BACK_TO_BACK",
    "AWAY_IS_BACK_TO_BACK",
]

FEATURE_COLUMNS = QUARTER_HALF_FEATURES + CONTEXT_FEATURES

# (target, table label, ROLL10 stat, how to combine). The naive construction
# mirrors the team-level one exactly: each side's own trailing scoring
# average, differenced for a margin and summed for a total. Deliberately NOT
# the already-netted ROLL10_*_MARGIN column - the point of a naive baseline
# is to be the obvious thing, and the obvious thing is the same shape that
# has been used for spread and totals since the first baseline script.
REGRESSION_TARGETS = [
    ("HOME_Q1_MARGIN", "Q1 spread", "Q1_PTS", "diff"),
    ("TOTAL_Q1_PTS", "Q1 total", "Q1_PTS", "sum"),
    ("HOME_HALF1_MARGIN", "1H spread", "HALF1_PTS", "diff"),
    ("TOTAL_HALF1_PTS", "1H total", "HALF1_PTS", "sum"),
]

CLASSIFICATION_TARGETS = [
    ("HOME_Q1_WIN", "Q1 winner", "Q1"),
    ("HOME_HALF1_WIN", "1H winner", "HALF1"),
]


def drop_incomplete_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter 1. Rows whose trailing Q1/1H form is not yet complete."""
    before = len(df)
    kept = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    dropped = before - len(kept)

    print(f"\nFilter 1 - incomplete rolling window (linear models cannot take NaN)")
    print(f"  dropped {dropped:,} of {before:,} rows ({dropped / before:.1%}), "
          f"leaving {len(kept):,}.")
    print(f"  includes the 3 games with no quarter data and the games that "
          f"follow them within one window.")

    residual = kept[FEATURE_COLUMNS].isna().sum()
    residual = residual[residual > 0]
    if not residual.empty:
        raise ValueError(f"NaN remains in the feature matrix:\n{residual}")

    return kept


def report_ties(df: pd.DataFrame) -> None:
    """Filter 2, quantified before it is applied. Classification only."""
    print(f"\nFilter 2 - tied period (the two winner targets only)")
    for period in PERIODS:
        tied = int((df[f"HOME_{period}_PTS"] == df[f"AWAY_{period}_PTS"]).sum())
        print(f"  {period:<6} {tied:,} of {len(df):,} rows tied ({tied / len(df):.1%}) "
              f"- no honest binary label, dropped from HOME_{period}_WIN only.")
    print("  All four regression targets keep these rows: a 0 margin and a "
          "real total are valid data.")


def scale(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Standardize, scaler fit on train only - never on the full frame."""
    scaler = StandardScaler()
    return (scaler.fit_transform(train[FEATURE_COLUMNS]),
            scaler.transform(test[FEATURE_COLUMNS]))


def evaluate_regression(target, label, stat, combine, train, test) -> list:
    section(f"{label.upper()} (target: {target})")

    # The three no-data games survive filter 1 only if their own windows are
    # complete, so the target itself is checked too.
    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])
    x_train, x_test = scale(train, test)
    y_train, y_test = train[target], test[target]

    results = []
    naive_name = f"Naive: ROLL10 {stat} {combine}"
    naive_pred = naive_prediction(test, stat, combine)
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    print(f"{naive_name:<30} MAE {naive_mae:6.2f}  RMSE {naive_rmse:6.2f}")
    results.append((label, naive_name, [("MAE", naive_mae), ("RMSE", naive_rmse)]))

    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{'LinearRegression':<30} MAE {mae:6.2f}  RMSE {rmse:6.2f}")
    results.append((label, "LinearRegression", [("MAE", mae), ("RMSE", rmse)]))

    print(f"  ({len(train):,} train / {len(test):,} test rows)")
    return results


def evaluate_classification(target, label, period, train, test) -> list:
    section(f"{label.upper()} (target: {target})")

    # This is filter 2 taking effect: a tied period carries NA, so dropna
    # removes exactly the ties plus any no-data row still present.
    before_train, before_test = len(train), len(test)
    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])
    print(f"Dropped {before_train - len(train):,} train and "
          f"{before_test - len(test):,} test rows with no decided winner "
          f"(tied {period}).")

    x_train, x_test = scale(train, test)
    y_train = train[target].astype(int)
    y_test = test[target].astype(int)

    results = []
    # Naive: always home. Its accuracy IS the test set's home-win rate for
    # this period, measured after ties are excluded.
    naive_acc = accuracy_score(y_test, np.ones(len(y_test), dtype=int))
    print(f"{'Naive: always home':<30} accuracy {naive_acc:.4f}  "
          f"<- the tie-excluded home rate")
    results.append((label, "Naive: always home", [("Accuracy", naive_acc)]))

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, model.predict(x_test))
    loss = log_loss(y_test, proba)
    print(f"{'LogisticRegression':<30} accuracy {acc:.4f}  log loss {loss:.4f}")
    results.append((label, "LogisticRegression",
                    [("Accuracy", acc), ("Log loss", loss)]))

    print(f"  ({len(train):,} train / {len(test):,} test rows)")
    return results


def main():
    section("DATA PREP")
    # Shared loader: applies the column-classification guard, converts the
    # boolean features to int and derives SEASON. Reused, not reimplemented.
    df = load_dataset()
    print(f"Loaded {len(df):,} games from model_dataset.csv")
    print(f"Feature set: {len(FEATURE_COLUMNS)} columns "
          f"({len(QUARTER_HALF_FEATURES)} quarter/half rolling + "
          f"{len(CONTEXT_FEATURES)} reused context)")
    print("  held out of common.FEATURE_COLUMNS - the 7 shipped models are "
          "untouched by this script.")

    df = drop_incomplete_windows(df)
    report_ties(df)

    section("SPLIT")
    train, test, test_seasons = split_by_season(df)

    results = []
    for target, label, stat, combine in REGRESSION_TARGETS:
        results += evaluate_regression(target, label, stat, combine, train, test)
    for target, label, period in CLASSIFICATION_TARGETS:
        results += evaluate_classification(target, label, period, train, test)

    print_summary(results, test_seasons, len(test))

    section("READ THIS BEFORE COMPARING TO THE TEAM-LEVEL TABLE")
    print("These MAEs are NOT comparable to the full-game spread/total numbers.")
    print("A quarter is a smaller quantity with less to be wrong about, so a")
    print("lower MAE here is arithmetic, not skill. The only meaningful")
    print("comparison is naive vs LinearRegression within each row above.")
    print()
    print("The two winner accuracies are conditional on the period being")
    print("decided - tied periods are excluded, which a sportsbook would push.")


if __name__ == "__main__":
    main()
