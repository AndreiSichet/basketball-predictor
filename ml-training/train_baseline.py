"""
Baseline models for moneyline, spread, totals, rebounds and assists.

Input:  data-pipeline/data/processed/model_dataset.csv
Output: printed metrics only, nothing saved.

Each target gets a naive baseline before an ML one, so later models have a
bar to clear.

Two choices made here rather than in the pipeline, because they are model
requirements, not properties of the data:
  - Rows with incomplete rolling windows are dropped; linear and logistic
    regression can't take NaN. XGBoost later can, and doesn't drop them.
  - Features are standardized with the scaler fit on train only. Fitting on
    everything would leak test statistics into training.
"""

from pathlib import Path

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

# Defined in common.py because inference needs them too. Re-exported here
# so existing imports from this module keep working.
from common import FEATURE_COLUMNS, ROLLING_FEATURE_COLUMNS

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data-pipeline"
    / "data"
    / "processed"
    / "model_dataset.csv"
)

# Held out as test. Chronological, not random: a random split would train on
# future games to predict past ones.
TEST_SEASON_COUNT = 2

BOOLEAN_FEATURE_COLUMNS = ["HOME_IS_BACK_TO_BACK", "AWAY_IS_BACK_TO_BACK"]

# Present in the dataset but deliberately not trained on. Empty right now:
# the availability columns that used to sit here are in FEATURE_COLUMNS as
# of the live-injury-report work, so listing them would tell the guard below
# to ignore columns that are genuinely in use - defeating its purpose.
# The mechanism stays for the next feature that needs to land in the dataset
# before it lands in the model - which is exactly what the rating columns
# below are: all 20 advanced-stat columns are present in model_dataset.csv
# and deliberately held out of FEATURE_COLUMNS. Two experiments (the full
# 5-metric bundle, then a PACE/TS_PCT-only subset) showed no gain over the
# 38-feature set, so the data stays built but unused. See CLAUDE.md.
UNUSED_FEATURE_COLUMNS = [
    f"{side}_{window}_{metric}"
    for side in ("HOME", "AWAY")
    for window in ("ROLL5", "ROLL10")
    for metric in ("OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "TS_PCT")
]

# The Q1/first-half rolling columns, held out for the same reason and by the
# same mechanism. They are built, validated and merged, but no model here
# consumes them: whether trailing quarter form earns a place in
# FEATURE_COLUMNS is what train_quarter_half_baseline.py exists to measure,
# and this project decides that with evidence rather than in advance.
#
# Held out of the SHARED list is not the same as unavailable: that script
# reads these columns straight out of model_dataset.csv by name, which is
# exactly why they have to be merged in even while unused here. Adding them
# to FEATURE_COLUMNS instead would silently change all 7 shipped models.
UNUSED_FEATURE_COLUMNS += [
    f"{side}_{window}_{metric}"
    for side in ("HOME", "AWAY")
    for window in ("ROLL5", "ROLL10")
    for metric in ("Q1_MARGIN", "Q1_PTS", "Q1_PTS_ALLOWED",
                   "HALF1_MARGIN", "HALF1_PTS", "HALF1_PTS_ALLOWED")
]

# Ids and post-game outcomes. Everything else in the file must be a feature;
# load_dataset() checks this.
ID_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "HOME_TEAM_ID",
    "HOME_TEAM_NAME",
    "AWAY_TEAM_ID",
    "AWAY_TEAM_NAME",
]
# Mirrors LABEL_COLUMNS in build_final_dataset.py. These REB/AST entries are
# raw single-game results, not the ROLL5_/ROLL10_ averages, which are features.
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
    # Q1 and 1H markets. Labels only - no model trains on them yet, but they
    # must be classified here or load_dataset()'s guard rejects the file for
    # every script that shares it, this one included.
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

METRIC_PRECISION = {"Accuracy": 4, "Log loss": 4, "MAE": 2, "RMSE": 2}

# (target column, table label, ROLL10 stat, how to combine). The target
# column is listed rather than derived because HOME_MARGIN doesn't follow
# the {STAT}_MARGIN naming the others use.
REGRESSION_TARGETS = [
    ("HOME_MARGIN", "Spread", "PTS", "diff"),
    ("TOTAL_PTS", "Totals", "PTS", "sum"),
    ("REB_MARGIN", "REB margin", "REB", "diff"),
    ("TOTAL_REB", "REB total", "REB", "sum"),
    ("AST_MARGIN", "AST margin", "AST", "diff"),
    ("TOTAL_AST", "AST total", "AST", "sum"),
]


def derive_season(game_date: pd.Series) -> pd.Series:
    """Season = year it tipped off in, August boundary.

    Same rule as build_rolling_features.py.
    """
    return game_date.dt.year.where(game_date.dt.month >= 8, game_date.dt.year - 1)


def elo_expected_score(rating: pd.Series, opponent_rating: pd.Series) -> pd.Series:
    """Elo win probability. Same formula as build_elo_ratings.py, no training."""
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = derive_season(df["GAME_DATE"])

    # Models need numbers, not True/False.
    for col in BOOLEAN_FEATURE_COLUMNS:
        df[col] = df[col].astype(int)

    unaccounted = (
        set(df.columns)
        - set(FEATURE_COLUMNS)
        - set(ID_COLUMNS)
        - set(LABEL_COLUMNS)
        - set(UNUSED_FEATURE_COLUMNS)
        - {"SEASON"}
    )
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if unaccounted or missing:
        raise ValueError(
            "FEATURE_COLUMNS is out of sync with the dataset.\n"
            f"  Columns in the file but unclassified: {sorted(unaccounted)}\n"
            f"  Features expected but absent: {sorted(missing)}"
        )

    return df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def drop_incomplete_windows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=ROLLING_FEATURE_COLUMNS).reset_index(drop=True)
    dropped = before - len(df)

    print(
        f"Dropped {dropped} of {before} rows ({dropped / before:.1%}) with incomplete "
        f"rolling windows, leaving {len(df)}."
    )

    residual = df[FEATURE_COLUMNS].isna().sum()
    residual = residual[residual > 0]
    if not residual.empty:
        raise ValueError(f"NaN remains in the feature matrix after the drop:\n{residual}")

    return df


def split_by_season(df: pd.DataFrame) -> tuple:
    seasons = sorted(df["SEASON"].unique())
    if len(seasons) <= TEST_SEASON_COUNT:
        raise ValueError(
            f"Need more than {TEST_SEASON_COUNT} seasons to hold out "
            f"{TEST_SEASON_COUNT}; found {len(seasons)}."
        )

    test_seasons = seasons[-TEST_SEASON_COUNT:]
    train = df[~df["SEASON"].isin(test_seasons)].reset_index(drop=True)
    test = df[df["SEASON"].isin(test_seasons)].reset_index(drop=True)

    train_seasons = seasons[:-TEST_SEASON_COUNT]
    print(
        f"Train: seasons {train_seasons[0]}-{train_seasons[-1]} "
        f"({len(train_seasons)} seasons, {len(train)} games)"
    )
    print(
        f"Test:  seasons {test_seasons[0]}-{test_seasons[-1]} "
        f"({len(test_seasons)} seasons, {len(test)} games)"
    )

    return train, test, test_seasons


def scale_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    x_test = scaler.transform(test[FEATURE_COLUMNS])
    print(f"Standardized {len(FEATURE_COLUMNS)} features (scaler fit on train only).")
    return x_train, x_test


def evaluate_moneyline(train, test, x_train, x_test) -> list:
    section("MONEYLINE (target: HOME_WIN)")
    y_train, y_test = train["HOME_WIN"], test["HOME_WIN"]
    results = []

    # Tier 1: home team always wins.
    naive_pred = np.ones(len(y_test), dtype=int)
    naive_acc = accuracy_score(y_test, naive_pred)
    print(f"Naive (always home)  accuracy {naive_acc:.4f}  <- the test set's home-win rate")
    results.append(("Moneyline", "Naive: always home", [("Accuracy", naive_acc)]))

    # Tier 2: what the Elo gap implies, untrained.
    elo_prob = elo_expected_score(test["HOME_TEAM_ELO"], test["AWAY_TEAM_ELO"])
    elo_acc = accuracy_score(y_test, (elo_prob > 0.5).astype(int))
    elo_loss = log_loss(y_test, elo_prob)
    print(f"Elo win probability  accuracy {elo_acc:.4f}  log loss {elo_loss:.4f}")
    results.append(("Moneyline", "Elo win probability", [("Accuracy", elo_acc), ("Log loss", elo_loss)]))

    # Tier 3: simple ML.
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    lr_acc = accuracy_score(y_test, model.predict(x_test))
    lr_loss = log_loss(y_test, proba)
    print(f"LogisticRegression   accuracy {lr_acc:.4f}  log loss {lr_loss:.4f}")
    results.append(("Moneyline", "LogisticRegression", [("Accuracy", lr_acc), ("Log loss", lr_loss)]))

    return results


def naive_prediction(test: pd.DataFrame, stat: str, combine: str) -> pd.Series:
    """Each team's own ROLL10 average, differenced or summed.

    Varies per matchup instead of being one league-wide constant.
    """
    home, away = test[f"HOME_ROLL10_{stat}"], test[f"AWAY_ROLL10_{stat}"]
    return home - away if combine == "diff" else home + away


def evaluate_regression_target(
    target: str, label: str, stat: str, combine: str, train, test, x_train, x_test
) -> list:
    section(f"{label.upper()} (target: {target})")
    y_train, y_test = train[target], test[target]
    results = []

    naive_name = f"Naive: ROLL10 {stat} {combine}"
    naive_pred = naive_prediction(test, stat, combine)
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    print(f"{naive_name:<28} MAE {naive_mae:6.2f}  RMSE {naive_rmse:6.2f}")
    results.append((label, naive_name, [("MAE", naive_mae), ("RMSE", naive_rmse)]))

    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{'LinearRegression':<28} MAE {mae:6.2f}  RMSE {rmse:6.2f}")
    results.append((label, "LinearRegression", [("MAE", mae), ("RMSE", rmse)]))

    return results


def format_metric(name: str, value: float) -> str:
    return f"{name} {value:.{METRIC_PRECISION[name]}f}"


def print_summary(results: list, test_seasons: list, test_rows: int) -> None:
    section(f"SUMMARY (test set: seasons {test_seasons[0]}-{test_seasons[-1]}, {test_rows} games)")

    rows = []
    for target, method, metrics in results:
        cells = [format_metric(name, value) for name, value in metrics]
        cells += ["-"] * (2 - len(cells))
        rows.append((target, method, cells[0], cells[1]))

    # Second metric column has no header: each cell names its own metric,
    # which differs between classification and regression.
    headers = ("TARGET", "METHOD", "METRIC", "")
    widths = [max(len(str(r[i])) for r in (*rows, headers)) for i in range(4)]

    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line.rstrip())
    print("-" * len(line))

    previous_target = None
    for row in rows:
        if previous_target is not None and row[0] != previous_target:
            print()
        previous_target = row[0]
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)).rstrip())


def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df)} games from {DATASET_PATH.name}")

    df = drop_incomplete_windows(df)
    train, test, test_seasons = split_by_season(df)
    x_train, x_test = scale_features(train, test)

    results = evaluate_moneyline(train, test, x_train, x_test)

    for target, label, stat, combine in REGRESSION_TARGETS:
        results += evaluate_regression_target(
            target, label, stat, combine, train, test, x_train, x_test
        )

    print_summary(results, test_seasons, len(test))


if __name__ == "__main__":
    main()
