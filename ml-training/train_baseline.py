"""
Baseline models for all three prediction targets: moneyline, spread, totals.

Input:  data-pipeline/data/processed/model_dataset.csv
Output: printed metrics only. Nothing is saved — these are reference numbers,
        not deployable models.

Every target gets a naive baseline before an ML one, so anything trained
later has a bar to clear. A logistic regression that can't beat "the home
team always wins" isn't worth shipping, and without this table there'd be
no way to know.

Two data decisions live here rather than in the pipeline, because they are
model requirements rather than properties of the data:

  - Early-season rows with incomplete rolling windows are DROPPED. Linear
    and logistic regression cannot ingest NaN. The gradient-boosting model
    coming later can, and should revisit this rather than inherit it.
  - Features are standardized with a scaler FIT ON THE TRAINING SET ONLY.
    These features span wildly different ranges (TEAM_ELO ~1500, FG_PCT
    ~0.45, REST_DAYS 0-7), which regression is sensitive to. Fitting the
    scaler on the full dataset would leak test-set statistics into training.
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

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data-pipeline"
    / "data"
    / "processed"
    / "model_dataset.csv"
)

# The most recent N seasons are held out. Splitting chronologically rather
# than randomly is the only honest option: a random split would train on
# future games to predict past ones.
TEST_SEASON_COUNT = 2

# The 17 pre-game features available for each side. model_dataset.csv carries
# each one twice, prefixed HOME_ and AWAY_, for 34 model inputs in total.
PER_SIDE_FEATURES = [
    "ROLL5_WIN_PCT",
    "ROLL5_PTS",
    "ROLL5_PLUS_MINUS",
    "ROLL5_FG_PCT",
    "ROLL5_REB",
    "ROLL5_AST",
    "ROLL5_TOV",
    "ROLL10_WIN_PCT",
    "ROLL10_PTS",
    "ROLL10_PLUS_MINUS",
    "ROLL10_FG_PCT",
    "ROLL10_REB",
    "ROLL10_AST",
    "ROLL10_TOV",
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
]

FEATURE_COLUMNS = [f"{side}_{feat}" for side in ("HOME", "AWAY") for feat in PER_SIDE_FEATURES]

ROLLING_FEATURE_COLUMNS = [
    c for c in FEATURE_COLUMNS if "ROLL5_" in c or "ROLL10_" in c
]

BOOLEAN_FEATURE_COLUMNS = ["HOME_IS_BACK_TO_BACK", "AWAY_IS_BACK_TO_BACK"]

# Identifiers and post-game outcomes. Everything else in the file is a
# feature — asserted at load time so the two lists can't drift apart.
ID_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "HOME_TEAM_ID",
    "HOME_TEAM_NAME",
    "AWAY_TEAM_ID",
    "AWAY_TEAM_NAME",
]
# Kept in sync with LABEL_COLUMNS in build_final_dataset.py. The REB/AST
# entries are raw single-game outcomes — unrelated to the ROLL5_/ROLL10_
# REB and AST features, which are trailing averages of prior games.
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

METRIC_PRECISION = {"Accuracy": 4, "Log loss": 4, "MAE": 2, "RMSE": 2}

# Every regression target follows one shape: a naive baseline built from the
# two teams' own ROLL10 average of that same stat — differenced for margins,
# summed for totals — then a LinearRegression on the full feature set.
#
# The target column is listed explicitly rather than derived from the stat
# because HOME_MARGIN breaks the {STAT}_MARGIN convention the others follow.
REGRESSION_TARGETS = [
    # (target column, table label, ROLL10 stat, how to combine)
    ("HOME_MARGIN", "Spread", "PTS", "diff"),
    ("TOTAL_PTS", "Totals", "PTS", "sum"),
    ("REB_MARGIN", "REB margin", "REB", "diff"),
    ("TOTAL_REB", "REB total", "REB", "sum"),
    ("AST_MARGIN", "AST margin", "AST", "diff"),
    ("TOTAL_AST", "AST total", "AST", "sum"),
]


def derive_season(game_date: pd.Series) -> pd.Series:
    """Season label = the calendar year it tipped off in, August boundary.

    Mirrors data-pipeline/preprocessing/build_rolling_features.py.
    """
    return game_date.dt.year.where(game_date.dt.month >= 8, game_date.dt.year - 1)


def elo_expected_score(rating: pd.Series, opponent_rating: pd.Series) -> pd.Series:
    """Closed-form Elo win probability, same formula as build_elo_ratings.py.

    No training involved — this is what the rating gap already implies.
    """
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON"] = derive_season(df["GAME_DATE"])

    # Regression needs numbers, not True/False.
    for col in BOOLEAN_FEATURE_COLUMNS:
        df[col] = df[col].astype(int)

    unaccounted = set(df.columns) - set(FEATURE_COLUMNS) - set(ID_COLUMNS) - set(LABEL_COLUMNS) - {"SEASON"}
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

    # Tier 1 - naive: home team always wins.
    naive_pred = np.ones(len(y_test), dtype=int)
    naive_acc = accuracy_score(y_test, naive_pred)
    print(f"Naive (always home)  accuracy {naive_acc:.4f}  <- the test set's home-win rate")
    results.append(("Moneyline", "Naive: always home", [("Accuracy", naive_acc)]))

    # Tier 2 - informed, untrained: what the Elo gap already implies.
    elo_prob = elo_expected_score(test["HOME_TEAM_ELO"], test["AWAY_TEAM_ELO"])
    elo_acc = accuracy_score(y_test, (elo_prob > 0.5).astype(int))
    elo_loss = log_loss(y_test, elo_prob)
    print(f"Elo win probability  accuracy {elo_acc:.4f}  log loss {elo_loss:.4f}")
    results.append(("Moneyline", "Elo win probability", [("Accuracy", elo_acc), ("Log loss", elo_loss)]))

    # Tier 3 - simple ML.
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    lr_acc = accuracy_score(y_test, model.predict(x_test))
    lr_loss = log_loss(y_test, proba)
    print(f"LogisticRegression   accuracy {lr_acc:.4f}  log loss {lr_loss:.4f}")
    results.append(("Moneyline", "LogisticRegression", [("Accuracy", lr_acc), ("Log loss", lr_loss)]))

    return results


def naive_prediction(test: pd.DataFrame, stat: str, combine: str) -> pd.Series:
    """Each team's own recent scoring/rebounding/assist average, combined.

    Varies per matchup rather than being one flat league-wide constant,
    which makes it a genuinely harder baseline for the model to beat.
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

    # Second metric column is unlabelled — each cell names its own metric,
    # since they differ between classification and regression targets.
    headers = ("TARGET", "METHOD", "METRIC", "")
    widths = [max(len(str(r[i])) for r in (*rows, headers)) for i in range(4)]

    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line.rstrip())
    print("-" * len(line))

    previous_target = None
    for row in rows:
        # Blank line between targets so the tiers group visually.
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
