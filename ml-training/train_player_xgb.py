"""
XGBoost for the five player-prop targets, against the baselines already
established in train_player_baseline.py.

  input: data-pipeline/data/processed/player_dataset.csv  (280,943 rows)

THE BAR IS HARDER HERE THAN IT WAS AT TEAM LEVEL, and that is worth saying
before any number appears. At team level, LinearRegression beat its naive
baselines by double-digit percentages, so XGBoost had a wide corridor to
compete in. Here naive and linear sit within about 1% of each other on
every target - a player's own trailing average is already close to the
whole story. Clearing the same >3-5% MAE bar against that is a much
steeper ask, and a 1% win should be read as noise, not progress.

TRAINS ON EVERY ROW, unlike the baseline. LinearRegression cannot take NaN,
so the baselines dropped every player-game without a complete rolling
window. XGBoost learns a default split direction for missing values, so it
trains on all 280,943 rows - roughly 55,000 more than the baselines saw.
Same deliberate choice as team-level training on all 13,199 games rather
than the filtered 11,465. The run prints the difference, so the larger
training set reads as intentional rather than as an oversight.

SCORED TWO WAYS, for the same reason. The three-way table is scored only on
test rows the baselines could also score, so naive, linear and XGBoost are
compared on identical games. XGBoost's number over the full test set is
reported separately - it is a genuine capability the others do not have,
but it is not a like-for-like comparison and is never presented as one.

BASELINES ARE RECOMPUTED HERE, not copied from the other script's output.
That makes the table internally consistent and doubles as a drift check:
if these numbers disagree with train_player_baseline.py's, something moved
between the two runs.

CONFIG IS REUSED, NOT RE-TUNED. max_depth=4 / learning_rate=0.05 comes
straight from the team-level work, where a 2x2 sweep across seven targets
improved nothing meaningfully. Re-running a sweep here would cost hours to
most likely rediscover that. If a player target shows real promise, tuning
it then is the cheap, evidence-led order.

WATCH FG3M SEPARATELY. It is the one target where LinearRegression came out
*worse* than the naive rolling average, and the hypothesis is that the five
team-context features are close to noise for a specific player's three-point
volume. If early stopping also converges unusually fast there, that is the
same "tree count collapse means added variance, not added signal" signature
the rejected advanced-stats experiment produced - independent, mechanism-
level support rather than the same bad number twice. Its feature importance
is printed in full for that reason.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import setup_mlflow, split_three_way

# Column groups and the leakage guard come from the scripts that own them.
PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
sys.path.insert(0, str(PIPELINE_DIR))
from build_player_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    OUTPUT_PATH as DATASET_PATH,
    PLAYER_FEATURE_COLUMNS,
    TEAM_CONTEXT_COLUMNS,
)

from train_player_baseline import (  # noqa: E402
    TARGET_LABELS,
    TARGETS,
    check_no_leakage,
)

# Same starting point as every team-level XGBoost run here.
PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "early_stopping_rounds": 50,
    "random_state": 42,
}

# The bar every regression target in this project has been held to.
IMPROVEMENT_BAR_PCT = 3.0

# Printed in full rather than top-N, because the hypothesis is about which
# features are *not* contributing.
FLAG_TARGET = "FG3M"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def complete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows a linear model could also use - no NaN anywhere in the inputs."""
    return df.dropna(subset=FEATURE_COLUMNS)


def fit_baselines(train: pd.DataFrame, validation: pd.DataFrame,
                  test_comparable: pd.DataFrame, target: str) -> tuple:
    """Naive and LinearRegression on exactly the rows the baseline used.

    Train and validation are combined: the baseline script had no
    validation split, so its training set was everything before the test
    seasons. Matching that is what makes the comparison honest.
    """
    fit_set = complete_rows(pd.concat([train, validation]))

    scaler = StandardScaler()
    fit_x = scaler.fit_transform(fit_set[FEATURE_COLUMNS])
    test_x = scaler.transform(test_comparable[FEATURE_COLUMNS])

    model = LinearRegression()
    model.fit(fit_x, fit_set[target])

    naive = mean_absolute_error(test_comparable[target],
                                test_comparable[f"ROLL10_{target}"])
    linear = mean_absolute_error(test_comparable[target], model.predict(test_x))
    return naive, linear


def train_target(target: str, train: pd.DataFrame, validation: pd.DataFrame,
                 test: pd.DataFrame, test_comparable: pd.DataFrame) -> dict:
    section(f"{TARGET_LABELS[target].upper()} (target: {target})")

    model = XGBRegressor(**PARAMS)
    model.fit(
        train[FEATURE_COLUMNS], train[target],
        eval_set=[(validation[FEATURE_COLUMNS], validation[target])],
        verbose=False,
    )
    trees = int(model.best_iteration)
    print(f"Early stopping at iteration {trees} of {PARAMS['n_estimators']}.")

    xgb_comparable = mean_absolute_error(
        test_comparable[target], model.predict(test_comparable[FEATURE_COLUMNS]))
    xgb_full = mean_absolute_error(test[target], model.predict(test[FEATURE_COLUMNS]))

    naive, linear = fit_baselines(train, validation, test_comparable, target)
    change = (xgb_comparable - linear) / linear * 100
    verdict = ("REAL GAIN" if change <= -IMPROVEMENT_BAR_PCT
               else "no meaningful gain")

    print(f"\nMETHOD                          MAE   vs LINEAR")
    print(f"{'-' * 46}")
    print(f"Naive: ROLL10_{target:<16}{naive:7.3f}")
    print(f"LinearRegression            {linear:7.3f}          -")
    print(f"XGBoost                     {xgb_comparable:7.3f}   {change:+8.1f}%")
    print(f"\nXGBoost on the full test set (incl. incomplete-window rows): "
          f"{xgb_full:.3f}")
    print("  No baseline counterpart - neither naive nor linear can score those rows.")

    importance = pd.Series(
        model.get_booster().get_score(importance_type="gain")
    ).sort_values(ascending=False)
    importance = importance / importance.sum() * 100

    if target == FLAG_TARGET:
        print(f"\n  {FLAG_TARGET} feature importance, in full "
              f"(% of total gain) - see the module docstring:")
        for name, value in importance.items():
            marker = "  <-- team context" if name in TEAM_CONTEXT_COLUMNS else ""
            print(f"    {name:<22}{value:6.2f}%{marker}")
        context_share = importance[
            [c for c in importance.index if c in TEAM_CONTEXT_COLUMNS]
        ].sum()
        print(f"    team-context share of total gain: {context_share:.2f}%")
    else:
        print("\n  top 5 by gain: " + ", ".join(
            f"{n} {v:.1f}%" for n, v in importance.head(5).items()))

    log_run(target, trees, naive, linear, xgb_comparable, xgb_full, change)

    return {"target": target, "naive": naive, "linear": linear,
            "xgb": xgb_comparable, "xgb_full": xgb_full,
            "change": change, "trees": trees, "verdict": verdict,
            "context_share": float(importance[
                [c for c in importance.index if c in TEAM_CONTEXT_COLUMNS]
            ].sum())}


def log_run(target, trees, naive, linear, xgb_comparable, xgb_full, change):
    import mlflow

    experiment = f"player_{target.lower()}"
    setup_mlflow(experiment)
    with mlflow.start_run(run_name=f"xgb-{experiment}"):
        mlflow.log_params(PARAMS)
        mlflow.log_metrics({
            "naive_mae": naive,
            "linear_mae": linear,
            "test_mae": xgb_comparable,
            "test_mae_full": xgb_full,
            "mae_change_pct_vs_linear": change,
            "best_iteration": trees,
        })


def main():
    check_no_leakage()

    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df):,} player-games - no rows dropped "
          f"(XGBoost handles NaN natively).")

    usable_by_baseline = len(complete_rows(df))
    print(f"  the baselines could only use {usable_by_baseline:,} of these; "
          f"XGBoost trains on {len(df) - usable_by_baseline:,} more rows.")

    train, validation, test = split_three_way(df)
    test_comparable = complete_rows(test)
    print(f"\nThree-way table scored on {len(test_comparable):,} complete-feature "
          f"test rows (what the baselines could score); "
          f"{len(test) - len(test_comparable):,} held aside for the secondary number.")

    results = [train_target(t, train, validation, test, test_comparable)
               for t in TARGETS]

    section(f"VERDICTS (bar: MAE improvement over LinearRegression > "
            f"{IMPROVEMENT_BAR_PCT:.1f}%)")
    print(f"{'TARGET':<16}{'NAIVE':>9}{'LINEAR':>9}{'XGBOOST':>9}"
          f"{'CHANGE':>9}{'TREES':>7}   VERDICT")
    print("-" * 76)
    for row in results:
        print(f"{TARGET_LABELS[row['target']]:<16}{row['naive']:>9.3f}"
              f"{row['linear']:>9.3f}{row['xgb']:>9.3f}{row['change']:>8.1f}%"
              f"{row['trees']:>7}   {row['verdict']}")

    cleared = sum(1 for r in results if r["verdict"] == "REAL GAIN")
    print(f"\n{cleared} of {len(results)} targets clear the bar.")

    section("TREE COUNTS - is FG3M converging early?")
    print("A much lower count than its peers is the same signature the rejected")
    print("advanced-stats experiment produced: added variance, not added signal.\n")
    for row in results:
        flag = "  <-- the flagged target" if row["target"] == FLAG_TARGET else ""
        print(f"  {TARGET_LABELS[row['target']]:<16}{row['trees']:>5} trees   "
              f"team-context gain {row['context_share']:5.1f}%{flag}")


if __name__ == "__main__":
    main()
