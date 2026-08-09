"""
XGBoost across all six regression targets, to answer one question: is the
moneyline information-ceiling result a property of that target alone, or of
the whole feature set?

Input:  data-pipeline/data/processed/model_dataset.csv
Output: one MLflow experiment per target under ml-training/mlruns.

The six targets share everything except the y column and the shape of their
naive baseline, so they run through a single parameterized function rather
than six near-identical scripts. The split and MLflow wiring come from
common.py, and the feature set from train_baseline.py, rather than being
restated here — every model has to split identically or its numbers stop
being comparable to the rest.

BASELINES ARE RECOMPUTED HERE, not transcribed from train_baseline.py. A
hardcoded number stays convincing after the thing it described has changed;
a recomputed one cannot.

FAIRNESS OF THE COMPARISON: LinearRegression is fit on train+validation
(2015-2023) while XGBoost fits on train (2015-2022) and spends validation
(2023) on deciding when to stop. That is deliberate — each model consumes
exactly the same span of history, in the manner its algorithm requires.
LinearRegression has no hyperparameters to stop early on, so withholding a
season from it would handicap it for no reason, and would also break the
reproduction of train_baseline.py's published numbers.

SCORING SETS: the naive and linear baselines cannot score early-season rows
at all — the naive formula IS the rolling columns, and LinearRegression
rejects NaN. So the three-way table is scored on the complete-window games
only. XGBoost's full-test-set number is reported separately, and has no
counterpart to compare against.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

from common import setup_mlflow, split_three_way
from train_baseline import (
    FEATURE_COLUMNS,
    REGRESSION_TARGETS,
    ROLLING_FEATURE_COLUMNS,
    load_dataset,
    naive_prediction,
    section,
)

# Same shape as the moneyline params, retargeted at a continuous outcome.
PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "random_state": 42,
}

# Below this, a difference in MAE is not worth acting on: it is smaller than
# the gap between reasonable hyperparameter choices, so it says nothing about
# whether the model class is actually better suited to the target.
IMPROVEMENT_THRESHOLD_PCT = 3.0


def experiment_name(label: str) -> str:
    """'REB margin' -> 'reb_margin', matching the 'moneyline' convention."""
    return label.lower().replace(" ", "_")


def score(y_true, predictions) -> tuple:
    return (
        mean_absolute_error(y_true, predictions),
        np.sqrt(mean_squared_error(y_true, predictions)),
    )


def fit_linear_baseline(train, validation, test, target: str):
    """Reproduce train_baseline.py's LinearRegression on this target.

    NaN rows dropped and features standardized with the scaler fit on the
    fitting set only — both required by the model, neither by XGBoost.
    """
    fit_set = pd.concat([train, validation]).dropna(subset=ROLLING_FEATURE_COLUMNS)

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_set[FEATURE_COLUMNS])
    x_test = scaler.transform(test[FEATURE_COLUMNS])

    model = LinearRegression().fit(x_fit, fit_set[target])
    return model.predict(x_test)


def fit_xgb(train, validation, target: str):
    model = XGBRegressor(**PARAMS)
    model.fit(
        train[FEATURE_COLUMNS],
        train[target],
        eval_set=[(validation[FEATURE_COLUMNS], validation[target])],
        verbose=False,
    )
    return model


def run_target(target, label, stat, combine, train, validation, test, test_comparable) -> dict:
    section(f"{label.upper()} (target: {target})")

    y_comparable = test_comparable[target]

    naive_mae, naive_rmse = score(y_comparable, naive_prediction(test_comparable, stat, combine))
    linear_mae, linear_rmse = score(
        y_comparable, fit_linear_baseline(train, validation, test_comparable, target)
    )

    model = fit_xgb(train, validation, target)
    xgb_mae, xgb_rmse = score(y_comparable, model.predict(test_comparable[FEATURE_COLUMNS]))
    full_mae, full_rmse = score(test[target], model.predict(test[FEATURE_COLUMNS]))

    # Negative = XGBoost beat LinearRegression. One convention everywhere,
    # in the table and in the verdict, so the signs never have to be re-read.
    change_pct = (xgb_mae - linear_mae) / linear_mae * 100
    improved = change_pct < -IMPROVEMENT_THRESHOLD_PCT

    print(f"Early stopping at iteration {model.best_iteration} of {PARAMS['n_estimators']}.")
    print(f"Best validation RMSE: {model.best_score:.3f}\n")

    rows = [
        (f"Naive: ROLL10 {stat} {combine}", naive_mae, naive_rmse),
        ("LinearRegression", linear_mae, linear_rmse),
        ("XGBoost", xgb_mae, xgb_rmse),
    ]
    print(f"{'METHOD':<24} {'MAE':>7} {'RMSE':>7} {'MAE vs LINREG':>15}")
    print("-" * 56)
    for name, mae, rmse in rows:
        if name == "LinearRegression":
            delta = "-"
        else:
            delta = f"{(mae - linear_mae) / linear_mae * 100:+.1f}%"
        print(f"{name:<24} {mae:>7.2f} {rmse:>7.2f} {delta:>15}")
    print("(negative = better than LinearRegression)")

    print(
        f"\nXGBoost on the full test set ({len(test)} games, incl. early season): "
        f"MAE {full_mae:.2f}  RMSE {full_rmse:.2f}"
    )
    print("  No baseline counterpart - neither naive nor linear can score those rows.")

    log_run(
        label, target, model, train, validation, test_comparable,
        {
            "naive_mae": naive_mae,
            "naive_rmse": naive_rmse,
            "linear_mae": linear_mae,
            "linear_rmse": linear_rmse,
            "test_mae": xgb_mae,
            "test_rmse": xgb_rmse,
            "test_mae_full": full_mae,
            "test_rmse_full": full_rmse,
            "mae_change_pct_vs_linear": change_pct,
            "best_iteration": model.best_iteration,
            "validation_rmse": model.best_score,
        },
    )

    return {
        "label": label,
        "linear_mae": linear_mae,
        "xgb_mae": xgb_mae,
        "change_pct": change_pct,
        "improved": improved,
    }


def log_run(label, target, model, train, validation, test_comparable, metrics):
    setup_mlflow(experiment_name(label))

    with mlflow.start_run(run_name=f"xgb-{experiment_name(label)}"):
        mlflow.log_params(PARAMS)
        mlflow.log_params(
            {
                "target": target,
                "n_features": len(FEATURE_COLUMNS),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test_comparable),
                "dropped_incomplete_rows": False,
            }
        )
        mlflow.log_metrics(metrics)

        sample = test_comparable[FEATURE_COLUMNS].head(5)
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=sample,
        )


def print_verdicts(results):
    section(f"VERDICTS (bar: MAE improvement over LinearRegression > {IMPROVEMENT_THRESHOLD_PCT}%)")

    width = max(len(r["label"]) for r in results)
    print(f"{'TARGET':<{width}}  {'LINREG':>7} {'XGBOOST':>8} {'CHANGE':>8}   VERDICT")
    print("-" * (width + 46))
    for r in results:
        verdict = "REAL GAIN" if r["improved"] else "no meaningful gain"
        print(
            f"{r['label']:<{width}}  {r['linear_mae']:>7.2f} {r['xgb_mae']:>8.2f} "
            f"{r['change_pct']:>+7.1f}%   {verdict}"
        )

    gains = sum(r["improved"] for r in results)
    print(f"\n{gains} of {len(results)} targets clear the bar.")


def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df)} games - no rows dropped (XGBoost handles NaN natively).")

    train, validation, test = split_three_way(df)
    test_comparable = test.dropna(subset=ROLLING_FEATURE_COLUMNS)
    print(
        f"\nThree-way table scored on {len(test_comparable)} complete-window games "
        f"(what LinearRegression was scored on); XGBoost additionally scored on "
        f"all {len(test)}."
    )

    results = [
        run_target(target, label, stat, combine, train, validation, test, test_comparable)
        for target, label, stat, combine in REGRESSION_TARGETS
    ]

    print_verdicts(results)
    print(f"\nSix experiments logged: {', '.join(experiment_name(r['label']) for r in results)}")


if __name__ == "__main__":
    main()
