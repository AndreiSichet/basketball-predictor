"""
A deliberately small hyperparameter grid, run to test one specific claim.

Input:  data-pipeline/data/processed/model_dataset.csv
Output: one tuned MLflow run appended to each target's existing experiment.

THE CLAIM UNDER TEST: after XGBoost failed to beat a linear model on all
seven targets, the conclusion drawn was that this is an information ceiling
in the feature set, not a modelling ceiling — and therefore that tuning
would move the numbers by fractions of a percent. That conclusion is load
bearing: it is the reason to go collect new features instead of tuning. It
should not be taken on trust just because it sounds reasonable.

So: a 2x2 grid (max_depth in {3, 6} x learning_rate in {0.05, 0.1}) across
all seven targets. This is not a real search and is not meant to be. It is
sized to be decisive in one direction only — if four configs spanning
shallow-to-deep and slow-to-fast move nothing, a larger sweep over the same
features is very unlikely to. If something does move, that is the signal to
spend real time on a proper search.

FIT COUNT: 35, not 28. The grid is 7 x 4, but each target also refits the
original hand-picked config (max_depth=4, learning_rate=0.05, which is not
a grid point) so that "delta vs original" is measured inside this process
on this data, rather than transcribed from an earlier terminal. The same
reasoning applies to the linear baselines, which are refit here too.

SELECTION IS ON VALIDATION, NEVER TEST. The whole point of holding out a
validation season is that the config can be chosen without consulting the
test set. Picking the config with the best test score would produce a
number that is guaranteed to look good and guaranteed to mean nothing.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

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

# Unlike the split logic that was pulled into common.py, these genuinely
# belong to the scripts they come from: this script's job is to compare
# against the configs those two scripts actually use, so it should point at
# them rather than keep a copy that can quietly fall out of date.
from train_moneyline_xgb import EXPERIMENT_NAME as MONEYLINE_EXPERIMENT
from train_moneyline_xgb import PARAMS as MONEYLINE_PARAMS
from train_moneyline_xgb import TARGET as MONEYLINE_TARGET
from train_regression_xgb import IMPROVEMENT_THRESHOLD_PCT
from train_regression_xgb import PARAMS as REGRESSION_PARAMS
from train_regression_xgb import experiment_name

GRID = {
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
}


@dataclass(frozen=True)
class Task:
    """One prediction target plus how to score and baseline it."""

    key: str          # MLflow experiment name, e.g. "reb_margin"
    label: str        # display name, e.g. "REB margin"
    target: str       # y column in model_dataset.csv
    classification: bool
    stat: str = ""    # ROLL10 stat behind the naive baseline (regression only)
    combine: str = ""  # "diff" or "sum" (regression only)

    @property
    def primary_metric(self) -> str:
        """The metric every comparison in this script is measured on.

        Log loss rather than accuracy for moneyline: accuracy over 2,138
        games moves in steps of ~0.05%, far too coarse to detect the small
        effects this grid is looking for, and it throws away confidence
        information the model does produce.
        """
        return "log loss" if self.classification else "MAE"

    @property
    def primary_format(self) -> str:
        return ".4f" if self.classification else ".2f"

    @property
    def baseline_name(self) -> str:
        return "LogisticRegression" if self.classification else "LinearRegression"


TASKS = [
    Task(
        key=MONEYLINE_EXPERIMENT,
        label="Moneyline",
        target=MONEYLINE_TARGET,
        classification=True,
    )
] + [
    Task(
        key=experiment_name(label),
        label=label,
        target=target,
        classification=False,
        stat=stat,
        combine=combine,
    )
    for target, label, stat, combine in REGRESSION_TARGETS
]


def base_params(task: Task) -> dict:
    return MONEYLINE_PARAMS if task.classification else REGRESSION_PARAMS


def fit_xgb(task: Task, params: dict, train, validation):
    """Fit with early stopping against the validation season.

    Both estimators report best_score as the eval_metric they were given
    (logloss / rmse), and both are lower-is-better, so config selection is
    a plain minimum in either case.
    """
    estimator = XGBClassifier if task.classification else XGBRegressor
    model = estimator(**params)
    model.fit(
        train[FEATURE_COLUMNS],
        train[task.target],
        eval_set=[(validation[FEATURE_COLUMNS], validation[task.target])],
        verbose=False,
    )
    return model


def score_model(task: Task, model, split) -> tuple:
    """Return (primary, secondary) test metrics for this task type."""
    x, y = split[FEATURE_COLUMNS], split[task.target]
    if task.classification:
        return log_loss(y, model.predict_proba(x)[:, 1]), accuracy_score(y, model.predict(x))
    predictions = model.predict(x)
    return (
        mean_absolute_error(y, predictions),
        np.sqrt(mean_squared_error(y, predictions)),
    )


def fit_linear_family_baseline(task: Task, train, validation, test) -> tuple:
    """Refit the linear baseline this target was originally measured against.

    Fit on train+validation (2015-2023), matching train_baseline.py: a model
    with no hyperparameters has nothing to stop early on, so withholding the
    validation season would handicap it for no reason. NaN rows dropped and
    features standardized because these models require it — XGBoost does not.
    """
    fit_set = pd.concat([train, validation]).dropna(subset=ROLLING_FEATURE_COLUMNS)

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_set[FEATURE_COLUMNS])
    x_test = scaler.transform(test[FEATURE_COLUMNS])
    y_fit, y_test = fit_set[task.target], test[task.target]

    if task.classification:
        model = LogisticRegression(max_iter=1000).fit(x_fit, y_fit)
        probabilities = model.predict_proba(x_test)[:, 1]
        return log_loss(y_test, probabilities), accuracy_score(y_test, model.predict(x_test))

    model = LinearRegression().fit(x_fit, y_fit)
    predictions = model.predict(x_test)
    return (
        mean_absolute_error(y_test, predictions),
        np.sqrt(mean_squared_error(y_test, predictions)),
    )


def naive_scores(task: Task, test) -> tuple:
    """The untrained baseline, which differs in kind between task types.

    Moneyline's naive tier predicts a hard "home wins", which has no
    meaningful log loss (a confident wrong call is infinitely penalised),
    so only its accuracy is reported.
    """
    y = test[task.target]
    if task.classification:
        return None, accuracy_score(y, np.ones(len(y), dtype=int))

    predictions = naive_prediction(test, task.stat, task.combine)
    return (
        mean_absolute_error(y, predictions),
        np.sqrt(mean_squared_error(y, predictions)),
    )


def relative_change(new: float, reference: float) -> float:
    """Percent change in a lower-is-better metric. Negative = improvement."""
    return (new - reference) / reference * 100


def run_grid(task: Task, train, validation):
    """Fit all four configs, returning them ranked best-validation-first."""
    results = []
    for depth, rate in product(GRID["max_depth"], GRID["learning_rate"]):
        params = {**base_params(task), "max_depth": depth, "learning_rate": rate}
        model = fit_xgb(task, params, train, validation)
        results.append(
            {
                "max_depth": depth,
                "learning_rate": rate,
                "params": params,
                "model": model,
                "validation_score": model.best_score,
                "trees": model.best_iteration,
            }
        )
    return sorted(results, key=lambda r: r["validation_score"])


def print_grid(task: Task, grid_results, selected):
    metric = "val logloss" if task.classification else "val RMSE"
    print(f"{'max_depth':>9} {'lr':>6} {'trees':>6} {metric:>12}")
    print("-" * 36)
    for result in sorted(grid_results, key=lambda r: (r["max_depth"], r["learning_rate"])):
        marker = "  <- selected" if result is selected else ""
        print(
            f"{result['max_depth']:>9} {result['learning_rate']:>6} "
            f"{result['trees']:>6} {result['validation_score']:>12.4f}{marker}"
        )
    print("(selected on validation only - the test set is not consulted here)")


def print_detail(task: Task, rows, baseline_primary):
    primary = task.primary_metric.upper()
    secondary = "ACCURACY" if task.classification else "RMSE"
    width = max(len(name) for name, _, _ in rows)

    print(f"\n{'METHOD':<{width}} {primary:>9} {secondary:>9} {'vs ' + task.baseline_name:>21}")
    print("-" * (width + 42))
    for name, value, other in rows:
        value_cell = f"{value:{task.primary_format}}" if value is not None else "-"
        other_cell = f"{other:.4f}" if task.classification else f"{other:.2f}"
        if name == task.baseline_name:
            delta = "-"
        elif value is None:
            delta = "-"
        else:
            delta = f"{relative_change(value, baseline_primary):+.1f}%"
        print(f"{name:<{width}} {value_cell:>9} {other_cell:>9} {delta:>21}")
    print("(negative = better; lower is better for both log loss and MAE)")


def log_tuned_run(task: Task, selected, train, validation, test_comparable, metrics):
    setup_mlflow(task.key)

    with mlflow.start_run(run_name=f"xgb-{task.key}-tuned"):
        mlflow.set_tags(
            {
                "search": "grid-2x2",
                "selected_on": "validation",
                "best_max_depth": selected["max_depth"],
                "best_learning_rate": selected["learning_rate"],
            }
        )
        mlflow.log_params(selected["params"])
        mlflow.log_params(
            {
                "target": task.target,
                "n_features": len(FEATURE_COLUMNS),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test_comparable),
                "configs_tried": len(GRID["max_depth"]) * len(GRID["learning_rate"]),
                "dropped_incomplete_rows": False,
            }
        )
        mlflow.log_metrics(metrics)

        sample = test_comparable[FEATURE_COLUMNS].head(5)
        predictor = selected["model"].predict_proba if task.classification else selected["model"].predict
        mlflow.xgboost.log_model(
            selected["model"],
            name="model",
            signature=infer_signature(sample, predictor(sample)),
            input_example=sample,
        )


def naive_label(task: Task) -> str:
    if task.classification:
        return "Naive: always home"
    return f"Naive: ROLL10 {task.stat} {task.combine}"


def tune_task(task: Task, train, validation, test, test_comparable) -> dict:
    section(f"{task.label.upper()} (target: {task.target})")

    grid_results = run_grid(task, train, validation)
    selected = grid_results[0]
    print_grid(task, grid_results, selected)

    original = fit_xgb(task, base_params(task), train, validation)

    tuned_primary, tuned_secondary = score_model(task, selected["model"], test_comparable)
    original_primary, original_secondary = score_model(task, original, test_comparable)
    full_primary, full_secondary = score_model(task, selected["model"], test)
    baseline_primary, baseline_secondary = fit_linear_family_baseline(
        task, train, validation, test_comparable
    )
    naive_primary, naive_secondary = naive_scores(task, test_comparable)

    config_label = f"d={selected['max_depth']} lr={selected['learning_rate']}"
    rows = [
        (naive_label(task), naive_primary, naive_secondary),
        (task.baseline_name, baseline_primary, baseline_secondary),
        ("XGBoost (original d=4 lr=0.05)", original_primary, original_secondary),
        (f"XGBoost (tuned {config_label})", tuned_primary, tuned_secondary),
    ]
    print_detail(task, rows, baseline_primary)

    full_cell = f"{full_primary:{task.primary_format}}"
    print(
        f"\nTuned model on the full test set ({len(test)} games): "
        f"{task.primary_metric} {full_cell}"
    )

    vs_original = relative_change(tuned_primary, original_primary)
    vs_baseline = relative_change(tuned_primary, baseline_primary)

    log_tuned_run(
        task,
        selected,
        train,
        validation,
        test_comparable,
        {
            "validation_score": selected["validation_score"],
            "best_iteration": selected["model"].best_iteration,
            "test_primary": tuned_primary,
            "test_secondary": tuned_secondary,
            "test_primary_full": full_primary,
            "test_secondary_full": full_secondary,
            "original_config_test_primary": original_primary,
            "baseline_test_primary": baseline_primary,
            "pct_change_vs_original": vs_original,
            "pct_change_vs_baseline": vs_baseline,
        },
    )

    return {
        "label": task.label,
        "metric": task.primary_metric,
        "format": task.primary_format,
        "config": config_label,
        "original": original_primary,
        "tuned": tuned_primary,
        "vs_original": vs_original,
        "vs_baseline": vs_baseline,
        "improved": vs_original < -IMPROVEMENT_THRESHOLD_PCT,
        "beats_baseline": vs_baseline < 0,
    }


def print_summary(results):
    section(f"TUNING SUMMARY (bar: >{IMPROVEMENT_THRESHOLD_PCT}% improvement over the original config)")

    label_w = max(len(r["label"]) for r in results)
    metric_w = max(len(r["metric"]) for r in results)
    config_w = max(len(r["config"]) for r in results)

    header = (
        f"{'TARGET':<{label_w}}  {'METRIC':<{metric_w}}  {'ORIGINAL':>9} {'TUNED':>9}  "
        f"{'BEST CONFIG':<{config_w}}  {'vs ORIG':>8} {'vs BASE':>8}  VERDICT"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        verdict = "REAL GAIN" if r["improved"] else "no meaningful gain"
        print(
            f"{r['label']:<{label_w}}  {r['metric']:<{metric_w}}  "
            f"{r['original']:>9{r['format']}} {r['tuned']:>9{r['format']}}  "
            f"{r['config']:<{config_w}}  "
            f"{r['vs_original']:>+7.1f}% {r['vs_baseline']:>+7.1f}%  {verdict}"
        )

    print("(negative = better. vs ORIG compares tuned against the hand-picked")
    print(" config; vs BASE compares it against LinearRegression/LogisticRegression.)")

    gains = sum(r["improved"] for r in results)
    beats = sum(r["beats_baseline"] for r in results)
    print(f"\n{gains} of {len(results)} targets improved meaningfully on the original config.")
    print(f"{beats} of {len(results)} tuned models beat their linear baseline at all.")

    if gains == 0:
        print(
            "\nThe information-ceiling reading survives this grid: no config tried "
            "moved any\ntarget past the bar. New features, not new hyperparameters."
        )
    else:
        print(
            "\nAt least one target responded to tuning - the ceiling reading is not "
            "safe as stated.\nA proper search on the affected targets is now worth the time."
        )


def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df)} games - no rows dropped (XGBoost handles NaN natively).")

    train, validation, test = split_three_way(df)
    test_comparable = test.dropna(subset=ROLLING_FEATURE_COLUMNS)

    configs = len(GRID["max_depth"]) * len(GRID["learning_rate"])
    print(
        f"\nScoring on {len(test_comparable)} complete-window test games, matching "
        f"every earlier comparison."
    )
    print(
        f"{len(TASKS)} targets x ({configs} grid configs + 1 original refit) = "
        f"{len(TASKS) * (configs + 1)} fits."
    )

    results = [tune_task(task, train, validation, test, test_comparable) for task in TASKS]
    print_summary(results)


if __name__ == "__main__":
    main()
