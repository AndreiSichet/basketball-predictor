"""
XGBoost for the six Q1 and first-half targets, against the linear baselines
just established in train_quarter_half_baseline.py.

  input: data-pipeline/data/processed/model_dataset.csv

NOTHING HERE IS RESTATED. The 30-column feature set and the six target
definitions come from the baseline script, the season boundaries from
common.split_three_way, and both parameter dicts from the team-level
XGBoost scripts they were tuned on. A second copy of any of them is a drift
bug waiting for someone to edit one and not the other.

TRAINS ON EVERY ROW IT HAS A LABEL FOR, which is the point of using trees
here. The linear baseline had to drop 1,788 rows (13.5%) whose trailing
Q1/1H windows were incomplete, because LinearRegression cannot take NaN.
XGBoost learns a default split direction for missing values, so it keeps
them - the same deliberate advantage taken at team level (13,199 vs 11,465)
and for player props (280,943 vs 225,060).

SCORED TWO WAYS for that reason. The comparison table uses only test rows
the linear model could also score, so naive, linear and XGBoost are judged
on identical games. XGBoost's number over the full test set is printed
separately: it is a real capability the others lack, but it is not a
like-for-like comparison and is never presented as one.

TIES ARE STILL DROPPED FROM THE TWO CLASSIFIERS, and this is the one place
the policy needed restating rather than inheriting. XGBoost tolerating NaN
in FEATURES says nothing about NaN in a LABEL: there is no third class for
a binary objective to predict, and 611 tied first quarters have no true
answer to score against. So the same rule as the linear baseline applies -
tied rows out of training, validation and test for HOME_Q1_WIN and
HOME_HALF1_WIN only, and the four regression targets keep them. Letting
XGBoost handle ties differently just because it handles missing features
differently would make the two scripts' numbers incomparable for a reason
that has nothing to do with the models.

  The three no-data games are dropped from every target for the same
  reason: a missing label is not a trainable one, whatever the model class.

WATCH THE TREE COUNTS. Early stopping converging much sooner on one target
than its peers has meant added variance rather than added signal twice in
this project - the advanced-stats bundle and FG3M. Q1 is the target most
likely to show it here: twelve minutes is short enough that most of its
outcome is variance rather than anything a trailing average can anticipate,
and the linear baseline already put Q1 winner at 0.5796 against 1H's
0.6343. The counts are printed together at the end for that comparison.
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
from xgboost import XGBClassifier, XGBRegressor

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

from common import setup_mlflow, split_three_way
from train_baseline import load_dataset, naive_prediction, section

# The 30 local features and the six targets - defined once, in the script
# that established them. FEATURE_COLUMNS here is that local set, NOT
# common.py's 38 shipped ones.
from train_quarter_half_baseline import (
    CLASSIFICATION_TARGETS,
    FEATURE_COLUMNS,
    REGRESSION_TARGETS,
)

# The exact configs the team-level work settled on, imported so they cannot
# drift apart. They differ only in objective/eval_metric.
from train_regression_xgb import PARAMS as REGRESSION_PARAMS, experiment_name
from train_moneyline_xgb import PARAMS as CLASSIFICATION_PARAMS

# Same bar every regression target in this project has been held to.
IMPROVEMENT_THRESHOLD_PCT = 3.0


def complete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows a linear model could also use - no NaN anywhere in the inputs."""
    return df.dropna(subset=FEATURE_COLUMNS)


def labelled(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows with a real label.

    For the four regression targets this removes only the three games with
    no quarter data. For the two classifiers it also removes tied periods,
    which carry NA precisely because no winner exists to predict.
    """
    return df.dropna(subset=[target])


def fit_linear(train, validation, test_comparable, target, classifier=False):
    """Reproduce the baseline script's model for this target.

    train + validation is exactly the baseline's training set: its two-way
    split held out the last two seasons, and this three-way split's train
    and validation together are the same seasons. Rows are dropped and
    features scaled because these models require it; XGBoost needs neither.
    """
    fit_set = labelled(complete_rows(pd.concat([train, validation])), target)

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_set[FEATURE_COLUMNS])
    x_test = scaler.transform(test_comparable[FEATURE_COLUMNS])

    if classifier:
        model = LogisticRegression(max_iter=1000).fit(
            x_fit, fit_set[target].astype(int))
        return model.predict(x_test), model.predict_proba(x_test)[:, 1]

    model = LinearRegression().fit(x_fit, fit_set[target])
    return model.predict(x_test), None


def run_regression(target, label, stat, combine, train, validation, test) -> dict:
    section(f"{label.upper()} (target: {target})")

    fit_train = labelled(train, target)
    fit_validation = labelled(validation, target)
    test_scored = labelled(test, target)
    test_comparable = complete_rows(test_scored)

    model = XGBRegressor(**REGRESSION_PARAMS)
    model.fit(
        fit_train[FEATURE_COLUMNS], fit_train[target],
        eval_set=[(fit_validation[FEATURE_COLUMNS], fit_validation[target])],
        verbose=False,
    )
    trees = int(model.best_iteration)

    y = test_comparable[target]
    naive_mae = mean_absolute_error(y, naive_prediction(test_comparable, stat, combine))
    linear_pred, _ = fit_linear(train, validation, test_comparable, target)
    linear_mae = mean_absolute_error(y, linear_pred)
    xgb_pred = model.predict(test_comparable[FEATURE_COLUMNS])
    xgb_mae = mean_absolute_error(y, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y, xgb_pred))
    full_mae = mean_absolute_error(
        test_scored[target], model.predict(test_scored[FEATURE_COLUMNS]))

    change = (xgb_mae - linear_mae) / linear_mae * 100
    improved = change < -IMPROVEMENT_THRESHOLD_PCT

    print(f"Trained on {len(fit_train):,} rows "
          f"({len(fit_train) - len(complete_rows(fit_train)):,} of them with an "
          f"incomplete window the linear model could not use).")
    print(f"Early stopping at iteration {trees} of "
          f"{REGRESSION_PARAMS['n_estimators']}.  "
          f"Best validation RMSE {model.best_score:.3f}\n")

    print(f"{'METHOD':<26}{'MAE':>8}{'vs LINEAR':>12}")
    print("-" * 46)
    print(f"{'Naive: ROLL10 ' + stat + ' ' + combine:<26}{naive_mae:>8.2f}"
          f"{(naive_mae - linear_mae) / linear_mae * 100:>+11.1f}%")
    print(f"{'LinearRegression':<26}{linear_mae:>8.2f}{'-':>12}")
    print(f"{'XGBoost':<26}{xgb_mae:>8.2f}{change:>+11.1f}%")
    print(f"\n  scored on {len(test_comparable):,} comparable test rows; "
          f"XGBoost over all {len(test_scored):,}: MAE {full_mae:.2f}")
    print("  (no baseline counterpart - the others cannot score those rows)")

    log_run(label, target, model, fit_train, fit_validation, test_comparable,
            REGRESSION_PARAMS, {
                "naive_mae": naive_mae, "linear_mae": linear_mae,
                "test_mae": xgb_mae, "test_rmse": xgb_rmse,
                "test_mae_full": full_mae,
                "mae_change_pct_vs_linear": change,
                "best_iteration": trees,
            })

    return {"label": label, "linear": linear_mae, "xgb": xgb_mae,
            "change": change, "improved": improved, "trees": trees,
            "metric": "MAE"}


def run_classification(target, label, period, train, validation, test) -> dict:
    section(f"{label.upper()} (target: {target})")

    # labelled() is where the tie policy takes effect - see the docstring.
    fit_train = labelled(train, target)
    fit_validation = labelled(validation, target)
    test_scored = labelled(test, target)
    test_comparable = complete_rows(test_scored)

    print(f"Tied {period} periods dropped: {len(train) - len(fit_train):,} train, "
          f"{len(test) - len(test_scored):,} test "
          f"(no winner exists, so there is no label to learn).")

    model = XGBClassifier(**CLASSIFICATION_PARAMS)
    model.fit(
        fit_train[FEATURE_COLUMNS], fit_train[target].astype(int),
        eval_set=[(fit_validation[FEATURE_COLUMNS],
                   fit_validation[target].astype(int))],
        verbose=False,
    )
    trees = int(model.best_iteration)

    y = test_comparable[target].astype(int)
    naive_acc = accuracy_score(y, np.ones(len(y), dtype=int))
    linear_pred, linear_proba = fit_linear(
        train, validation, test_comparable, target, classifier=True)
    linear_acc = accuracy_score(y, linear_pred)
    linear_loss = log_loss(y, linear_proba)

    proba = model.predict_proba(test_comparable[FEATURE_COLUMNS])[:, 1]
    xgb_acc = accuracy_score(y, model.predict(test_comparable[FEATURE_COLUMNS]))
    xgb_loss = log_loss(y, proba)

    y_full = test_scored[target].astype(int)
    full_acc = accuracy_score(y_full, model.predict(test_scored[FEATURE_COLUMNS]))
    full_loss = log_loss(
        y_full, model.predict_proba(test_scored[FEATURE_COLUMNS])[:, 1])

    # Log loss is the verdict metric: it scores the probability, not just
    # which side of 0.5 it fell on. Same reasoning as the moneyline work.
    change = (xgb_loss - linear_loss) / linear_loss * 100
    improved = change < -IMPROVEMENT_THRESHOLD_PCT

    print(f"Trained on {len(fit_train):,} rows.")
    print(f"Early stopping at iteration {trees} of "
          f"{CLASSIFICATION_PARAMS['n_estimators']}.  "
          f"Best validation log loss {model.best_score:.4f}\n")

    print(f"{'METHOD':<26}{'ACCURACY':>10}{'LOG LOSS':>10}{'vs LINEAR':>12}")
    print("-" * 58)
    print(f"{'Naive: always home':<26}{naive_acc:>10.4f}{'-':>10}{'-':>12}")
    print(f"{'LogisticRegression':<26}{linear_acc:>10.4f}{linear_loss:>10.4f}"
          f"{'-':>12}")
    print(f"{'XGBoost':<26}{xgb_acc:>10.4f}{xgb_loss:>10.4f}{change:>+11.1f}%")
    print(f"\n  scored on {len(test_comparable):,} comparable test rows; "
          f"XGBoost over all {len(test_scored):,}: "
          f"accuracy {full_acc:.4f}  log loss {full_loss:.4f}")

    log_run(label, target, model, fit_train, fit_validation, test_comparable,
            CLASSIFICATION_PARAMS, {
                "naive_accuracy": naive_acc,
                "linear_accuracy": linear_acc, "linear_log_loss": linear_loss,
                "test_accuracy": xgb_acc, "test_log_loss": xgb_loss,
                "test_accuracy_full": full_acc, "test_log_loss_full": full_loss,
                "log_loss_change_pct_vs_linear": change,
                "best_iteration": trees,
            })

    return {"label": label, "linear": linear_loss, "xgb": xgb_loss,
            "change": change, "improved": improved, "trees": trees,
            "metric": "log loss"}


def log_run(label, target, model, train, validation, test, params, metrics):
    setup_mlflow(experiment_name(label))
    with mlflow.start_run(run_name=f"xgb-{experiment_name(label)}"):
        mlflow.log_params(params)
        mlflow.log_params({
            "target": target,
            "n_features": len(FEATURE_COLUMNS),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "dropped_incomplete_rows": False,
        })
        mlflow.log_metrics(metrics)

        sample = test[FEATURE_COLUMNS].head(5)
        mlflow.xgboost.log_model(
            model, name="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=sample,
        )


def print_verdicts(results):
    section(f"VERDICTS (bar: improvement over the linear model > "
            f"{IMPROVEMENT_THRESHOLD_PCT}%)")
    width = max(len(r["label"]) for r in results)
    print(f"{'TARGET':<{width}}  {'METRIC':<9}{'LINEAR':>9}{'XGBOOST':>9}"
          f"{'CHANGE':>9}{'TREES':>7}   VERDICT")
    print("-" * (width + 56))
    for r in results:
        verdict = "REAL GAIN" if r["improved"] else "no meaningful gain"
        print(f"{r['label']:<{width}}  {r['metric']:<9}{r['linear']:>9.4f}"
              f"{r['xgb']:>9.4f}{r['change']:>+8.1f}%{r['trees']:>7}   {verdict}")

    cleared = sum(r["improved"] for r in results)
    print(f"\n{cleared} of {len(results)} targets clear the bar.")


def print_tree_diagnostic(results):
    section("TREE COUNTS - is any target converging early?")
    print("Early stopping firing much sooner than on comparable targets has")
    print("meant added variance rather than added signal twice here: the")
    print("advanced-stats bundle and FG3M. See the module docstring.\n")

    lowest = min(results, key=lambda r: r["trees"])
    for r in sorted(results, key=lambda r: r["trees"]):
        marker = "  <-- lowest" if r is lowest else ""
        print(f"  {r['label']:<12}{r['trees']:>5} trees{marker}")

    others = [r["trees"] for r in results if r is not lowest]
    ratio = lowest["trees"] / (sum(others) / len(others))
    print(f"\n  {lowest['label']} stopped at {ratio:.0%} of the mean of the "
          f"other five ({sum(others) / len(others):.0f}).")
    if "Q1" in lowest["label"]:
        print("  That is the target the docstring predicted: twelve minutes is")
        print("  mostly short-window variance, so there is less for additional")
        print("  trees to fit before validation stops improving.")
    else:
        print("  NOT the target predicted - worth understanding before trusting")
        print("  either this result or the reasoning that expected Q1.")


def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df):,} games - no rows dropped for missing features "
          f"(XGBoost handles NaN natively).")
    print(f"  the linear baseline could use only "
          f"{len(complete_rows(df)):,} of them.")
    print(f"Feature set: {len(FEATURE_COLUMNS)} columns, imported from "
          f"train_quarter_half_baseline.")

    train, validation, test = split_three_way(df)

    results = [run_regression(target, label, stat, combine, train, validation, test)
               for target, label, stat, combine in REGRESSION_TARGETS]
    results += [run_classification(target, label, period, train, validation, test)
                for target, label, period in CLASSIFICATION_TARGETS]

    print_verdicts(results)
    print_tree_diagnostic(results)


if __name__ == "__main__":
    main()
