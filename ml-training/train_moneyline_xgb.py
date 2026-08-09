"""
XGBoost moneyline model — the first model expected to beat the baselines.

Input:  data-pipeline/data/processed/model_dataset.csv
Output: an MLflow run under ml-training/mlruns (params, metrics, model artifact).

Three things differ from train_baseline.py, each deliberately:

  1. THREE-WAY chronological split. The baseline used train/test; this adds a
     validation season between them. Early stopping needs a set to measure
     "stopped improving" against, and using the test set for that would
     quietly turn it into a training signal — the reported test number would
     become optimistic in a way nothing downstream could detect.

  2. NO row dropping. XGBoost handles NaN natively (it learns a default
     branch direction at each split), so the early-season rows the baseline
     was forced to discard are kept here — about 1,400 extra training games.
     Measured, this buys very little: training with the rows dropped instead
     scores 0.6763/0.6088 against 0.6754/0.6067 with them kept, i.e. a wash
     within noise. The capability is real; the payoff on this dataset is not.
     Worth re-measuring if the feature set grows.

  3. NO scaling. Trees split on thresholds within a single feature, so
     monotonic rescaling cannot change which splits get chosen. The
     StandardScaler the baseline needed would be inert here.

TEST-SET NOTE: the test set is deliberately restricted to the same
complete-window games the baseline scored on. Scoring the extra early-season
rows here would make the headline number incomparable to the baselines this
script exists to beat — a different, harder set of games. The unrestricted
test set is reported separately, since that one reflects real deployment,
where early-season games still have to be predicted.
"""

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from common import (
    TEST_SEASONS,
    TRACKING_URI,
    VALIDATION_SEASON,
    setup_mlflow,
    split_three_way,
)

# Shared with the baseline rather than re-declared: the 34-column feature
# list and the loader's column-drift guard must agree across both scripts,
# and duplicating them is how they would silently stop agreeing.
from train_baseline import (
    FEATURE_COLUMNS,
    ROLLING_FEATURE_COLUMNS,
    load_dataset,
    section,
)

TARGET = "HOME_WIN"
EXPERIMENT_NAME = "moneyline"

# n_estimators is an upper bound, not a target — early stopping picks the
# real count. Depth is kept shallow and sampling aggressive because the
# signal here is weak (the baselines top out near 68% accuracy) and this
# dataset is small enough for a deep model to memorize rather than learn.
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    # xgboost >= 2.0 takes this on the constructor; .fit() no longer accepts
    # it. eval_set is still passed to .fit() below.
    "early_stopping_rounds": 50,
    "random_state": 42,
}

# Transcribed from train_baseline.py on the identical test set — NOT
# recomputed here. If that script's numbers move, update these.
BASELINES = [
    ("Naive: always home", 0.5458, None),
    ("Elo win probability", 0.6674, 0.6130),
    ("LogisticRegression", 0.6773, 0.6045),
]
REFERENCE_METHOD = "LogisticRegression"


def report_retained_rows(train, validation):
    """Quantify advantage #2: rows the baseline dropped that XGBoost keeps."""
    kept = 0
    for name, split in (("train", train), ("validation", validation)):
        incomplete = split[ROLLING_FEATURE_COLUMNS].isna().any(axis=1).sum()
        kept += incomplete
        print(f"  {name}: {incomplete} incomplete-window rows kept of {len(split)}")
    print(f"  {kept} extra training rows the baseline had to discard.")


def evaluate(model, split, label):
    x, y = split[FEATURE_COLUMNS], split[TARGET]
    proba = model.predict_proba(x)[:, 1]
    accuracy = accuracy_score(y, model.predict(x))
    loss = log_loss(y, proba)
    print(f"{label:<44} accuracy {accuracy:.4f}  log loss {loss:.4f}")
    return accuracy, loss


def print_top_features(model, count=10):
    section(f"TOP {count} FEATURES BY GAIN")
    scores = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:count]
    width = max(len(name) for name, _ in ranked)
    for name, gain in ranked:
        print(f"  {name:<{width}}  {gain:8.2f}")


def print_comparison(accuracy, loss, test_rows):
    section(f"MONEYLINE COMPARISON (test: seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1]}, {test_rows} games)")

    reference = next(acc for name, acc, _ in BASELINES if name == REFERENCE_METHOD)
    rows = [*BASELINES, ("XGBoost", accuracy, loss)]

    print(f"{'METHOD':<22} {'ACCURACY':>9} {'LOG LOSS':>9} {'vs ' + REFERENCE_METHOD:>18}")
    print("-" * 61)
    for name, acc, ll in rows:
        loss_cell = f"{ll:.4f}" if ll is not None else "-"
        delta = "-" if name == REFERENCE_METHOD else f"{acc - reference:+.4f}"
        print(f"{name:<22} {acc:>9.4f} {loss_cell:>9} {delta:>18}")


def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df)} games - no rows dropped (XGBoost handles NaN natively).")

    train, validation, test = split_three_way(df)
    report_retained_rows(train, validation)

    # Scored on exactly the rows the baseline scored on, so the headline
    # number stays comparable. See TEST-SET NOTE in the module docstring.
    test_comparable = test.dropna(subset=ROLLING_FEATURE_COLUMNS)
    print(
        f"\nScoring on {len(test_comparable)} complete-window test games "
        f"(baseline-comparable); {len(test) - len(test_comparable)} early-season "
        "test rows held aside for the secondary number."
    )

    setup_mlflow(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgb-moneyline") as run:
        section("TRAINING")
        model = XGBClassifier(**PARAMS)
        model.fit(
            train[FEATURE_COLUMNS],
            train[TARGET],
            eval_set=[(validation[FEATURE_COLUMNS], validation[TARGET])],
            verbose=False,
        )
        print(
            f"Early stopping at iteration {model.best_iteration} "
            f"of a possible {PARAMS['n_estimators']}."
        )
        print(f"Best validation log loss: {model.best_score:.4f}")

        section("TEST RESULTS")
        accuracy, loss = evaluate(model, test_comparable, "XGBoost (baseline-comparable set)")
        full_accuracy, full_loss = evaluate(model, test, "XGBoost (full test set, incl. early season)")

        mlflow.log_params(PARAMS)
        mlflow.log_params(
            {
                "n_features": len(FEATURE_COLUMNS),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test_comparable),
                "validation_season": VALIDATION_SEASON,
                "test_seasons": f"{TEST_SEASONS[0]}-{TEST_SEASONS[-1]}",
                "dropped_incomplete_rows": False,
            }
        )
        mlflow.log_metrics(
            {
                "best_iteration": model.best_iteration,
                "validation_log_loss": model.best_score,
                "test_accuracy": accuracy,
                "test_log_loss": loss,
                "test_accuracy_full": full_accuracy,
                "test_log_loss_full": full_loss,
            }
        )

        sample = test_comparable[FEATURE_COLUMNS].head(5)
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=infer_signature(sample, model.predict_proba(sample)),
            input_example=sample,
        )

        print_top_features(model)
        print_comparison(accuracy, loss, len(test_comparable))

        print(f"\nMLflow run {run.info.run_id} logged to experiment '{EXPERIMENT_NAME}'.")
        print(f"View with: mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main()
