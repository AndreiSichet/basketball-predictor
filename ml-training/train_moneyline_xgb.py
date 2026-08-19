"""
XGBoost moneyline model.

Input:  data-pipeline/data/processed/model_dataset.csv
Output: an MLflow run under ml-training/mlruns (params, metrics, model).

Three differences from train_baseline.py:

  1. Three-way split. Early stopping needs a validation set; using test for
     that would turn it into a training signal and inflate the result.

  2. No row dropping. XGBoost handles NaN natively, so the ~1,400
     early-season rows the baseline dropped are kept. Measured, this is a
     wash: dropping them scores 0.6763/0.6088 vs 0.6754/0.6067 keeping them.
     Re-measure if the feature set grows.

  3. No scaling. Trees split on thresholds within one feature, so rescaling
     can't change which splits are chosen.

Test set is restricted to the same complete-window games the baseline
scored on, so the headline number stays comparable. The full test set is
reported separately as the deployment-realistic number.
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

# Imported, not re-declared: the feature list and the loader's drift guard
# must stay identical across scripts.
from train_baseline import (
    FEATURE_COLUMNS,
    ROLLING_FEATURE_COLUMNS,
    load_dataset,
    section,
)

TARGET = "HOME_WIN"
EXPERIMENT_NAME = "moneyline"

# n_estimators is an upper bound; early stopping picks the real count.
# Shallow depth and heavy subsampling because the signal is weak and the
# dataset is small enough for a deep model to memorize it.
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
    # xgboost >= 2.0 takes this on the constructor, not .fit().
    "early_stopping_rounds": 50,
    "random_state": 42,
}

# Copied from train_baseline.py on the same test set. Update if those move.
BASELINES = [
    ("Naive: always home", 0.5458, None),
    ("Elo win probability", 0.6674, 0.6130),
    ("LogisticRegression", 0.6773, 0.6045),
]
REFERENCE_METHOD = "LogisticRegression"


def report_retained_rows(train, validation):
    """Count the incomplete-window rows the baseline dropped and this keeps."""
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

    # Same rows the baseline scored on, so the comparison holds.
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
