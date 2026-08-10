"""
Retrain the seven production models on the complete dataset.

Input:  data-pipeline/data/processed/model_dataset.csv
Output: ml-training/models/<target>.json (XGBoost native format), plus one
        MLflow run per model under the "production" experiment.

Everything before this script split the data to answer "which model should
we ship?". That question is now settled, so the split has done its job and
holding 2023-2025 back any longer would just mean deploying a model that
ignores three seasons of real signal. These models train on all of it.

THESE MODELS ARE UNEVALUATED, BY CONSTRUCTION. There is no held-out data
left to score them on, so this script deliberately logs no performance
metrics — a number here would only invite being read as validation of the
shipped model. What each configuration is worth was established during
selection, and those numbers live in the per-target experiments
("moneyline", "spread", ...) alongside the runs that produced them.

ONE CONFIG FOR ALL SEVEN. The 2x2 grid found no target where tuning beat
the hand-picked max_depth=4 / learning_rate=0.05 by a meaningful margin, so
seven slightly different configs would buy noise at the cost of seven
things to document and keep straight. Tree counts still differ per target,
because those are not a tuning choice — each is what early stopping
measured for that target.
"""

from pathlib import Path

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from xgboost import XGBClassifier, XGBRegressor

from common import TRACKING_URI, setup_mlflow
from train_baseline import FEATURE_COLUMNS, REGRESSION_TARGETS, load_dataset, section
from train_moneyline_xgb import EXPERIMENT_NAME as MONEYLINE_EXPERIMENT
from train_moneyline_xgb import PARAMS as MONEYLINE_PARAMS
from train_moneyline_xgb import TARGET as MONEYLINE_TARGET
from train_regression_xgb import PARAMS as REGRESSION_PARAMS
from train_regression_xgb import experiment_name

MODELS_DIR = Path(__file__).resolve().parent / "models"
PRODUCTION_EXPERIMENT = "production"

# Tree counts as measured by early stopping on the 2023 validation season,
# under the max_depth=4 / learning_rate=0.05 config, in train_moneyline_xgb.py
# and train_regression_xgb.py.
#
# Frozen here rather than read from MLflow at runtime for one specific
# reason: ml-training/mlruns/ is gitignored, so a script that fetched these
# live could not rebuild the production models from a fresh clone. That
# makes these numbers part of the model definition, and they belong in
# version control with everything else that defines it. verify_tree_counts()
# below re-checks them against MLflow whenever it happens to be available.
TREE_COUNTS = {
    "moneyline": 118,
    "spread": 82,
    "totals": 133,
    "reb_margin": 78,
    "reb_total": 46,
    "ast_margin": 63,
    "ast_total": 141,
}

# The config those tree counts were measured under. Any run that used a
# different depth or learning rate is a tuning run, not the original.
ORIGINAL_MAX_DEPTH = 4
ORIGINAL_LEARNING_RATE = 0.05

# (experiment key, display label, target column, is_classification). Derived
# from REGRESSION_TARGETS rather than restated, so a target cannot be added
# to the pipeline and silently skipped here.
TASKS = [(MONEYLINE_EXPERIMENT, "Moneyline", MONEYLINE_TARGET, True)] + [
    (experiment_name(label), label, target, False)
    for target, label, _stat, _combine in REGRESSION_TARGETS
]


def final_params(classification: bool, trees: int) -> dict:
    """The selection-phase config, with early stopping swapped for a fixed count.

    early_stopping_rounds has to come out: it requires an eval_set, and there
    is no held-out set here. n_estimators stops being an upper bound and
    becomes the actual number of trees built.
    """
    source = MONEYLINE_PARAMS if classification else REGRESSION_PARAMS
    params = {k: v for k, v in source.items() if k != "early_stopping_rounds"}
    params["n_estimators"] = trees
    return params


def verify_tree_counts() -> None:
    """Re-check the frozen counts against the runs they were taken from.

    Advisory only. MLflow is not required to build production models, but
    when it is present there is no reason to let a stale transcription go
    unnoticed.
    """
    section("TREE COUNT VERIFICATION")
    mlflow.set_tracking_uri(TRACKING_URI)

    checked = mismatched = 0
    for key, _label, _target, _classification in TASKS:
        try:
            runs = mlflow.search_runs(
                experiment_names=[key],
                filter_string=(
                    f"params.max_depth = '{ORIGINAL_MAX_DEPTH}' "
                    f"and params.learning_rate = '{ORIGINAL_LEARNING_RATE}'"
                ),
                order_by=["start_time DESC"],
                max_results=1,
            )
        except Exception as error:  # noqa: BLE001 - advisory check, never fatal
            print(f"  {key:<12} could not query MLflow ({type(error).__name__}) - skipped")
            continue

        if len(runs) == 0 or "metrics.best_iteration" not in runs.columns:
            print(f"  {key:<12} no original-config run recorded - skipped")
            continue

        recorded = int(runs["metrics.best_iteration"].iloc[0])
        frozen = TREE_COUNTS[key]
        checked += 1
        if recorded == frozen:
            print(f"  {key:<12} {frozen:>4} trees  OK")
        else:
            mismatched += 1
            print(f"  {key:<12} {frozen:>4} trees  MISMATCH - MLflow recorded {recorded}")

    if mismatched:
        print(
            f"\n{mismatched} of {checked} counts disagree with MLflow. The training "
            "scripts have\nmoved since these were frozen - reconcile before shipping "
            "these models."
        )
    elif checked:
        print(f"\nAll {checked} verifiable counts match their recorded runs.")


def train_final(label: str, target: str, classification: bool, trees: int, df):
    params = final_params(classification, trees)
    estimator = XGBClassifier if classification else XGBRegressor

    model = estimator(**params)
    model.fit(df[FEATURE_COLUMNS], df[target], verbose=False)
    return model, params


def log_production_run(key, label, target, classification, model, params, path, sample, rows):
    with mlflow.start_run(run_name=key):
        mlflow.set_tags(
            {
                "stage": "final",
                "selection_experiment": key,
                "trained_on": "full dataset, no holdout",
                "evaluated": "no - no holdout remains by design",
            }
        )
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "target": target,
                "label": label,
                "task": "classification" if classification else "regression",
                "n_features": len(FEATURE_COLUMNS),
                "training_rows": rows,
                "model_file": path.name,
            }
        )

        # No metrics: see the module docstring. Nothing here has been scored.
        predictions = model.predict_proba(sample) if classification else model.predict(sample)
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=infer_signature(sample, predictions),
            input_example=sample,
        )
        mlflow.log_artifact(str(path))


def print_summary(rows, total_games):
    section(f"PRODUCTION MODELS (trained on all {total_games} games)")

    label_w = max(len(r["label"]) for r in rows)
    file_w = max(len(r["file"]) for r in rows)

    header = (
        f"{'TARGET':<{label_w}}  {'DEPTH':>5} {'LR':>5} {'TREES':>6}  "
        f"{'FILE':<{file_w}}  SAVED"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<{label_w}}  {r['max_depth']:>5} {r['learning_rate']:>5} "
            f"{r['trees']:>6}  {r['file']:<{file_w}}  {r['saved']}"
        )

    print(f"\n{len(rows)} models written to {MODELS_DIR}")
    print(f"Logged to MLflow experiment '{PRODUCTION_EXPERIMENT}' with stage=final.")
    print("No metrics logged - these models have no holdout set to score against.")


def main():
    section("DATA PREP")
    df = load_dataset()
    print(f"Loaded {len(df)} games - training on all of them, no split.")
    print("Incomplete rolling windows kept: XGBoost handles NaN natively.")

    verify_tree_counts()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sample = df[FEATURE_COLUMNS].head(5)

    setup_mlflow(PRODUCTION_EXPERIMENT)

    section("TRAINING")
    summary = []
    for key, label, target, classification in TASKS:
        trees = TREE_COUNTS[key]
        model, params = train_final(label, target, classification, trees, df)

        path = MODELS_DIR / f"{key}.json"
        model.save_model(str(path))

        log_production_run(
            key, label, target, classification, model, params, path, sample, len(df)
        )

        print(f"  {label:<12} {trees:>4} trees -> {path.name}")
        summary.append(
            {
                "label": label,
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "trees": trees,
                "file": path.name,
                "saved": "OK" if path.exists() else "FAILED",
            }
        )

    print_summary(summary, len(df))


if __name__ == "__main__":
    main()
