"""
Retrain the six Q1 / first-half models on the complete dataset and ship them.

  input:  data-pipeline/data/processed/model_dataset.csv
  output: ml-training/models_quarter_half/<key>.joblib
          ml-training/models_quarter_half/manifest.json
          one MLflow run per model under "production_quarter_half"

Same shape as finalize_models.py - selection is over, so the holdout has
done its job and holding seasons back now would only ship models that ignore
them. No metrics are logged: there is nothing left to score against, and a
number here would be misread as validation of the shipped model.

THESE ARE LINEAR MODELS, NOT XGBOOST, and that is the finding rather than a
shortcut. train_quarter_half_xgb.py trained trees on 1,266 more rows per
target and lost on all six, by +0.2% to +0.9%. When a more expressive model
class with strictly more data cannot match a linear one, shipping the linear
one is what the evidence says. Three practical consequences follow, and each
changes something about this script:

  1. A SCALER IS PART OF THE MODEL. XGBoost needed none; these are
     meaningless without one. So each artifact is a Pipeline holding the
     fitted StandardScaler and the estimator together. Saving them as two
     files would let them drift apart, which is the same class of mistake as
     restating a constant instead of importing it.

  2. THEY CANNOT TAKE NaN. XGBoost trains on all 13,199 games; these train
     on the 11,411 with complete rolling windows. At serving time a fixture
     whose trailing Q1/1H window is incomplete - early in a season - CANNOT
     BE SCORED AT ALL, rather than scored worse. The caller has to handle
     that, the same way MAX_DAYS_AHEAD is handled: refuse and say why.

  3. THE TWO WINNER MODELS ARE CONDITIONAL ON A DECIDED PERIOD. They are
     trained only on games where the period had a winner, because 611 tied
     first quarters have no binary label. So a predicted probability means
     "P(home leads | the period is not tied)". A sportsbook pushes a tied
     period, so this matches the real market - but it is not the same
     quantity as the full-game moneyline, and must not be displayed as if
     it were.

WRITTEN TO models_quarter_half/, DELIBERATELY NOT models/. The inference
service globs models/*.json and refuses to boot unless the stems match its
seven-entry MODEL_REGISTRY exactly. Adding files there would break the
running app on startup - the guard doing its job, at the worst moment. A
separate directory keeps the shipped contract untouched until the serving
work deliberately extends it.

Q1_WINNER SHIPS, LABELLED WEAK RATHER THAN OMITTED. At 0.5796 accuracy
against a 0.5184 naive it is the weakest of the six, and close enough to a
coin flip that presenting it beside 1H winner's 0.6343 without qualification
would misrepresent it. It is still genuinely better than chance: a constant
base-rate predictor scores 0.6925 log loss and this scores 0.6670, a 3.7%
gain. So it is shipped with confidence="low" in the manifest - a field the
API and frontend can read, not a comment they cannot. Same principle as the
stale badge and the disabled-with-reason schedule cards: visible and
honestly weak beats hidden.

Run:  python ml-training/finalize_quarter_half_models.py
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import setup_mlflow
from train_baseline import load_dataset, section
from train_quarter_half_baseline import (
    CLASSIFICATION_TARGETS,
    FEATURE_COLUMNS,
    REGRESSION_TARGETS,
)
from train_regression_xgb import experiment_name

MODELS_DIR = Path(__file__).resolve().parent / "models_quarter_half"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
PRODUCTION_EXPERIMENT = "production_quarter_half"

# Measured on the 2024-2025 test set by train_quarter_half_baseline.py, and
# frozen here for the same reason finalize_models.py freezes tree counts:
# mlruns/ is gitignored, so a fresh clone cannot look them up. They describe
# the SELECTION run, never these models - these have no holdout.
#
# naive is the beaten baseline: ROLL10 diff/sum for regression, always-home
# for classification.
SELECTION_METRICS = {
    "q1_spread": {"metric": "MAE", "naive": 7.03, "model": 6.61},
    "q1_total": {"metric": "MAE", "naive": 6.88, "model": 6.58},
    "1h_spread": {"metric": "MAE", "naive": 9.59, "model": 8.91},
    "1h_total": {"metric": "MAE", "naive": 10.20, "model": 9.78},
    "q1_winner": {"metric": "accuracy", "naive": 0.5184, "model": 0.5796,
                  "log_loss": 0.6670},
    "1h_winner": {"metric": "accuracy", "naive": 0.5291, "model": 0.6343,
                  "log_loss": 0.6462},
}

# How much of a prediction to believe, as a field rather than a footnote.
# Only q1_winner is "low"; see the module docstring for the decision.
CONFIDENCE = {
    "q1_spread": "medium",
    "q1_total": "medium",
    "1h_spread": "medium",
    "1h_total": "medium",
    "q1_winner": "low",
    "1h_winner": "medium",
}

CONFIDENCE_NOTES = {
    "low": ("Barely above the naive baseline - a single quarter is mostly "
            "short-window variance. Present with a visible caveat."),
    "medium": ("A real, modest gain over the naive baseline. No model in "
               "this set is strong; none beat a linear fit."),
}

# (key, label, target, is_classification), built from the same lists the
# baseline and XGBoost scripts use so a target cannot be silently skipped.
TASKS = [
    (experiment_name(label), label, target, False)
    for target, label, _stat, _combine in REGRESSION_TARGETS
] + [
    (experiment_name(label), label, target, True)
    for target, label, _period in CLASSIFICATION_TARGETS
]


def training_rows(df, target: str):
    """Rows this model can actually learn from.

    Two filters, both forced by the model class rather than chosen:
    complete features because linear models reject NaN inputs, and a present
    label because a tied period has no binary answer. See the docstring.
    """
    return df.dropna(subset=FEATURE_COLUMNS).dropna(subset=[target])


def build_pipeline(classification: bool) -> Pipeline:
    """Scaler and estimator as one artifact - see docstring point 1."""
    estimator = (LogisticRegression(max_iter=1000) if classification
                 else LinearRegression())
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def train_final(target: str, classification: bool, df):
    rows = training_rows(df, target)
    pipeline = build_pipeline(classification)
    y = rows[target].astype(int) if classification else rows[target]
    pipeline.fit(rows[FEATURE_COLUMNS], y)
    return pipeline, len(rows)


def log_production_run(key, label, target, classification, pipeline, path,
                       sample, rows):
    with mlflow.start_run(run_name=key):
        mlflow.set_tags({
            "stage": "final",
            "selection_experiment": key,
            "trained_on": "full dataset, no holdout",
            "evaluated": "no - no holdout remains by design",
            "model_class": "LogisticRegression" if classification
                           else "LinearRegression",
            "beat_xgboost": "yes - see train_quarter_half_xgb.py",
        })
        mlflow.log_params({
            "target": target,
            "label": label,
            "task": "classification" if classification else "regression",
            "n_features": len(FEATURE_COLUMNS),
            "training_rows": rows,
            "model_file": path.name,
            "confidence": CONFIDENCE[key],
        })

        # No metrics on purpose, see module docstring.
        predictions = (pipeline.predict_proba(sample) if classification
                       else pipeline.predict(sample))
        mlflow.sklearn.log_model(
            pipeline, name="model",
            signature=infer_signature(sample, predictions),
            input_example=sample,
        )
        mlflow.log_artifact(str(path))


def write_manifest(entries, total_rows):
    """Everything a serving layer needs, without importing this package.

    The confidence field is the point: it travels with the model instead of
    living in a docstring the API cannot read.
    """
    manifest = {
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "requires_complete_features": True,
        "note": ("Linear models - a row with any NaN feature cannot be "
                 "scored and must be refused, not imputed."),
        "models": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {MANIFEST_PATH}")


def print_summary(entries, total_games):
    section("PRODUCTION MODELS (Q1 / first half)")
    label_w = max(len(e["label"]) for e in entries)

    header = (f"{'TARGET':<{label_w}}  {'CLASS':<20}{'ROWS':>7}{'CONF':>9}  "
              f"{'FILE':<18}SAVED")
    print(header)
    print("-" * len(header))
    for e in entries:
        print(f"{e['label']:<{label_w}}  {e['model_class']:<20}"
              f"{e['training_rows']:>7,}{e['confidence']:>9}  "
              f"{e['file']:<18}{e['saved']}")

    print(f"\n{len(entries)} models written to {MODELS_DIR}")
    print(f"Trained on the {entries[0]['training_rows']:,}-{entries[-1]['training_rows']:,} "
          f"of {total_games:,} games with complete rolling windows "
          f"(linear models reject NaN).")
    print(f"Logged to MLflow experiment '{PRODUCTION_EXPERIMENT}', stage=final.")
    print("No metrics logged - these models have no holdout set to score against.")

    print("\nConfidence tiers:")
    for tier in ("medium", "low"):
        names = [e["label"] for e in entries if e["confidence"] == tier]
        if names:
            print(f"  {tier:<8}{', '.join(names)}")
            print(f"          {CONFIDENCE_NOTES[tier]}")


def main():
    section("DATA PREP")
    df = load_dataset()
    complete = df.dropna(subset=FEATURE_COLUMNS)
    print(f"Loaded {len(df):,} games; {len(complete):,} have complete Q1/1H "
          f"rolling windows.")
    print("Unlike finalize_models.py, the unusable rows cannot be kept: these")
    print("are linear models, and NaN is not an input they accept.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sample = complete[FEATURE_COLUMNS].head(5)
    setup_mlflow(PRODUCTION_EXPERIMENT)

    section("TRAINING")
    entries = []
    for key, label, target, classification in TASKS:
        pipeline, rows = train_final(target, classification, df)

        path = MODELS_DIR / f"{key}.joblib"
        joblib.dump(pipeline, path)

        log_production_run(key, label, target, classification, pipeline, path,
                           sample, rows)

        print(f"  {label:<12} {rows:>6,} rows -> {path.name}")
        entries.append({
            "key": key,
            "label": label,
            "target": target,
            "task": "classification" if classification else "regression",
            "model_class": type(pipeline.named_steps["model"]).__name__,
            "training_rows": rows,
            "file": path.name,
            "confidence": CONFIDENCE[key],
            "confidence_note": CONFIDENCE_NOTES[CONFIDENCE[key]],
            "selection": SELECTION_METRICS[key],
            "conditional_on_decided_period": classification,
            "saved": "OK" if path.exists() else "FAILED",
        })

    write_manifest(entries, len(df))
    print_summary(entries, len(df))


if __name__ == "__main__":
    main()
