"""
Load a saved model and score it against a given dataframe.

EXTRACTED AT THE THIRD CALL SITE, not the second. train_moneyline_xgb.py and
train_regression_xgb.py each grew their own copy of "predict, then compute
the metrics" - two copies is a judgement call. continuous_retrain.py is the
third, and it needs to score MODELS LOADED FROM DISK rather than ones it
just trained, which neither existing copy does. Same threshold that
justified InferenceClient and GameLookup on the Java side.

  Honest note: the two training scripts have NOT been retrofitted onto this
  module. Their numbers are on record throughout CLAUDE.md, and swapping
  their scoring path is a change that would need those runs repeated to
  verify. Doing it is safe and worth doing; doing it silently as a side
  effect of building the retrain job is not.

THE METRIC PER TARGET IS FIXED HERE, deliberately, because a promotion gate
that let the caller pick its own metric could be made to pass by choosing a
kinder one. Moneyline is judged on LOG LOSS rather than accuracy: it is a
probability model, accuracy throws away everything except which side of 0.5
a prediction fell on, and CLAUDE.md section 16 already records a case where
the two moved in opposite directions.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

from common import FEATURE_COLUMNS
from train_baseline import REGRESSION_TARGETS
from train_moneyline_xgb import TARGET as MONEYLINE_TARGET

MODELS_DIR = Path(__file__).resolve().parent / "models"

# (file stem, label, target column, is_classification) for all seven shipped
# models. Built from REGRESSION_TARGETS rather than typed out, so a target
# added there cannot be silently skipped here - the same construction
# finalize_models.py uses.
TASKS = [("moneyline", "Moneyline", MONEYLINE_TARGET, True)] + [
    (label.lower().replace(" ", "_"), label, target, False)
    for target, label, _stat, _combine in REGRESSION_TARGETS
]

# Lower is better for both, which is what lets the gate compare them with
# one rule instead of two.
PRIMARY_METRIC = {True: "log_loss", False: "mae"}


def load_model(key: str, classification: bool, directory: Path = None):
    """One saved model, as the sklearn-API estimator that wrote it.

    XGBClassifier/XGBRegressor rather than a raw Booster, matching how the
    inference service loads them: the booster API would happily accept a
    mis-ordered feature vector, and these keep the feature-name check.
    """
    directory = directory or MODELS_DIR
    path = directory / f"{key}.json"
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}")

    model = XGBClassifier() if classification else XGBRegressor()
    model.load_model(str(path))
    return model


def evaluate(model, frame, target: str, classification: bool) -> dict:
    """Score one model on one dataframe. Never trains, never mutates.

    The frame is used as-is: no rows dropped, no NaN filled. XGBoost handles
    missing values natively, and silently filtering here would mean the
    candidate and production were judged on different rows.
    """
    x = frame[FEATURE_COLUMNS]
    y = frame[target]

    if classification:
        proba = model.predict_proba(x)[:, 1]
        return {
            "log_loss": float(log_loss(y, proba)),
            "accuracy": float(accuracy_score(y, model.predict(x))),
        }

    predictions = model.predict(x)
    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
    }


def primary(metrics: dict, classification: bool) -> float:
    """The one number the gate compares. See the module docstring."""
    return metrics[PRIMARY_METRIC[classification]]
