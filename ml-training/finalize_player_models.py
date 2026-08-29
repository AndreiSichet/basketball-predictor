"""
Ship the five player-prop targets as a HYBRID: two models per target, plus a
rule for choosing between them.

  input:  data-pipeline/data/processed/player_dataset.csv
  output: ml-training/models_player_props/<target>_linear.joblib
          ml-training/models_player_props/<target>_xgb.json
          ml-training/models_player_props/manifest.json
          one MLflow run per artifact under "production_player_props"

WHY TWO MODELS, WHEN NEITHER WINS. Every prior finalize_* script here ships
one model per target because one model was better. That is not the situation.
Across five targets XGBoost moved MAE by -0.63% to +0.13% against
LinearRegression - nothing, against a 3% bar. What separates them is not
accuracy but COVERAGE:

    LinearRegression   225,060 of 280,943 rows   cannot consume NaN
    XGBoost            280,943 of 280,943 rows   learns a default split

The ~55,900 rows the linear model cannot touch are real players in real
games - rookies, call-ups, returns from long absence - whose trailing
history is incomplete. Refusing to predict for them is a worse answer than
predicting slightly less well. So the linear model handles the rows it was
measured on, and XGBoost covers the rest.

THE ROUTING RULE, stated once so it cannot drift:

    all 12 rolling features present  ->  <target>_linear
    any of them missing             ->  <target>_xgb

It is in manifest.json as data, not prose, for the same reason the
quarter/half confidence field is: a serving layer can read a manifest and
cannot read a docstring.

TWELVE COLUMNS DECIDE IT, NOT SEVENTEEN - and that is verified, not
assumed. The five team-context columns are not all guaranteed present:
REST_DAYS is legitimately NaN for a team's first appearance in the data. A
row with complete rolling history but a NaN REST_DAYS would be routed to the
linear model, which would then raise. Measured on the real dataset, that row
does not exist - all 319 context-NaN rows are a strict SUBSET of the 55,883
rolling-incomplete rows, because a team's first game is necessarily also a
player's first game for that team. So the twelve-column rule is sufficient.

  But that is a property of this data, not a law. assert_routing_is_safe()
  re-checks it on every run and refuses to ship if it ever stops holding,
  rather than letting a future rebuild produce a router that crashes on one
  row in a hundred thousand.

TREE COUNTS COME FROM MLFLOW, NOT FROM MEMORY. Frozen below because mlruns/
is gitignored and a fresh clone could not look them up, then checked against
the recorded runs on every execution - the same arrangement finalize_models.py
uses, and the same standard applied when §18's figures were recovered rather
than recalled.

NO METRICS ARE LOGGED. Selection is over and these train on everything, so
there is no holdout left to score against; a number here would be misread as
validation of the shipped model.

Run:  python ml-training/finalize_player_models.py
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import TRACKING_URI, setup_mlflow

# The filter, the targets and the loader come from the script that defined
# them. drop_incomplete() in particular IS the definition of "complete" this
# whole comparison was built on - restating it here would let the shipped
# router disagree with the measurement that justified it.
from train_player_baseline import (
    TARGET_LABELS,
    TARGETS,
    check_no_leakage,
    drop_incomplete,
    load_dataset,
    section,
)
from train_player_xgb import PARAMS as XGB_PARAMS

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "data-pipeline" / "preprocessing"
sys.path.insert(0, str(PIPELINE_DIR))
from build_player_dataset import (  # noqa: E402
    FEATURE_COLUMNS,
    PLAYER_FEATURE_COLUMNS,
    TEAM_CONTEXT_COLUMNS,
)

MODELS_DIR = Path(__file__).resolve().parent / "models_player_props"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
PRODUCTION_EXPERIMENT = "production_player_props"

# Early-stopping results from the real train_player_xgb.py run, under
# max_depth=4 / learning_rate=0.05. Frozen for a fresh clone; verified
# against MLflow by verify_tree_counts() on every run.
TREE_COUNTS = {"PTS": 158, "REB": 363, "AST": 140, "FG3M": 144, "PRA": 372}

# What the baselines were measured on. Asserted, not trusted.
EXPECTED_COMPLETE_ROWS = 225_060

LINEAR_KEY, XGB_KEY = "linear", "xgb"


def artifact_paths(target: str) -> tuple:
    stem = target.lower()
    return (MODELS_DIR / f"{stem}_{LINEAR_KEY}.joblib",
            MODELS_DIR / f"{stem}_{XGB_KEY}.json")


def select_model(row: pd.Series) -> str:
    """THE ROUTING RULE. Returns "linear" or "xgb" for one player-game.

    Takes a Series holding at least the 12 rolling features. See the module
    docstring on why twelve and not seventeen.
    """
    complete = row[PLAYER_FEATURE_COLUMNS].notna().all()
    return LINEAR_KEY if complete else XGB_KEY


def assert_routing_is_safe(df: pd.DataFrame) -> dict:
    """Refuse to ship a router that could hand NaN to the linear model.

    The rule reads 12 columns but the linear model consumes 17. That is only
    safe while every context-NaN row is also rolling-incomplete. Checked
    here rather than assumed - see the module docstring.
    """
    warmup = df[PLAYER_FEATURE_COLUMNS].isna().any(axis=1)
    context = df[TEAM_CONTEXT_COLUMNS].isna().any(axis=1)
    unsafe = int(((~warmup) & context).sum())

    print(f"  rolling incomplete            : {int(warmup.sum()):>8,}")
    print(f"  team context NaN              : {int(context.sum()):>8,}")
    print(f"  context NaN AND rolling ok    : {unsafe:>8,}  <- must be 0")

    if unsafe:
        raise RuntimeError(
            f"{unsafe:,} rows have complete rolling history but NaN team "
            f"context. The 12-column routing rule would send them to the "
            f"linear model, which cannot consume NaN. Either widen the rule "
            f"to all {len(FEATURE_COLUMNS)} feature columns or fix the "
            f"upstream gap before shipping."
        )
    print("  -> the 12-column rule is sufficient on this dataset.")
    return {"rolling_incomplete": int(warmup.sum()),
            "context_nan": int(context.sum()),
            "unsafe_rows": unsafe}


def verify_tree_counts() -> None:
    """Check the frozen counts against the MLflow runs they came from."""
    section("TREE COUNT VERIFICATION (frozen vs recorded)")
    mlflow.set_tracking_uri(TRACKING_URI)

    checked = mismatched = 0
    for target in TARGETS:
        try:
            runs = mlflow.search_runs(
                experiment_names=[f"player_{target.lower()}"],
                order_by=["start_time DESC"], max_results=1,
            )
        except Exception as error:  # noqa: BLE001 - advisory, never fatal
            print(f"  {target:<6} could not query MLflow "
                  f"({type(error).__name__}) - skipped")
            continue

        if len(runs) == 0 or "metrics.best_iteration" not in runs.columns:
            print(f"  {target:<6} no recorded run - skipped")
            continue

        recorded = int(runs["metrics.best_iteration"].iloc[0])
        frozen = TREE_COUNTS[target]
        checked += 1
        if recorded == frozen:
            print(f"  {target:<6} {frozen:>4} trees  OK")
        else:
            mismatched += 1
            print(f"  {target:<6} {frozen:>4} trees  MISMATCH - "
                  f"MLflow recorded {recorded}")

    if mismatched:
        raise RuntimeError(
            f"{mismatched} of {checked} tree counts disagree with MLflow. "
            f"These models would not be the ones that were measured."
        )
    if checked:
        print(f"\n  All {checked} counts match their recorded runs.")


def train_linear(complete: pd.DataFrame, target: str) -> Pipeline:
    """Scaler and estimator as one artifact, as for the quarter/half models."""
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("model", LinearRegression())])
    pipeline.fit(complete[FEATURE_COLUMNS], complete[target])
    return pipeline


def train_xgb(df: pd.DataFrame, target: str) -> tuple:
    """Trained on EVERY row - the reason this side of the hybrid exists."""
    params = {k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"}
    params["n_estimators"] = TREE_COUNTS[target]

    model = XGBRegressor(**params)
    model.fit(df[FEATURE_COLUMNS], df[target], verbose=False)
    return model, params


def log_production_run(target, kind, model, params, path, sample, rows):
    with mlflow.start_run(run_name=f"{target.lower()}_{kind}"):
        mlflow.set_tags({
            "stage": "final",
            "trained_on": "full dataset, no holdout",
            "evaluated": "no - no holdout remains by design",
            "hybrid_role": ("complete rolling history" if kind == LINEAR_KEY
                            else "fallback for incomplete history"),
        })
        mlflow.log_params(params)
        mlflow.log_params({
            "target": target, "label": TARGET_LABELS[target], "kind": kind,
            "n_features": len(FEATURE_COLUMNS), "training_rows": rows,
            "model_file": path.name,
        })
        predictions = model.predict(sample)
        logger = mlflow.sklearn if kind == LINEAR_KEY else mlflow.xgboost
        logger.log_model(model, name="model",
                         signature=infer_signature(sample, predictions),
                         input_example=sample)
        mlflow.log_artifact(str(path))


def write_manifest(entries, counts, total_rows, complete_rows):
    manifest = {
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "routing": {
            "decide_on": PLAYER_FEATURE_COLUMNS,
            "rule": ("if every column in decide_on is present, use the "
                     "'linear' model; otherwise use the 'xgb' model"),
            "linear_requires_complete_features": True,
            "xgb_handles_missing": True,
            "why": ("The two models are equivalent on accuracy (-0.63% to "
                    "+0.13% MAE). They differ in coverage: linear was fit on "
                    f"{complete_rows:,} rows and rejects NaN, xgb on all "
                    f"{total_rows:,} and does not."),
            "context_columns_not_in_rule": TEAM_CONTEXT_COLUMNS,
            "context_exclusion_verified": (
                f"REST_DAYS can be NaN, so the rule reads 12 of "
                f"{len(FEATURE_COLUMNS)} columns. Verified safe: all "
                f"{counts['context_nan']} context-NaN rows are also "
                f"rolling-incomplete, so none can reach the linear model. "
                f"Re-checked on every build."),
        },
        "models": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {MANIFEST_PATH}")


def exercise_router(df: pd.DataFrame, entries) -> None:
    """Run the rule on real rows, end to end, rather than assert it on paper."""
    section("ROUTING EXERCISE (real rows, real artifacts)")

    loaded = {}
    for e in entries:
        loaded[(e["target"], LINEAR_KEY)] = joblib.load(
            MODELS_DIR / e["files"][LINEAR_KEY])
        booster = XGBRegressor()
        booster.load_model(str(MODELS_DIR / e["files"][XGB_KEY]))
        loaded[(e["target"], XGB_KEY)] = booster

    complete_mask = df[PLAYER_FEATURE_COLUMNS].notna().all(axis=1)

    # A player-game with full trailing history.
    veteran = df[complete_mask].iloc[0]
    # A genuine first appearance: every rolling column missing.
    debut = df[df[PLAYER_FEATURE_COLUMNS].isna().all(axis=1)].iloc[0]

    for name, row in (("complete history", veteran), ("first appearance", debut)):
        chosen = select_model(row)
        expected = LINEAR_KEY if name == "complete history" else XGB_KEY
        missing = int(row[PLAYER_FEATURE_COLUMNS].isna().sum())
        print(f"\n  {name}: {row['PLAYER_NAME']} on "
              f"{pd.to_datetime(row['GAME_DATE']).date()}")
        print(f"    rolling features missing : {missing} of "
              f"{len(PLAYER_FEATURE_COLUMNS)}")
        print(f"    router selected          : {chosen}  "
              f"({'correct' if chosen == expected else 'WRONG'})")

        frame = row[FEATURE_COLUMNS].to_frame().T.astype(float)
        for target in TARGETS:
            model = loaded[(target, chosen)]
            value = float(model.predict(frame)[0])
            print(f"      {TARGET_LABELS[target]:<16}{value:7.2f}   "
                  f"(via {chosen})")
        if chosen == LINEAR_KEY and missing:
            raise RuntimeError("router sent an incomplete row to the linear model")


def compare_at_boundary(df: pd.DataFrame, entries) -> None:
    """Both models on the first row that just barely qualifies as complete.

    Not pass/fail. The two models are trained on different row populations,
    so they SHOULD differ; the point is to see how much, at the exact point
    where the router flips, before trusting the handover blind.
    """
    section("BOUNDARY COMPARISON (the row where routing flips)")

    complete = df[PLAYER_FEATURE_COLUMNS].notna().all(axis=1)
    ordered = df.assign(_complete=complete).sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"])

    # First qualifying appearance for a player who has enough games to make
    # the boundary meaningful rather than an artefact of a short career.
    first_complete = ordered[ordered["_complete"]].groupby("PLAYER_ID").head(1)
    counts = ordered.groupby("PLAYER_ID").size()
    eligible = first_complete[first_complete["PLAYER_ID"].map(counts) > 40]
    row = eligible.iloc[0]

    prior = ordered[(ordered["PLAYER_ID"] == row["PLAYER_ID"])
                    & (ordered["GAME_DATE"] < row["GAME_DATE"])]
    print(f"  {row['PLAYER_NAME']} on "
          f"{pd.to_datetime(row['GAME_DATE']).date()}")
    print(f"  prior appearances in the data: {len(prior)} "
          f"(ROLL10 needs 10, so this is the first game that qualifies)")
    print(f"  router selects: {select_model(row)}\n")

    frame = row[FEATURE_COLUMNS].to_frame().T.astype(float)
    print(f"  {'TARGET':<16}{'LINEAR':>9}{'XGB':>9}{'DIFF':>9}{'ACTUAL':>9}")
    print("  " + "-" * 52)
    for e in entries:
        target = e["target"]
        lin = float(joblib.load(MODELS_DIR / e["files"][LINEAR_KEY])
                    .predict(frame)[0])
        booster = XGBRegressor()
        booster.load_model(str(MODELS_DIR / e["files"][XGB_KEY]))
        xgb = float(booster.predict(frame)[0])
        print(f"  {TARGET_LABELS[target]:<16}{lin:>9.2f}{xgb:>9.2f}"
              f"{xgb - lin:>+9.2f}{row[target]:>9.0f}")

    print("\n  Divergence here is expected, not a fault: the two are fit on")
    print("  different row populations. This is a look at the handover, not a test.")


def main():
    section("DATA PREP")
    check_no_leakage()
    df = load_dataset()

    print("\nRouting safety:")
    counts = assert_routing_is_safe(df)

    complete = drop_incomplete(df)
    if len(complete) != EXPECTED_COMPLETE_ROWS:
        raise RuntimeError(
            f"complete-history rows: {len(complete):,}, expected "
            f"{EXPECTED_COMPLETE_ROWS:,}. The filter this router ships with "
            f"no longer matches the one the baselines were measured on."
        )
    print(f"Complete-history rows reproduce the baseline set exactly: "
          f"{len(complete):,}")

    verify_tree_counts()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sample = complete[FEATURE_COLUMNS].head(5)
    setup_mlflow(PRODUCTION_EXPERIMENT)

    section("TRAINING (two models per target)")
    entries = []
    for target in TARGETS:
        linear_path, xgb_path = artifact_paths(target)

        pipeline = train_linear(complete, target)
        joblib.dump(pipeline, linear_path)
        log_production_run(target, LINEAR_KEY, pipeline,
                           {"model_class": "LinearRegression"},
                           linear_path, sample, len(complete))

        model, params = train_xgb(df, target)
        model.save_model(str(xgb_path))
        log_production_run(target, XGB_KEY, model, params, xgb_path, sample,
                           len(df))

        print(f"  {TARGET_LABELS[target]:<16}"
              f"linear {len(complete):>7,} rows -> {linear_path.name:<20}"
              f"xgb {len(df):>7,} rows -> {xgb_path.name}")

        entries.append({
            "target": target,
            "label": TARGET_LABELS[target],
            "files": {LINEAR_KEY: linear_path.name, XGB_KEY: xgb_path.name},
            "training_rows": {LINEAR_KEY: len(complete), XGB_KEY: len(df)},
            "model_class": {LINEAR_KEY: "Pipeline(StandardScaler, LinearRegression)",
                            XGB_KEY: "XGBRegressor"},
            "trees": TREE_COUNTS[target],
            "saved": all(p.exists() for p in (linear_path, xgb_path)),
        })

    write_manifest(entries, counts, len(df), len(complete))

    section("ARTIFACTS LOAD BACK FROM DISK")
    row = complete[FEATURE_COLUMNS].head(1)
    for e in entries:
        lin = joblib.load(MODELS_DIR / e["files"][LINEAR_KEY])
        booster = XGBRegressor()
        booster.load_model(str(MODELS_DIR / e["files"][XGB_KEY]))
        print(f"  {e['label']:<16}linear {float(lin.predict(row)[0]):7.2f}   "
              f"xgb {float(booster.predict(row)[0]):7.2f}   both OK")

    exercise_router(df, entries)
    compare_at_boundary(df, entries)

    section("SUMMARY")
    print(f"{len(entries) * 2} artifacts + manifest written to {MODELS_DIR}")
    print(f"Logged to MLflow experiment '{PRODUCTION_EXPERIMENT}', stage=final.")
    print("No metrics logged - these models have no holdout set to score against.")
    print("\nThis is the first hybrid in the project: the manifest's routing")
    print("block, not the caller, defines which model answers a given request.")


if __name__ == "__main__":
    main()
