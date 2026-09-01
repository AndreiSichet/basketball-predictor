"""
Retrain the seven team-level models on current data, and refuse to promote
anything that got worse.

  output: ml-training/models_candidate/<target>.json  (only if the gate passes)

NEVER OVERWRITES ml-training/models/. Passing candidates land in a separate
directory; deciding to ship them is a human's job, and in the eventual
GitHub Actions wrapper that decision surfaces as a pull request. This script
does the work and reports; it does not deploy.

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER COMPARISON IN THIS PROJECT

Every prior model comparison here asked "does this new idea earn a place" -
new features, a different model family, a tuning grid - and held the answer
to a >3-5% improvement bar. This asks something else entirely: "does the
model still work, now that there is more data". Retraining is not a
hypothesis. So the gate is a REGRESSION GUARD, one-directional and loose:
a candidate may be slightly worse and still promote, because month-to-month
noise is real and a fresh model reflecting current reality is worth a
fraction of a percent. What it may not do is degrade meaningfully.

PRODUCTION IS RE-FIT, NOT RE-SCORED. THIS IS THE CORRECTNESS FIX.

The obvious version of this gate loads the shipped .json files and scores
them on the fresh window. That is provably wrong here, and the bias was
measured rather than suspected: finalize_models.py trains the shipped
models on the ENTIRE dataset with no holdout - deliberately, since
selection was already finished - so any window drawn from that dataset is
part of their training data. Scored that way, production returned 9.58 MAE
on spread over the last 300 games against the 10.74 it scored on a genuine
holdout. It was being graded on its own homework, and a candidate that
honestly held the window out looked worse by comparison. Every target
appeared to regress by 5-12% for that reason alone.

So the gate does not score the shipped weights. It takes the shipped
model's ARCHITECTURE - the same config finalize_models.py uses, plus the
tree count read back off the artifact itself - and re-fits it on exactly
the training rows the candidate gets. Both models are then fit on the same
data and scored on rows neither has seen. That is the only version of this
comparison that measures what it claims to.

  The shipped files on disk are never touched. The re-fit copy exists
  only inside the comparison and is never written anywhere.

  Residual asymmetry, stated plainly: the candidate uses the validation
  window to choose its tree count by early stopping, while production's
  count is already fixed. That is symmetric in kind - production's count
  was chosen the same way when it was trained - but it is not identical.

THE SPLIT IS ROLLING, not season-based. common.split_three_way is a fixed
historical boundary - train through 2022, validate 2023, test 2024-25 -
which is right for a one-time study and wrong for a job that runs
repeatedly: the test window would never move, and eventually the model
would be trained on data that overlaps it. Here the most recent games are
the test window, the ones before that are validation for early stopping,
and everything earlier is training.

THE TRIGGER IS ACCUMULATED DATA, NOT A CALENDAR. Running weekly on a bare
timer would mean retraining on almost nothing new - about 50 games arrive
per week against a 500-game window. Nothing else in this project runs on a
clock when the real condition is "enough new data exists"; the ingestion
scripts are skip-and-resume for exactly that reason. So this checks how
many games have arrived since production's own training cutoff and skips
with an honest message below the threshold.

  Production's cutoff is read from a marker file written beside the models
  when candidates are produced. Before the first promotion there is no
  marker, and the check says so rather than guessing.

SCOPE: the seven team-level models only. Quarter/half and player props are
deliberately out - same "smallest defensible slice first" discipline as
Q1-and-1H-only and PACE/TS_PCT-only before them. Player-level data IS still
refreshed, because availability is part of the shipped 38 features.

Run:  python ml-training/continuous_retrain.py [--skip-refresh]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from common import FEATURE_COLUMNS
from finalize_models import final_params
from model_evaluation import TASKS, evaluate, load_model, primary
from train_baseline import load_dataset, section
from train_moneyline_xgb import PARAMS as MONEYLINE_PARAMS
from train_regression_xgb import PARAMS as REGRESSION_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "data-pipeline"
GAMES_FINAL = PIPELINE / "data" / "processed" / "games_final.csv"

PRODUCTION_DIR = Path(__file__).resolve().parent / "models"
CANDIDATE_DIR = Path(__file__).resolve().parent / "models_candidate"

# Written beside a set of models, recording the data they were trained on.
# Without it the "how much is new" question has no answer.
TRAINED_THROUGH = "trained_through.json"

# The pipeline, in dependency order. Ingestion first - both fetchers skip
# what is already on disk, so a run pays only for genuinely new games, not
# a full re-pull. Quarter scores and advanced stats are deliberately absent:
# out of scope, and re-fetching either would cost hours to feed models this
# job does not touch.
REFRESH_STEPS = [
    PIPELINE / "ingestion" / "fetch_games.py",
    PIPELINE / "ingestion" / "fetch_player_boxscores.py",
    PIPELINE / "preprocessing" / "validate_games.py",
    PIPELINE / "preprocessing" / "build_games_table.py",
    PIPELINE / "preprocessing" / "build_rolling_features.py",
    PIPELINE / "preprocessing" / "build_rest_days.py",
    PIPELINE / "preprocessing" / "build_elo_ratings.py",
    PIPELINE / "preprocessing" / "build_player_rolling_minutes.py",
    PIPELINE / "preprocessing" / "build_team_availability.py",
    PIPELINE / "preprocessing" / "build_final_dataset.py",
]

# THE TEST WINDOW, IN GAMES. model_dataset.csv is one row per game, so 500
# rows is 500 games - about 40% of a 1,230-game season. Note this differs
# from a "team-game" count, which would be half as many games; the dataset's
# grain is the game, so that is the unit used here.
#
# Big enough that one unusual week cannot swing the verdict, small enough
# that new data works into it within a season. A real parameter, not a
# constant.
DEFAULT_TEST_GAMES = 500

# How much worse a candidate may be and still promote, as a percentage of
# re-fit production's score on the same window. Deliberately loose and
# one-directional - see the module docstring on why this is not the 3-5%
# bar used for feature decisions.
DEFAULT_TOLERANCE_PCT = 2.0

# How many genuinely new games must exist before retraining is worth doing.
# Roughly two weeks of a full NBA schedule. Adjustable, and the right value
# is a judgement about how fast the signal moves, not a fact.
DEFAULT_MIN_NEW_GAMES = 100


def snapshot(path: Path) -> dict:
    """Enough of games_final.csv to tell whether anything new arrived."""
    if not path.exists():
        return {"rows": 0, "latest": None}
    frame = pd.read_csv(path, usecols=["GAME_ID", "GAME_DATE"])
    return {
        "rows": len(frame),
        "latest": str(pd.to_datetime(frame["GAME_DATE"]).max().date()),
    }


def read_trained_through(directory: Path):
    """What data the models in this directory were trained on, or None."""
    path = directory / TRAINED_THROUGH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_trained_through(directory: Path, dataset: pd.DataFrame) -> None:
    (directory / TRAINED_THROUGH).write_text(
        json.dumps({
            "trained_through": str(dataset["GAME_DATE"].max().date()),
            "games": int(len(dataset)),
        }, indent=2),
        encoding="utf-8",
    )


def refresh_data() -> None:
    """Rerun ingestion and preprocessing, in order, stopping on failure."""
    section("REFRESHING DATA")
    for step in REFRESH_STEPS:
        print(f"\n--- {step.name}")
        result = subprocess.run(
            [sys.executable, str(step)], cwd=str(PROJECT_ROOT), check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{step.name} failed with exit code {result.returncode}. "
                f"Stopping: every later step reads what this one writes, so "
                f"continuing would train on a half-built dataset."
            )


def rolling_split(dataset: pd.DataFrame, test_games: int) -> tuple:
    """Most recent games as test, the ones before as validation, rest train."""
    ordered = dataset.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    if len(ordered) < test_games * 3:
        raise RuntimeError(
            f"{len(ordered):,} games is too few to carve a {test_games}-game "
            f"test and validation window out of and still leave a training set."
        )

    test = ordered.iloc[-test_games:]
    validation = ordered.iloc[-2 * test_games:-test_games]
    train = ordered.iloc[: -2 * test_games]

    def span(frame):
        return f"{frame['GAME_DATE'].min().date()} to {frame['GAME_DATE'].max().date()}"

    print(f"  train      {len(train):>6,} games   {span(train)}")
    print(f"  validation {len(validation):>6,} games   {span(validation)}")
    print(f"  test       {len(test):>6,} games   {span(test)}   <- the gate's window")
    return train, validation, test


def train_candidate(target: str, classification: bool, train, validation):
    """One candidate model, on the frozen architecture.

    Not re-tuned. A 2x2 sweep across all seven targets improved nothing
    meaningfully when it was run, and a scheduled job is the wrong place to
    rediscover that. Early stopping still chooses the tree count, which is
    why the validation window exists.
    """
    params = MONEYLINE_PARAMS if classification else REGRESSION_PARAMS
    model = (XGBClassifier if classification else XGBRegressor)(**params)
    model.fit(
        train[FEATURE_COLUMNS], train[target],
        eval_set=[(validation[FEATURE_COLUMNS], validation[target])],
        verbose=False,
    )
    return model


def refit_production(shipped, target: str, classification: bool, train):
    """Production's architecture, re-fit on the candidate's training rows.

    NOT the shipped weights. See the module docstring: those were fit on
    everything including the test window, so scoring them there measures
    memorisation.

    The architecture comes from finalize_models.final_params - the single
    definition of how production models are built - and the tree count is
    read back off the shipped artifact rather than from a frozen dict, so
    this reflects what is actually deployed even if that dict drifts.
    """
    trees = shipped.get_booster().num_boosted_rounds()
    params = final_params(classification, trees)
    model = (XGBClassifier if classification else XGBRegressor)(**params)
    model.fit(train[FEATURE_COLUMNS], train[target], verbose=False)
    return model, trees


def run_gate(train, validation, test, production_dir: Path,
             tolerance_pct: float) -> list:
    """Train each candidate, re-fit production on the same rows, compare."""
    results = []

    for key, label, target, classification in TASKS:
        candidate = train_candidate(target, classification, train, validation)
        candidate_metrics = evaluate(candidate, test, target, classification)

        shipped = load_model(key, classification, production_dir)
        incumbent, trees = refit_production(shipped, target, classification, train)
        incumbent_metrics = evaluate(incumbent, test, target, classification)

        # Kept only to make the bias visible in the output. NOT used by the
        # gate - this is the number that looked authoritative and was not.
        shipped_metrics = evaluate(shipped, test, target, classification)

        cand = primary(candidate_metrics, classification)
        prod = primary(incumbent_metrics, classification)
        change_pct = (cand - prod) / prod * 100          # negative = better
        passed = change_pct <= tolerance_pct

        results.append({
            "key": key, "label": label, "target": target,
            "classification": classification,
            "candidate": cand, "production": prod,
            "shipped_as_is": primary(shipped_metrics, classification),
            "change_pct": change_pct, "passed": passed,
            "candidate_trees": int(candidate.best_iteration),
            "production_trees": trees,
            "model": candidate,
        })

    return results


def print_table(results, tolerance_pct: float) -> None:
    section(f"PROMOTION GATE (a candidate may be up to {tolerance_pct:.1f}% worse)")
    print("PRODUCTION is the shipped architecture RE-FIT on the candidate's own")
    print("training rows, so neither model has seen the test window. The shipped")
    print("weights are not the comparison - see the rightmost column.\n")

    width = max(len(r["label"]) for r in results)
    print(f"{'TARGET':<{width}}  {'METRIC':<9}{'PRODUCTION':>11}{'CANDIDATE':>11}"
          f"{'CHANGE':>9}{'TREES c/p':>11}   {'VERDICT':<18}{'shipped as-is':>14}")
    print("-" * (width + 78))
    for r in results:
        metric = "log loss" if r["classification"] else "MAE"
        verdict = "pass" if r["passed"] else "FAIL - regressed"
        trees = f"{r['candidate_trees']}/{r['production_trees']}"
        print(f"{r['label']:<{width}}  {metric:<9}{r['production']:>11.4f}"
              f"{r['candidate']:>11.4f}{r['change_pct']:>+8.2f}%{trees:>11}"
              f"   {verdict:<18}{r['shipped_as_is']:>14.4f}")

    print("\n(negative change = the candidate is better)")
    print("'shipped as-is' scores the deployed weights directly on this window,")
    print("and is never compared - it is shown so the bias stays visible. For")
    print("models built by finalize_models.py it is optimistic, because those")
    print("games ARE in their training set. For a model that genuinely predates")
    print("the window it is honest, and the gap to PRODUCTION is simply the")
    print("value of the extra training data.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-refresh", action="store_true",
                        help="do not rerun ingestion/preprocessing")
    parser.add_argument("--production-dir", type=Path, default=PRODUCTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--test-games", type=int, default=DEFAULT_TEST_GAMES)
    parser.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT)
    parser.add_argument("--min-new-games", type=int, default=DEFAULT_MIN_NEW_GAMES)
    parser.add_argument("--force", action="store_true",
                        help="retrain regardless of how little new data exists")
    args = parser.parse_args()

    section("NEW DATA CHECK")
    before = snapshot(GAMES_FINAL)
    print(f"  before refresh: {before['rows']:,} team-game rows, "
          f"latest {before['latest']}")

    if args.skip_refresh:
        print("  --skip-refresh: pipeline not rerun.")
    else:
        refresh_data()

    after = snapshot(GAMES_FINAL)
    print(f"\n  after refresh:  {after['rows']:,} team-game rows, "
          f"latest {after['latest']}")

    section("DATA PREP")
    dataset = load_dataset()
    print(f"Loaded {len(dataset):,} games.")

    # How much of this is genuinely new to the deployed models?
    marker = read_trained_through(args.production_dir)
    if marker is None:
        print(f"\n  No {TRAINED_THROUGH} beside the production models, so how "
              f"much data\n  they have already seen is unknown. Not guessing.")
        if not args.force:
            print("  Skipping. Pass --force to retrain anyway; the marker is "
                  "written\n  alongside any candidates produced, so this "
                  "resolves itself after one run.")
            return 0
        print("  --force given: continuing.")
    else:
        cutoff = pd.Timestamp(marker["trained_through"])
        new_games = int((dataset["GAME_DATE"] > cutoff).sum())
        print(f"\n  production trained through {marker['trained_through']} "
              f"({marker['games']:,} games)")
        print(f"  genuinely new games since then: {new_games:,}")

        if new_games < args.min_new_games and not args.force:
            # An honest answer, not a silent no-op. Expected for most of the
            # year - the NBA offseason runs roughly May to October, and no
            # amount of retraining invents games nobody played.
            print(f"\n  Fewer than {args.min_new_games} new games. Nothing worth "
                  f"retraining on.")
            print("  Pass --force to retrain anyway.")
            return 0
        if new_games < args.min_new_games:
            print(f"\n  Below the {args.min_new_games}-game threshold, but "
                  f"--force was given.")

    print()
    train, validation, test = rolling_split(dataset, args.test_games)

    section("TRAINING CANDIDATES AND RE-FITTING PRODUCTION")
    print(f"Production architecture read from {args.production_dir}")
    print("Both models fit on the same training rows; neither sees the test window.")
    results = run_gate(train, validation, test, args.production_dir,
                       args.tolerance_pct)

    print_table(results, args.tolerance_pct)

    passed = [r for r in results if r["passed"]]
    section("RESULT")
    print(f"{len(passed)} of {len(results)} targets passed the gate.")

    if len(passed) != len(results):
        failed = ", ".join(r["label"] for r in results if not r["passed"])
        print(f"\nNOT PROMOTING ANY MODEL. Regressed: {failed}")
        print("All seven ship together - they share a feature set and a")
        print("dataset, so promoting a subset would leave the served models")
        print("trained on different data as each other.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("*.json"):
        stale.unlink()
    for r in results:
        r["model"].save_model(str(args.output_dir / f"{r['key']}.json"))
    write_trained_through(args.output_dir, dataset)

    print(f"\nAll passed. {len(results)} candidate models written to "
          f"{args.output_dir}")
    print(f"A {TRAINED_THROUGH} marker was written alongside them, so the next")
    print("run can tell how much data these have already seen.")
    print("ml-training/models/ is UNTOUCHED. Promoting these is a separate,")
    print("human decision - a pull request once the scheduled job exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
