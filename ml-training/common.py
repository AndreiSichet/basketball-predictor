"""
Shared infrastructure for the model training scripts.

Anything two or more training scripts need lives here, so that they depend
on this module rather than on each other. Before this existed, the
regression script imported its split from the moneyline script — which
worked, but implied a relationship between the two models that doesn't
exist, and would have gotten worse with every script added.

The chronological split especially belongs here rather than in any one
script. A future quarter-score or player-prop model that split even
slightly differently would produce numbers that look comparable to the
existing ones without being comparable at all, and nothing in the output
would reveal it.
"""

from pathlib import Path

# NOTE: mlflow is imported inside setup_mlflow() rather than here on purpose.
# The inference service imports this module for FEATURE_COLUMNS and never
# tracks an experiment, and mlflow drags in Flask, SQLAlchemy and a large
# dependency tree that has no business in a serving container.

# Split boundaries. Train is "everything before validation" rather than a
# literal range, so extending the dataset backwards grows training instead
# of silently reclassifying seasons.
VALIDATION_SEASON = 2023
TEST_SEASONS = (2024, 2025)

# MLflow 3.x refuses the plain-directory ("file store") backend, which is in
# maintenance mode, so tracking goes to SQLite. Both the database and the
# artifacts live under ml-training/mlruns/, which .gitignore already covers.
TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"
TRACKING_URI = f"sqlite:///{(TRACKING_DIR / 'mlflow.db').as_posix()}"
ARTIFACT_URI = (TRACKING_DIR / "artifacts").as_uri()


# The 17 pre-game features available for each side. model_dataset.csv carries
# each one twice, prefixed HOME_ and AWAY_, for 34 model inputs in total.
#
# ORDER IS PART OF THE CONTRACT. The saved models store these names and
# validate against them at predict time, so anything assembling a feature
# vector for inference must build it in exactly this order.
PER_SIDE_FEATURES = [
    "ROLL5_WIN_PCT",
    "ROLL5_PTS",
    "ROLL5_PLUS_MINUS",
    "ROLL5_FG_PCT",
    "ROLL5_REB",
    "ROLL5_AST",
    "ROLL5_TOV",
    "ROLL10_WIN_PCT",
    "ROLL10_PTS",
    "ROLL10_PLUS_MINUS",
    "ROLL10_FG_PCT",
    "ROLL10_REB",
    "ROLL10_AST",
    "ROLL10_TOV",
    "REST_DAYS",
    "IS_BACK_TO_BACK",
    "TEAM_ELO",
]

FEATURE_COLUMNS = [f"{side}_{feat}" for side in ("HOME", "AWAY") for feat in PER_SIDE_FEATURES]

ROLLING_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if "ROLL5_" in c or "ROLL10_" in c]


def setup_mlflow(experiment_name: str):
    """Point MLflow at the SQLite store, creating the experiment if needed.

    The experiment name is required rather than defaulted: a shared helper
    defaulting to one particular model's experiment is how runs end up
    logged somewhere nobody intended.

    Artifact location can only be set at creation time, so this has to go
    through create_experiment rather than set_experiment. Runs nest under
    the shared root by their own unique id, so experiments can share one
    artifact root without colliding.
    """
    import mlflow  # deferred: see the note at the top of this module

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(experiment_name)


def split_three_way(df):
    """Chronological train / validation / test split by season.

    Validation exists so a model can decide when to stop training without
    ever touching the test set — doing that on test would quietly turn it
    into a training signal, and the reported number would become optimistic
    in a way nothing downstream could detect.
    """
    train = df[df["SEASON"] < VALIDATION_SEASON]
    validation = df[df["SEASON"] == VALIDATION_SEASON]
    test = df[df["SEASON"].isin(TEST_SEASONS)]

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            f"Empty split - seasons present: {sorted(df['SEASON'].unique())}, "
            f"expected data before {VALIDATION_SEASON} and in {TEST_SEASONS}."
        )

    train_seasons = sorted(train["SEASON"].unique())
    print(f"Train:      seasons {train_seasons[0]}-{train_seasons[-1]} ({len(train)} games)")
    print(f"Validation: season  {VALIDATION_SEASON} ({len(validation)} games)")
    print(f"Test:       seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1]} ({len(test)} games)")

    return train, validation, test
