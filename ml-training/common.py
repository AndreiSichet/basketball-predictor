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

import mlflow

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
