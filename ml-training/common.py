"""
Shared code for the training scripts and the inference service.

Lives here so scripts depend on this module instead of importing each other.
"""

from pathlib import Path

# mlflow is imported inside setup_mlflow(), not here: the inference service
# imports this module for FEATURE_COLUMNS and would otherwise pull in
# mlflow's whole dependency tree.

# Train is everything before VALIDATION_SEASON, so adding older seasons
# grows training rather than reclassifying existing ones.
VALIDATION_SEASON = 2023
TEST_SEASONS = (2024, 2025)

# MLflow 3.x rejects the plain-directory backend, so tracking uses SQLite.
# Both the db and artifacts sit under ml-training/mlruns/ (gitignored).
TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"
TRACKING_URI = f"sqlite:///{(TRACKING_DIR / 'mlflow.db').as_posix()}"
ARTIFACT_URI = (TRACKING_DIR / "artifacts").as_uri()


# The 19 pre-game features per team. model_dataset.csv holds each twice,
# prefixed HOME_ and AWAY_, giving 38 model inputs.
# Order matters: the saved models store these names and check them at
# predict time, so inference must build the row in this order.
#
# ABSENT_COUNT and WEIGHTED_ABSENT_MIN are appended last, so the original
# 34 keep their positions. They were held out until a live roster source
# existed, because live_features.py could not compute them for an unplayed
# game and a 0 default would have meant "assume a full roster" - a 0
# standing in for "unknown". That source now exists: the NBA's official
# injury report, fetched and reconciled by injury_availability.py (CLAUDE.md
# section 15). When the report is unavailable they resolve to NaN, never 0,
# which XGBoost handles natively.
#
# Any model in models/ trained before this change expects 34 columns and
# will reject a 38-column row. finalize_models.py must be rerun alongside
# any edit to this list.
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
    "ABSENT_COUNT",
    "WEIGHTED_ABSENT_MIN",
]

FEATURE_COLUMNS = [f"{side}_{feat}" for side in ("HOME", "AWAY") for feat in PER_SIDE_FEATURES]

ROLLING_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if "ROLL5_" in c or "ROLL10_" in c]


def setup_mlflow(experiment_name: str):
    """Point MLflow at the SQLite store, creating the experiment if needed.

    Name is required, not defaulted, so runs can't land in the wrong
    experiment. Artifact location can only be set at creation time, hence
    create_experiment rather than set_experiment.
    """
    import mlflow  # deferred, see note at top of module

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(experiment_name)


def split_three_way(df):
    """Chronological train / validation / test split by season.

    Validation is what early stopping measures against. Using test for that
    would turn it into a training signal and inflate the reported score.
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
