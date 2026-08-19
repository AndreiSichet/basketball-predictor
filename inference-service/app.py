"""
Prediction API for the seven basketball models.

Run with:  uvicorn app:app --port 8000    (from inference-service/)

The games table and all seven models load once at startup. A request then
only builds a 34-column feature row and runs seven predictions.

Every response includes data_as_of and stale. The service can only see
games already in games_final.csv, so if the pipeline hasn't run, results
are silently computed from old history - no error, just worse numbers.
That's invisible to callers unless the response says so.

Feature building lives in ml-training/live_features.py, not here, so there
is one implementation with a verification harness rather than a copy.
"""

from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier, XGBRegressor

# ml-training is a sibling folder, not an installed package, so it goes on
# sys.path. Same tradeoff live_features.py makes to reach the pipeline.
ML_TRAINING_DIR = Path(__file__).resolve().parents[1] / "ml-training"
if str(ML_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(ML_TRAINING_DIR))

from common import FEATURE_COLUMNS  # noqa: E402
from live_features import get_live_features, load_games_final  # noqa: E402

MODELS_DIR = ML_TRAINING_DIR / "models"

# Responses are flagged stale past this. Two days, not one, because the
# pipeline running slightly late is normal.
STALE_AFTER_DAYS = 2

# How far ahead a prediction may be requested. REST_DAYS is measured from
# the last game in the data, so for a fixture further out - with unplayed
# games in between - the gap would be measured against the wrong game.
MAX_DAYS_AHEAD = 1

# What this service serves. (model file stem, response field, is_classification)
MODEL_REGISTRY = [
    ("moneyline", "home_win_probability", True),
    ("spread", "home_margin", False),
    ("totals", "total_points", False),
    ("reb_margin", "rebound_margin", False),
    ("reb_total", "total_rebounds", False),
    ("ast_margin", "assist_margin", False),
    ("ast_total", "total_assists", False),
]


class PredictionRequest(BaseModel):
    home_team_id: int = Field(..., description="NBA team id of the home side")
    away_team_id: int = Field(..., description="NBA team id of the away side")
    game_date: date = Field(..., description="Tip-off date, YYYY-MM-DD")


class Predictions(BaseModel):
    home_win_probability: float
    home_margin: float
    total_points: float
    rebound_margin: float
    total_rebounds: float
    assist_margin: float
    total_assists: float


class PredictionResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    predictions: Predictions


class ServiceState:
    """Everything loaded once at startup and shared across requests."""

    games_final_df: pd.DataFrame
    models: dict
    known_team_ids: set
    data_as_of: pd.Timestamp


state = ServiceState()


def load_models() -> dict:
    """Load all seven models as sklearn-API estimators.

    XGBClassifier/XGBRegressor rather than raw Booster, to keep
    predict()/predict_proba() and their feature-name validation. The
    booster API would accept a mis-ordered feature vector without complaint.
    """
    on_disk = {path.stem for path in MODELS_DIR.glob("*.json")}
    expected = {key for key, _field, _clf in MODEL_REGISTRY}

    if on_disk != expected:
        # Fail at startup rather than let a model go unserved because
        # someone added a file and not a registry entry.
        raise RuntimeError(
            f"models/ does not match the registry.\n"
            f"  missing from disk: {sorted(expected - on_disk) or 'none'}\n"
            f"  present but unregistered: {sorted(on_disk - expected) or 'none'}"
        )

    models = {}
    for key, _field, classification in MODEL_REGISTRY:
        model = XGBClassifier() if classification else XGBRegressor()
        model.load_model(str(MODELS_DIR / f"{key}.json"))
        models[key] = model
    return models


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.games_final_df = load_games_final()
    state.data_as_of = pd.Timestamp(state.games_final_df["GAME_DATE"].max())
    state.known_team_ids = set(state.games_final_df["TEAM_ID"].unique())
    state.models = load_models()

    print(
        f"Loaded {len(state.games_final_df)} team-game rows, "
        f"{len(state.known_team_ids)} teams, {len(state.models)} models. "
        f"Data as of {state.data_as_of.date()}."
    )
    yield


app = FastAPI(
    title="Basketball prediction service",
    description="Seven XGBoost models over engineered pre-game features.",
    version="1.0.0",
    lifespan=lifespan,
)


def freshness() -> tuple:
    """How far behind the underlying data is, as of right now."""
    days_behind = (pd.Timestamp(datetime.now().date()) - state.data_as_of).days
    return days_behind, days_behind > STALE_AFTER_DAYS


@app.get("/health")
def health():
    """Liveness plus freshness, so monitoring can alert on stale data."""
    days_behind, stale = freshness()
    return {
        "status": "ok",
        "models_loaded": len(state.models),
        "data_as_of": state.data_as_of.date().isoformat(),
        "days_behind": days_behind,
        "stale": stale,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    game_date = pd.Timestamp(request.game_date)

    if request.home_team_id == request.away_team_id:
        raise HTTPException(400, "home_team_id and away_team_id must differ.")

    unknown = [
        team_id
        for team_id in (request.home_team_id, request.away_team_id)
        if team_id not in state.known_team_ids
    ]
    if unknown:
        raise HTTPException(
            400,
            f"No history for team id(s) {unknown}. Features cannot be built for a "
            f"team with no completed games in the data.",
        )

    latest_allowed = state.data_as_of + pd.DateOffset(days=MAX_DAYS_AHEAD)
    if game_date > latest_allowed:
        raise HTTPException(
            400,
            f"game_date {request.game_date} is more than {MAX_DAYS_AHEAD} day past "
            f"the newest game in the data ({state.data_as_of.date()}). Rest-day "
            f"features would be computed against the wrong prior game. Latest "
            f"accepted date is {latest_allowed.date()}.",
        )

    try:
        features = get_live_features(
            request.home_team_id,
            request.away_team_id,
            game_date,
            state.games_final_df,
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(400, f"Could not build features: {error}") from error

    results = {}
    for key, field, classification in MODEL_REGISTRY:
        model = state.models[key]
        value = (
            model.predict_proba(features)[0, 1]
            if classification
            else model.predict(features)[0]
        )
        results[field] = float(value)

    days_behind, stale = freshness()
    return PredictionResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        predictions=Predictions(**results),
    )
