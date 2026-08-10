"""
Prediction API for the seven basketball models.

Run with:  uvicorn app:app --port 8000    (from inference-service/)

Everything expensive happens once at startup: the historical games table is
parsed and all seven models are loaded into memory. A request then does
nothing but build a 34-column feature row and run seven predictions, which
is the only reason this can answer in milliseconds rather than seconds.

FRESHNESS IS PART OF THE CONTRACT, NOT AN AFTERTHOUGHT. This service can
only see games that are already in games_final.csv. If the pipeline has not
run since last night, predictions are computed from stale history and are
silently worse — no error, no exception, just quietly degraded numbers.
That failure is invisible from the outside, so every response carries
data_as_of and stale, whether or not the caller asked. A client that
ignores them is making an informed choice; one that never sees them is not.

The feature-building logic deliberately lives in ml-training/live_features.py
rather than here. Inference features must match training features exactly,
and the surest way to guarantee that is for there to be one implementation
with a verification harness attached, not a copy in the serving layer.
"""

from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier, XGBRegressor

# ml-training is a sibling folder, not an installed package. Same tradeoff
# the feature module already makes to reach the pipeline scripts: an
# explicit path insert, in exchange for never duplicating shared logic.
ML_TRAINING_DIR = Path(__file__).resolve().parents[1] / "ml-training"
if str(ML_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(ML_TRAINING_DIR))

from common import FEATURE_COLUMNS  # noqa: E402
from live_features import get_live_features, load_games_final  # noqa: E402

MODELS_DIR = ML_TRAINING_DIR / "models"

# Past this many days behind, responses are flagged stale. Two days rather
# than one because the pipeline running a little late is normal operations,
# not a fault worth crying wolf over.
STALE_AFTER_DAYS = 2

# How far past the newest game a prediction may be requested. One day is the
# real limit rather than an arbitrary one: REST_DAYS is measured from the
# last game present in the data, so for a fixture further out — with games
# in between that have not been played yet — that gap would be computed
# against the wrong prior game. The models were never trained on that.
MAX_DAYS_AHEAD = 1

# What this service serves, declared explicitly rather than discovered.
# (model file stem, response field, is_classification)
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

    Loading into XGBClassifier/XGBRegressor rather than a raw Booster keeps
    predict()/predict_proba() and the feature-name validation that comes
    with them — the booster API would happily accept a mis-ordered feature
    vector and return a confident wrong answer.
    """
    on_disk = {path.stem for path in MODELS_DIR.glob("*.json")}
    expected = {key for key, _field, _clf in MODEL_REGISTRY}

    if on_disk != expected:
        # Loud at startup beats a target silently going unserved because
        # someone added a model and not a registry entry.
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
    """Liveness plus the freshness signal, so monitoring can alert on staleness."""
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
