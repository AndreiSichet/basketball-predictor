"""
Prediction API for the seven basketball models.

Run with:  uvicorn app:app --port 8000    (from inference-service/)

The games table and all seven models load once at startup. A request then
only builds the feature row (width follows common.FEATURE_COLUMNS) and
runs seven predictions.

Every response includes data_as_of and stale. The service can only see
games already in games_final.csv, so if the pipeline hasn't run, results
are silently computed from old history - no error, just worse numbers.
That's invisible to callers unless the response says so.

Feature building lives in ml-training/live_features.py, not here, so there
is one implementation with a verification harness rather than a copy.

/schedule is the odd one out: it reaches the live NBA API rather than the
local dataset, so it is the only endpoint that can fail for reasons that
have nothing to do with this service. It returns fixtures, never
predictions - most of what it returns is not predictable (see
MAX_DAYS_AHEAD), and deciding that is the caller's job.
"""

from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from nba_api.stats.endpoints import scheduleleaguev2
from pydantic import BaseModel, Field
from xgboost import XGBClassifier, XGBRegressor

# ml-training is a sibling folder, not an installed package, so it goes on
# sys.path. Same tradeoff live_features.py makes to reach the pipeline.
ML_TRAINING_DIR = Path(__file__).resolve().parents[1] / "ml-training"
if str(ML_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(ML_TRAINING_DIR))

from common import FEATURE_COLUMNS  # noqa: E402
from live_features import (  # noqa: E402
    get_live_features,
    load_games_final,
    season_of,
)

MODELS_DIR = ML_TRAINING_DIR / "models"

# Responses are flagged stale past this. Two days, not one, because the
# pipeline running slightly late is normal.
STALE_AFTER_DAYS = 2

# How far ahead a prediction may be requested. REST_DAYS is measured from
# the last game in the data, so for a fixture further out - with unplayed
# games in between - the gap would be measured against the wrong game.
MAX_DAYS_AHEAD = 1

# Schedule lookahead when the caller doesn't say. Unrelated to
# MAX_DAYS_AHEAD: this is how far to *list*, not how far to predict.
SCHEDULE_DAYS_AHEAD_DEFAULT = 14
SCHEDULE_TIMEOUT_SECONDS = 45

# 3rd digit of the zero-padded 10-digit game id: 1=preseason, 2=regular,
# 3=All-Star, 4=playoffs, 5=play-in, 6=NBA Cup final. The models only ever
# saw regular-season games (the training pull is filtered the same way), so
# listing anything else would offer fixtures they have no business scoring.
REGULAR_SEASON_GAME_ID_DIGIT = "2"

# gameStatus 1=scheduled, 2=live, 3=final. Only 1 is unplayed.
GAME_STATUS_SCHEDULED = 1

# The schedule changes rarely and the upstream call costs seconds, so a
# button press does not need to hit nba_api every time. fetch_games.py is
# deliberately gentle with this API for the same reason.
SCHEDULE_CACHE_TTL_SECONDS = 6 * 60 * 60

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


class ScheduledGame(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date


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

# (season, fetched_at, dataframe) for the last schedule pulled.
_schedule_cache: dict = {}


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


def season_string(today: date) -> str:
    """Season as ScheduleLeagueV2 wants it: "2026-27".

    season_of wraps the pipeline's derive_season, so the August boundary
    that decides which season a date belongs to stays defined in exactly
    one place. Only the formatting is new.
    """
    start_year = season_of(pd.Timestamp(today))
    return f"{start_year}-{str(start_year + 1)[2:]}"


def fetch_schedule_frame(season: str) -> pd.DataFrame:
    """The season's full schedule, cached briefly."""
    cached = _schedule_cache.get(season)
    if cached is not None:
        age = (datetime.now() - cached["fetched_at"]).total_seconds()
        if age < SCHEDULE_CACHE_TTL_SECONDS:
            return cached["frame"]

    try:
        frame = scheduleleaguev2.ScheduleLeagueV2(
            season=season, league_id="00", timeout=SCHEDULE_TIMEOUT_SECONDS
        ).get_data_frames()[0]
    except Exception as error:
        # Anything from a DNS failure to nba_api handing back HTML instead
        # of JSON. The request was fine; the upstream was not.
        raise HTTPException(
            502, f"Could not reach the NBA schedule API for season {season}: {error}"
        ) from error

    _schedule_cache[season] = {"fetched_at": datetime.now(), "frame": frame}
    return frame


@app.get("/schedule", response_model=list[ScheduledGame])
def schedule(
    days_ahead: int = Query(SCHEDULE_DAYS_AHEAD_DEFAULT, ge=1, le=365),
):
    """Upcoming regular-season fixtures, as candidates to display.

    Says nothing about whether any of them can actually be predicted -
    /predict enforces MAX_DAYS_AHEAD and will reject most of these. An
    offseason gap or an unreleased schedule is an empty list, not an error:
    "nothing scheduled" is an answer, not a failure.
    """
    today = pd.Timestamp(datetime.now().date())
    frame = fetch_schedule_frame(season_string(today.date()))

    if frame.empty:
        return []

    # Derive before filtering, never after. .assign() of a Series onto a
    # zero-row frame reindexes the frame back up to the Series' index
    # (pandas 2.3), so filtering first would turn "nothing scheduled" into
    # a full frame of NaN - and the offseason is exactly when that happens.
    frame = frame.assign(
        parsed_date=pd.to_datetime(frame["gameDate"], format="%m/%d/%Y %H:%M:%S"),
        padded_game_id=frame["gameId"].astype(str).str.zfill(10),
    )
    horizon = today + pd.DateOffset(days=days_ahead)

    upcoming = frame[
        (frame["parsed_date"] >= today)
        & (frame["parsed_date"] <= horizon)
        & (frame["gameStatus"] == GAME_STATUS_SCHEDULED)
        & (frame["padded_game_id"].str[2] == REGULAR_SEASON_GAME_ID_DIGIT)
    ].sort_values(["parsed_date", "padded_game_id"])

    return [
        ScheduledGame(
            home_team_id=int(row.homeTeam_teamId),
            away_team_id=int(row.awayTeam_teamId),
            game_date=row.parsed_date.date(),
        )
        for row in upcoming.itertuples()
    ]


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
