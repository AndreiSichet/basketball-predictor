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
import json
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from nba_api.stats.endpoints import scheduleleaguev2
from pydantic import BaseModel, ConfigDict, Field
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
from live_quarter_half_features import (  # noqa: E402
    InsufficientQuarterHalfHistory,
    get_live_quarter_half_features,
    load_quarter_half_history,
)
from live_player_features import (  # noqa: E402
    # The 17 player-prop inputs, distinct from common.FEATURE_COLUMNS's 38.
    FEATURE_COLUMNS as FEATURE_COLUMNS_PLAYER,
    describe as describe_roster,
    get_live_player_features,
    load_player_history,
)

MODELS_DIR = ML_TRAINING_DIR / "models"
QH_MODELS_DIR = ML_TRAINING_DIR / "models_quarter_half"
PP_MODELS_DIR = ML_TRAINING_DIR / "models_player_props"

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

# The six Q1/first-half models. A SEPARATE registry with its own strict
# check, not an extension of the one above: these are joblib Pipelines
# rather than XGBoost .json files, and folding them together would mean
# relaxing the check that has already caught a real problem once.
QH_MODEL_REGISTRY = [
    ("q1_spread", "q1_home_margin", False),
    ("q1_total", "q1_total_points", False),
    ("1h_spread", "half1_home_margin", False),
    ("1h_total", "half1_total_points", False),
    ("q1_winner", "q1_home_win_probability", True),
    ("1h_winner", "half1_home_win_probability", True),
]

# REQUIRED on the two winner markets, never optional. These models were
# trained only on periods that had a winner, because a tied quarter has no
# binary label. The number is therefore P(home leads | not tied) - which a
# sportsbook would push - and is NOT comparable to the full-game moneyline.
# Shipping the probability without this string attached would invite
# exactly that comparison.
CONDITIONAL_INTERPRETATION = "P(home leads | not tied)"

# Five targets, two artifacts each. Counted as ten because that is what is
# on disk and what the registry check compares against.
PP_TARGETS = ["PTS", "REB", "AST", "FG3M", "PRA"]
PP_ROUTES = ["linear", "xgb"]

# How many players per team a prop board actually wants. The feature module
# returns everyone with usable history - over a hundred for a long-running
# franchise - because deciding who is worth showing is a presentation
# question, not a feature-engineering one. The frame arrives sorted by
# ROLL10_MIN descending, so this is a head().
PLAYER_PROPS_PER_TEAM = 10


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


class QuarterHalfPrediction(BaseModel):
    """One Q1/1H market.

    interpretation is present only on the two winner markets, where the
    number means something narrower than it looks. Absent elsewhere rather
    than an empty string, so a client can test for it.
    """

    market: str
    value: float
    confidence: str
    interpretation: str | None = None


class QuarterHalfResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    predictions: list[QuarterHalfPrediction]


class PlayerPrediction(BaseModel):
    # "model_used" collides with pydantic's protected model_ namespace.
    # The API contract wins: the field name says exactly what it holds, and
    # renaming it to satisfy a library default would be the wrong trade.
    model_config = ConfigDict(protected_namespaces=())

    player_id: int
    player_name: str
    model_used: str
    predictions: dict[str, float]


class TeamPlayerProps(BaseModel):
    team_id: int
    is_home: bool
    availability_known: bool
    # Populated only when availability_known is False, carrying the exact
    # wording live_player_features.describe() produces rather than a second
    # phrasing of the same caveat.
    availability_note: str | None = None
    players: list[PlayerPrediction]


class PlayerPropsResponse(BaseModel):
    home_team_id: int
    away_team_id: int
    game_date: date
    data_as_of: date
    stale: bool
    days_behind: int
    teams: list[TeamPlayerProps]


class ServiceState:
    """Everything loaded once at startup and shared across requests."""

    games_final_df: pd.DataFrame
    models: dict
    known_team_ids: set
    data_as_of: pd.Timestamp

    # Quarter/half: six Pipelines plus the history frame they score from.
    qh_models: dict
    qh_confidence: dict
    qh_history_df: pd.DataFrame

    # Player props: ten artifacts, the routing rule, and the 59 MB history.
    pp_models: dict
    pp_routing_columns: list
    player_history_df: pd.DataFrame


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


def load_quarter_half_models() -> tuple:
    """The six Q1/1H Pipelines, plus each one's declared confidence.

    Its own strict registry check, deliberately mirroring load_models()
    rather than sharing it. Same failure mode being guarded against - a
    file added without a registry entry, or the reverse - but a different
    directory and a different artifact type, and a combined check would
    have to be loose enough to accept both.

    Confidence is read from the shipped manifest, never hardcoded here: it
    is a property of how the model scored, and the manifest is where that
    was recorded.
    """
    on_disk = {path.stem for path in QH_MODELS_DIR.glob("*.joblib")}
    expected = {key for key, _field, _clf in QH_MODEL_REGISTRY}

    if on_disk != expected:
        raise RuntimeError(
            f"models_quarter_half/ does not match QH_MODEL_REGISTRY.\n"
            f"  missing from disk: {sorted(expected - on_disk) or 'none'}\n"
            f"  present but unregistered: {sorted(on_disk - expected) or 'none'}"
        )

    manifest_path = QH_MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} is missing. Confidence labels and the feature "
            f"order live there; guessing either would be worse than failing."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    confidence = {entry["key"]: entry["confidence"] for entry in manifest["models"]}

    missing_confidence = expected - set(confidence)
    if missing_confidence:
        raise RuntimeError(
            f"manifest.json has no confidence entry for "
            f"{sorted(missing_confidence)}. Serving a prediction without its "
            f"confidence label is exactly what the field exists to prevent."
        )

    models = {key: joblib.load(QH_MODELS_DIR / f"{key}.joblib")
              for key, _field, _clf in QH_MODEL_REGISTRY}
    return models, confidence


def load_player_prop_models() -> tuple:
    """The ten player-prop artifacts and the routing rule that picks between.

    Two artifacts per target, so the registry check counts ten files: five
    <target>_linear.joblib and five <target>_xgb.json. The routing columns
    come from the manifest, not from this file - the whole point of putting
    them there was that the serving layer reads the rule rather than
    carrying a second copy that could drift.
    """
    expected = {f"{t.lower()}_{route}" for t in PP_TARGETS for route in PP_ROUTES}
    on_disk = ({path.stem for path in PP_MODELS_DIR.glob("*.joblib")}
               | {path.stem for path in PP_MODELS_DIR.glob("*.json")
                  if path.stem != "manifest"})

    if on_disk != expected:
        raise RuntimeError(
            f"models_player_props/ does not match the expected artifact set.\n"
            f"  missing from disk: {sorted(expected - on_disk) or 'none'}\n"
            f"  present but unregistered: {sorted(on_disk - expected) or 'none'}"
        )

    manifest_path = PP_MODELS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"{manifest_path} is missing. It carries the routing rule, and a "
            f"hybrid with no router is not servable."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routing_columns = manifest["routing"]["decide_on"]

    models = {}
    for target in PP_TARGETS:
        stem = target.lower()
        models[(target, "linear")] = joblib.load(
            PP_MODELS_DIR / f"{stem}_linear.joblib")
        booster = XGBRegressor()
        booster.load_model(str(PP_MODELS_DIR / f"{stem}_xgb.json"))
        models[(target, "xgb")] = booster
    return models, routing_columns


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.games_final_df = load_games_final()
    state.data_as_of = pd.Timestamp(state.games_final_df["GAME_DATE"].max())
    state.known_team_ids = set(state.games_final_df["TEAM_ID"].unique())
    state.models = load_models()

    # Derived once here rather than per request: the pairing and reindex
    # cost seconds, and every quarter/half prediction needs the same frame.
    state.qh_history_df = load_quarter_half_history()
    state.qh_models, state.qh_confidence = load_quarter_half_models()

    # THE BIG ONE. player_boxscores_with_rolling.csv is ~59 MB on disk and
    # materially more in memory - by far the largest thing this service
    # holds, and a real step up in the container's footprint. It is loaded
    # narrowed to the columns the feature module reads, but it is still the
    # reason this image is no longer small.
    state.player_history_df = load_player_history()
    state.pp_models, state.pp_routing_columns = load_player_prop_models()

    print(
        f"Loaded {len(state.games_final_df)} team-game rows, "
        f"{len(state.known_team_ids)} teams, {len(state.models)} team models. "
        f"Data as of {state.data_as_of.date()}."
    )
    print(
        f"Quarter/half: {len(state.qh_models)} models, "
        f"{len(state.qh_history_df):,} team-game rows."
    )
    print(
        f"Player props: {len(state.pp_models)} artifacts, "
        f"{len(state.player_history_df):,} player-game rows, "
        f"routing on {len(state.pp_routing_columns)} columns."
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
    """Liveness plus freshness, so monitoring can alert on stale data.

    models_loaded is BROKEN OUT BY FAMILY, not summed. A single 23 would
    still be 23 if one family loaded twice and another not at all; three
    named counts say which one is wrong. Same reason /predict reports
    data_as_of rather than just a boolean.

    NOTE: this changed shape from an int. .github/workflows/ci.yml asserts
    on it, and was updated in the same commit.
    """
    days_behind, stale = freshness()
    return {
        "status": "ok",
        "models_loaded": {
            "team": len(state.models),
            "quarter_half": len(state.qh_models),
            "player_props": len(state.pp_models),
        },
        "data_as_of": state.data_as_of.date().isoformat(),
        "days_behind": days_behind,
        "stale": stale,
    }


def validate_matchup(request: "PredictionRequest") -> pd.Timestamp:
    """Checks every prediction endpoint must make. Returns the parsed date.

    Extracted when the third endpoint arrived rather than copied a second
    time - three divergent copies of the MAX_DAYS_AHEAD rule is exactly how
    one endpoint quietly starts accepting dates the others reject.
    """
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

    return game_date


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
    game_date = validate_matchup(request)

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


@app.post("/predict/quarter-half", response_model=QuarterHalfResponse)
def predict_quarter_half(request: PredictionRequest):
    """Six Q1 / first-half markets for one fixture.

    Fails differently from /predict, and the difference is the model class.
    These are linear pipelines: they cannot consume a NaN at all, so a team
    without a complete trailing window is unscoreable rather than scoreable
    with degraded input. That surfaces as a 400 with the reason, not a 500
    from inside sklearn's input validation.
    """
    game_date = validate_matchup(request)

    try:
        features = get_live_quarter_half_features(
            request.home_team_id,
            request.away_team_id,
            game_date,
            state.qh_history_df,
            state.games_final_df,
        )
    except InsufficientQuarterHalfHistory as error:
        raise HTTPException(400, str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(400, f"Could not build features: {error}") from error

    predictions = []
    for key, field, classification in QH_MODEL_REGISTRY:
        model = state.qh_models[key]
        value = (
            model.predict_proba(features)[0, 1]
            if classification
            else model.predict(features)[0]
        )
        predictions.append(
            QuarterHalfPrediction(
                market=field,
                value=float(value),
                confidence=state.qh_confidence[key],
                # Required on the winners, omitted on the rest.
                interpretation=CONDITIONAL_INTERPRETATION if classification else None,
            )
        )

    days_behind, stale = freshness()
    return QuarterHalfResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        predictions=predictions,
    )


def player_props_for_team(team_id: int, opponent_id: int, game_date, is_home: bool):
    """Top-N players for one side, each scored by the model its row routes to.

    ROSTER AVAILABILITY IS DEFERRED, exactly as team-level availability is.
    injury_report=None is passed deliberately: the container has neither the
    injury-fetching code nor a Java runtime for it, the same packaging gap
    already documented for the 38-feature models. So every row comes back
    availability_known=False and nobody is excluded - which is reported, not
    silently presented as a clean roster. Both capabilities unblock together
    when that packaging decision is made.
    """
    roster = get_live_player_features(
        team_id,
        opponent_id,
        game_date,
        is_home,
        state.player_history_df,
        state.games_final_df,
        injury_report=None,
        decide_on=state.pp_routing_columns,
    )

    availability_known = (
        bool(roster["AVAILABILITY_KNOWN"].iloc[0]) if not roster.empty else False
    )
    note = None if availability_known else describe_roster(roster)

    # Arrives sorted by ROLL10_MIN descending, so this is the busiest N.
    roster = roster.head(PLAYER_PROPS_PER_TEAM)

    players = []
    if not roster.empty:
        # Batched by route rather than one call per player: five targets by
        # two routes is ten predict() calls per team, against fifty.
        scores = {}
        for route in PP_ROUTES:
            block = roster[roster["ROUTE"] == route]
            if block.empty:
                continue
            matrix = block[FEATURE_COLUMNS_PLAYER]
            for target in PP_TARGETS:
                values = state.pp_models[(target, route)].predict(matrix)
                for player_id, value in zip(block["PLAYER_ID"], values):
                    scores.setdefault(int(player_id), {})[target] = float(value)

        for row in roster.itertuples():
            players.append(
                PlayerPrediction(
                    player_id=int(row.PLAYER_ID),
                    player_name=str(row.PLAYER_NAME),
                    model_used=row.ROUTE,
                    predictions=scores[int(row.PLAYER_ID)],
                )
            )

    return TeamPlayerProps(
        team_id=team_id,
        is_home=is_home,
        availability_known=availability_known,
        availability_note=note,
        players=players,
    )


@app.post("/predict/player-props", response_model=PlayerPropsResponse)
def predict_player_props(request: PredictionRequest):
    """Both teams' prop boards for one fixture, in one call.

    Both sides together because that is the real unit of use - a prop board
    is a game's worth of players, not a team's - and because the two share
    the validation, the freshness block and the loaded history frames.
    """
    game_date = validate_matchup(request)

    try:
        teams = [
            player_props_for_team(
                request.home_team_id, request.away_team_id, game_date, True
            ),
            player_props_for_team(
                request.away_team_id, request.home_team_id, game_date, False
            ),
        ]
    except (ValueError, KeyError) as error:
        raise HTTPException(400, f"Could not build features: {error}") from error

    days_behind, stale = freshness()
    return PlayerPropsResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        game_date=request.game_date,
        data_as_of=state.data_as_of.date(),
        stale=stale,
        days_behind=days_behind,
        teams=teams,
    )
