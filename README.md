# Modellarium

A personal sportsbook, in active development.

For each NBA game it predicts the winner, the margin, and the totals for points, rebounds and assists — the same again for the first quarter and the first half — plus five stats for each player expected to play. Every number comes from models trained here on eleven seasons of real results. Nothing is scraped from a bookmaker.

The NBA is the only sport so far. The pipeline is built so another can be added alongside it rather than replacing it.

---

## What it predicts

Three families, all reachable in the running app.

| Family | Markets | Notes |
|---|---|---|
| **Full game** | 7 | Home win probability, home margin, total points, rebound margin/total, assist margin/total |
| **First quarter and first half** | 6 | Winner, margin and total for each period. Q1 and 1H only, not all four quarters |
| **Player props** | 5 per player | Points, rebounds, assists, three-pointers made, and the three combined. Up to 10 players per team |

Two of the markets carry qualifiers that travel all the way to the screen rather than being hidden: the quarter and half winner probabilities are conditional on the period not being tied, and the first-quarter winner is labelled low confidence because it measurably is.

## Scope, stated plainly

**Predictions are for the next game only.** A game's features are built from each team's form, rest and Elo *going into* it, and a fixture two days out may have an unplayed game in between that changes those inputs. The schedule view lists fixtures months ahead; anything beyond the next matchday is shown with a visible reason rather than hidden.

**The data currently ends 2026-04-12**, the close of the 2025-26 regular season. The app says so on screen and marks itself stale. Fresh predictions resume once the pipeline is rerun with 2026-27 data.

**Roster availability is not yet applied at serving time.** Absence features measurably improve the spread prediction and are trained into the models, but computing them live needs an injury-report fetch the container cannot currently run. Player boards say so instead of implying a clean bill of health.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data pipeline | Python — pandas, numpy, nba_api |
| Model training & tracking | Python — scikit-learn, XGBoost, MLflow |
| Model serving | FastAPI, uvicorn |
| Backend | Java 21 (target) on JDK 25, Spring Boot 4.1, PostgreSQL 17 |
| Frontend | React 19 (create-react-app), plain CSS |
| Containerisation | Docker + Compose, all four services |
| CI | GitHub Actions — four parallel jobs |
| Continuous retraining | GitHub Actions on a self-hosted Windows runner, weekly |
| Not yet built | Cloud deployment, CD |

Direct dependencies are pinned in all three `requirements.txt` files.

## Repository structure

```
basketball-predictor/
├── data-pipeline/          Ingestion and feature engineering (Python)
│   ├── ingestion/          Games, player box scores, quarter scores,
│   │                       team advanced stats, live injury report
│   ├── preprocessing/      Validation, rolling features, rest days, Elo,
│   │                       availability, and the final dataset build
│   └── data/               raw/ and processed/ (mostly gitignored)
├── ml-training/            Baselines, XGBoost, tuning, finalisation
│   ├── models/             7 production models (committed)
│   ├── models_quarter_half/    6 models + manifest
│   ├── models_player_props/    10 artifacts + routing manifest
│   ├── continuous_retrain.py   Scheduled retrain with a promotion gate
│   └── model_evaluation.py     Shared scoring
├── inference-service/      FastAPI — /health, /schedule, three /predict routes
├── backend/                Spring Boot REST API + PostgreSQL
│   └── backend_insights.txt    Plain-language guide to the backend
├── frontend/               React — browse view and a two-tab detail view
├── infra/                  Terraform (not started)
├── .github/workflows/      ci.yml, continuous-retrain.yml
└── docker-compose.yml
```

## Running it

Everything, in one command:

```bash
docker compose up --build
```

Four services: `postgres`, `inference-service`, `backend`, `frontend`. The app is at `http://localhost:3000`.

To run the pieces separately, see the per-service notes in `CLAUDE.md`.

## Rebuilding the data and models

The pipeline scripts run in dependency order — ingestion, validation, features, Elo, then the final dataset — followed by the training scripts. Both fetchers skip what is already on disk, so a rerun pays only for genuinely new games.

The retrain job automates this weekly and **never deploys**: candidate models land in a directory nothing serves, and promotion opens a pull request for a human to merge.

---

## Status

Working end to end, locally and under Docker Compose.

- [x] Data pipeline — ingestion, validation, feature engineering, final dataset (13,199 games, 38 features)
- [x] Model training — baselines, XGBoost, tuning, finalised production models
- [x] Live feature computation for an unplayed matchup
- [x] Inference service — three prediction endpoints, health and schedule
- [x] Backend — Spring Boot, PostgreSQL persistence, cached fixture sync
- [x] Frontend — browse and detail views over all three prediction families
- [x] Docker and Compose — all four services
- [x] CI — four parallel jobs, all green
- [x] Continuous retraining with a promotion gate
- [ ] Cloud deployment
- [ ] CD on merge

### Next

Serving roster availability is the highest-value remaining work — it is the only change so far that measurably beat the accuracy ceiling — but it cannot be verified until the 2026-27 season opens and real injury reports exist. Cloud deployment and CD are the last two items on the original milestone list.

Longer term: market odds as a model input, drift monitoring across a season, and a second sport.

---

## A note on accuracy

Three independent model families — a closed-form Elo formula, linear/logistic regression, and tuned XGBoost — converge on the same accuracy band on every full-game target. That is an information ceiling in the feature set, not a modelling problem, and it is treated as one: further tuning is not pursued.

The one thing that has broken through it is player availability, worth roughly 5.8% on spread error. It is trained in and waiting on the serving work above.

Predictions are honest about their own weakness where they have one. A market that scores barely better than a base rate ships labelled as such rather than being quietly dropped or presented as equal to the rest.
