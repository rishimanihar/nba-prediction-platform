# NBA Prediction Platform

An end-to-end ML engineering project that estimates the home team's pre-game NBA win probability and expected point differential. The initial model uses only information available before tip-off, then grows into a deployed frontend application.

## Why this project

The goal is not to claim certainty about sports outcomes. The goal is to demonstrate careful ML engineering: reproducible data, leakage-safe features, time-aware evaluation, model serving, containerization, orchestration, and monitoring.

## The v1 prediction

For every scheduled game, the system will provide:

- home-team win probability
- predicted point differential
- model version and training date
- a concise explanation of the most important pre-game inputs

## Build order

1. **Baseline ML**: load historical games, build pre-game features, train a logistic regression model, and record metrics.
2. **Data layer**: add a repeatable NBA data ingestion job and PostgreSQL storage.
3. **Serving**: expose predictions through FastAPI.
4. **Product**: build a Next.js dashboard for upcoming games and model performance.
5. **MLOps**: track experiments and model versions with MLflow; automate retraining and evaluation.
6. **Deployment**: run locally with Docker Compose, then deploy services and scheduled jobs to Kubernetes.

## First milestone: baseline model

The first model deliberately stays simple. It predicts whether the home team wins using rolling pre-game team form, scoring margin, rest days, schedule density, venue-specific form, and pre-game Elo ratings. Every feature is calculated before the predicted game starts. A simple logistic-regression model gives later model changes an honest point of comparison.

### Prerequisites

Install these before running the code:

- Python 3.11 or newer
- Git
- Docker Desktop (needed in a later milestone, not today)

On Windows, install Python from [python.org](https://www.python.org/downloads/) and select **Add Python to PATH** during setup. Docker Desktop can wait until the API is working locally.

### Create an environment

```powershell
cd C:\Users\rishi\Documents\Codex\2026-09-10\i-w\outputs\nba-prediction-platform
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,training]"
```

### Download historical games

The project uses the `nba_api` Python package, which reads NBA.com's `ScheduleLeagueV2` endpoint. Unlike matchup text, this source identifies home and away teams explicitly for neutral-site games. Start with the last seven completed regular seasons. This may take a few minutes because the script deliberately pauses between requests.

```powershell
python -m nba_predictor.data.nba_api_ingest --start-season 2018 --end-season 2024
```

This writes `data/raw/games.csv`. Ingestion fails if a configured completed season has missing or duplicate games. The normalized file has these columns:

```text
game_id,game_date,season,home_team,away_team,home_score,away_score
```

`game_date` is an ISO date such as `2024-10-22`. The pipeline keeps only completed regular-season games. It intentionally does not use box-score information from the game being predicted as a feature.

### Train the baseline

```powershell
python -m nba_predictor.train --input data/raw/games.csv --holdout-season 2024-25
```

The command evaluates the lean logistic model with an 80/20 chronological development split, then refits the saved artifact on all development seasons. The optional holdout season is excluded from both the evaluation split and saved artifact.

### Run tests

```powershell
pytest
ruff check .
```

### Run walk-forward evaluation

Reserve the newest complete season from model selection and evaluate candidate models on each earlier season in chronological order:

```powershell
python -m nba_predictor.evaluation.walk_forward --input data/raw/games.csv --holdout-season 2024-25
```

The command compares an always-home accuracy reference, a training-period home-win-rate probability reference, Elo-only forecasts, baseline, enhanced, and lean logistic-regression feature sets, a regularized histogram gradient-boosting challenger, and a regularized XGBoost challenger. The lean set removes only the two 5-game win-rate features. It writes fold-by-fold metrics, a mean-and-standard-deviation summary, out-of-fold predictions, calibration points, a calibration chart, and 95% paired bootstrap confidence intervals to `artifacts/evaluation/`. It does not evaluate the reserved holdout season.

### Run feature ablation and coefficient analysis

Measure the contribution of form, venue, schedule, and Elo feature groups without evaluating the reserved holdout season:

```powershell
python -m nba_predictor.evaluation.feature_ablation --input data/raw/games.csv --holdout-season 2024-25
```

The report in `artifacts/ablation/` includes group-removal metrics, paired bootstrap intervals, and standardized logistic-regression coefficients. Coefficients describe association within the model; ablation results are the stronger evidence for a group's predictive contribution.

### Run the curated feature sweep

Compare six purposeful logistic-regression feature combinations, while keeping the 2024-25 season untouched for final evaluation:

```powershell
python -m nba_predictor.evaluation.feature_sweep --input data/raw/games.csv --holdout-season 2024-25
```

The sweep tests Elo alone and targeted additions of venue form, medium-term form, and schedule context. It deliberately excludes the two five-game win-rate features because the ablation did not support them. Results are saved to `artifacts/feature_sweep/`; `sweep_deltas.csv` compares the current lean feature set directly with every smaller set using paired bootstrap intervals.

### Run the final holdout evaluation

After freezing the lean logistic model and feature set, evaluate it once on the untouched 2024-25 season:

```powershell
python -m nba_predictor.evaluation.final_holdout --input data/raw/games.csv --holdout-season 2024-25
```

The command trains only on earlier seasons and saves final metrics, 95% bootstrap intervals, and per-game predictions to `artifacts/final_holdout/`. It refuses to overwrite an existing final report, which protects the holdout from becoming another model-selection dataset.

## Serve the production model

After the final holdout evaluation is accepted, reinstall the project to add the API dependencies and retrain the approved model on every completed game through 2024-25:

```powershell
python -m pip install -e ".[dev,training]"
python -m nba_predictor.train --input data/raw/games.csv
```

The second command replaces the development artifact with a production artifact trained on all 8,289 games. Start the API from the repository root:

```powershell
uvicorn nba_predictor.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` to exercise the API interactively. Available endpoints are:

- `GET /health`: confirms that the service and model loaded successfully.
- `GET /model`: reports training data, feature version, and final holdout evidence.
- `POST /predict`: returns home- and away-win probabilities for validated pregame features.

The prediction request accepts the 16 independent pregame values. The service derives back-to-back flags, Elo difference, and Elo home-win probability so callers cannot submit internally inconsistent values. By default, it loads `artifacts/home_win_lean_logistic.joblib`, `artifacts/metrics_lean_logistic.json`, and the corrected final evaluation report. These paths can be overridden with `NBA_MODEL_PATH`, `NBA_MODEL_METADATA_PATH`, and `NBA_FINAL_EVALUATION_PATH`.

## Run the API with Docker

Docker packages the API runtime, source code, and approved model artifacts into one reproducible image. Build it from the repository root:

```powershell
docker build --tag nba-predictor-api:local .
```

Run the container and publish its internal port `8000` to port `8000` on the host:

```powershell
docker run --rm --name nba-predictor-api --publish 8000:8000 nba-predictor-api:local
```

In a second PowerShell terminal, exercise the health, metadata, and prediction endpoints:

```powershell
.\scripts\smoke_test_api.ps1
docker inspect --format '{{.State.Health.Status}}' nba-predictor-api
docker image inspect --format '{{.Config.User}}' nba-predictor-api:local
```

The smoke test should pass, the container health should be `healthy` after its startup period, and the configured user should be `10001:10001`. The image contains only API dependencies; install the `training` extra locally for ingestion, evaluation, and model training.

## What comes next

Next, the API will be packaged with Docker and connected to a scheduled feature-generation pipeline. The frontend can then request predictions without constructing model features itself, followed by Kubernetes deployment and monitoring.
