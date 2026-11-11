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
- `POST /value`: compares those probabilities with a two-way American moneyline, removes
  the sportsbook vig, and reports model edge and estimated net return per unit staked.
- `GET /games/upcoming`: returns today's automatically calculated matchup predictions from
  a fresh saved schedule snapshot.

The prediction request accepts the 16 independent pregame values. The service derives back-to-back flags, Elo difference, and Elo home-win probability so callers cannot submit internally inconsistent values. By default, it loads `artifacts/home_win_lean_logistic.joblib`, `artifacts/metrics_lean_logistic.json`, and the corrected final evaluation report. These paths can be overridden with `NBA_MODEL_PATH`, `NBA_MODEL_METADATA_PATH`, and `NBA_FINAL_EVALUATION_PATH`.

The value endpoint additionally accepts home and away moneylines from the same sportsbook
and timestamp. Its expected-value output is a model estimate, not a guarantee or a wagering
recommendation. Historical profitability must be evaluated on odds that were actually
available before each game's tip-off.

### Generate predictions for scheduled games

Run the prediction job once each NBA game day from the repository root:

```powershell
python -m nba_predictor.predict_upcoming --season 2026-27 --history data/raw/games.csv data/raw/games_2025_26.csv
```

The job fetches one current-season NBA schedule, combines previous seasons with completed
current-season games strictly before today in Eastern time, calculates the same rolling
features used in training, and scores only scheduled regular-season games for today. It
writes `artifacts/upcoming_predictions.json` with model version, generation time, matchup,
probabilities, and predicted winner. No odds feed or API key is needed. The frozen model
remains trained through 2024-25; 2025-26 results add pregame context but do not retrain it.

Start the API and request `GET /games/upcoming`. The endpoint reads the latest snapshot on
each request and returns HTTP 503 if it is missing, from another date, or for a different
model artifact. On a day without scheduled games it returns an empty games list. For an
offline replay, pass `--schedule-csv path/to/schedule.csv --date YYYY-MM-DD`; the saved
schedule must use NBA API's `ScheduleLeagueV2` column names. The live fetch scores only
today's Eastern date; future dates need future results that are not known yet. In Docker,
provide the JSON
snapshot as a read-only mount at `/app/artifacts/upcoming_predictions.json`, or set
`NBA_UPCOMING_PREDICTIONS_PATH` to the mounted path. Refresh this file daily with a job
scheduler when deploying the service.

### Refresh predictions every morning on Windows

Install a local Windows Scheduled Task once from your regular PowerShell terminal:

```powershell
powershell.exe -NoProfile -File .\scripts\install_daily_predictions_task.ps1 -WhatIf
powershell.exe -NoProfile -File .\scripts\install_daily_predictions_task.ps1 -RunNow
```

The first command previews the task. The second registers `NBA Predictor Daily Predictions`
and starts one refresh immediately; that first run may take several minutes. The task runs
at **9:00 AM in your computer's local time zone**. It runs as your signed-in Windows user,
requires a network connection, and starts after a missed run when the computer becomes
available. The runner tries the NBA schedule request up to three times, then writes each
attempt to `artifacts/logs/upcoming_YYYY-MM-DD.log`. A failed refresh leaves the last
complete JSON file in place; the API rejects that file once its date is stale. The job
does not call an odds service or require an API key.

The task uses the current `2026-27` season and the two historical files named above.
Update `scripts/run_daily_predictions.ps1` when a new NBA season starts and its prior
season games file has been saved. This local task runs only while you are signed in and
the computer is on. A deployed API will need its own scheduled job and a shared or mounted
snapshot file. To inspect or remove the local task, open Windows Task Scheduler and search
for `NBA Predictor Daily Predictions`.

## Ingest historical NBA moneylines and backtest value

### Free OddsPapi route

Create a free [OddsPapi account](https://oddspapi.io/us/sign-up) and copy its key from account settings. In PowerShell, run these commands in the project directory and keep the terminal open:

```powershell
$secure = Read-Host "Paste OddsPapi key" -AsSecureString
$env:ODDSPAPI_API_KEY = [System.Net.NetworkCredential]::new("", $secure).Password
python -m nba_predictor.odds.oddspapi --check-key
```

The check calls the unmetered account endpoint and prints only whether the key was accepted. It does not store the key in a file. OddsPapi's free historical archive begins in January 2026, so use 2025-26 or newer NBA games for matching; the existing `data/raw/games.csv` ends at 2024-25. Once a matching games CSV is available, ingest a UTC fixture day:

```powershell
python -m nba_predictor.odds.oddspapi --date 2026-02-01 --games data/raw/games_2025_26.csv --bookmakers fanduel,draftkings,pinnacle
```

This writes `data/raw/oddspapi_moneylines.csv` and an `_unmatched.csv` file. OddsPapi provides decimal prices and timestamped line moves; the importer reconstructs active two-way market `111` quotes before tip, preserves the exact decimal prices, and includes approximate American equivalents. The backtest uses the exact decimal prices when present. The historical endpoint permits up to three bookmaker slugs per request and has a five-second cooldown. Fixture discovery uses a request from the free monthly allowance.

Collect a range of UTC fixture dates with daily checkpoints. This example includes the already collected January 16 file without fetching it again:

```powershell
python -m nba_predictor.odds.oddspapi_batch --start-date 2026-01-17 --end-date 2026-01-23 --games data/raw/games_2025_26.csv --bookmakers fanduel --include data/raw/oddspapi_2026-01-16.csv
```

The batch job writes one matched and one unmatched CSV per date under `data/raw/oddspapi_days/`, then writes `data/raw/oddspapi_combined.csv`. Rerunning the same command skips completed dates. A manifest prevents reuse of saved daily data with a different bookmaker list or games path. If an API request fails, rerun the command to continue from the last complete day. The seven-day example can take several minutes because the historical endpoint has a five-second cooldown per fixture.
If OddsPapi returns `404 NOT_FOUND` for one fixture's historical odds, ingestion records its ID and teams in that day's `_missing_history.csv` and continues. Inspect those files when reviewing coverage; a missing archive is not a matched quote. Other HTTP errors still stop the batch for investigation.
If the fixtures endpoint returns `404 FIXTURE_NOT_FOUND` for a date without NBA games, the
batch saves empty daily files and continues to the next date.
The importer waits between empty fixture days and retries temporary `429 RATE_LIMITED`
responses using the provider's wait time. A `REQUEST_LIMIT_EXCEEDED` quota error still stops
the batch; saved daily files allow a later resume.
The range filters OddsPapi fixtures by UTC date; an NBA game on a US evening may be on the next UTC date. Query boundary dates on both sides of the desired US calendar window and let the batch job remove repeated quotes. Quotes for postponed games remain unmatched when the listed tip date does not correspond to a completed NBA game; do not attach them to the later rescheduled result.

To backtest against the 2025-26 season, score it with the frozen model trained through 2024-25, then join the quotes at a fixed pre-tip lead time:

```powershell
python -m nba_predictor.odds.predict_history --target-games data/raw/games_2025_26.csv --output artifacts/odds/predictions_2025_26.csv
python -m nba_predictor.odds.backtest --quotes data/raw/oddspapi_2026-01-16.csv --predictions artifacts/odds/predictions_2025_26.csv --output artifacts/odds/value_backtest_2026-01-16.csv --hours-before-tip 1
```

The prediction command checks that the target season was excluded from training and that each team has at most one game per calendar date. A single day's betting return is only a pipeline check; use many dates and review coverage before drawing conclusions about profitability.

After running the value backtest, save a reproducible calibration and selection report:

```powershell
python -m nba_predictor.odds.diagnostics --backtest artifacts/odds/value_backtest_2026-01-16_to_31.csv --output-dir artifacts/odds/diagnostics_2026_01
```

The report contains model and no-vig market Brier scores and log losses, five fixed probability bins, and four fixed estimated-value bins. These bins describe the held-out results; do not choose new betting thresholds from the same games and present their performance as an independent test.

### Compare game predictions with the market

The main project goal is predicting NBA winners. Score the frozen model on every completed
2025-26 game, then compare it with FanDuel only where an eligible pregame quote exists:

```powershell
python -m nba_predictor.evaluation.outcome_comparison --games data/raw/games_2025_26.csv --predictions artifacts/odds/predictions_2025_26.csv --backtest artifacts/odds/value_backtest_through_2026-03-01.csv --hours-before-tip 1 --output artifacts/odds/outcome_comparison_2025_26.json
```

The JSON report gives accuracy, ROC AUC, Brier score, and log loss for the model on the full
season and for both the model and the no-vig bookmaker on exactly the same priced games.
Lower Brier score and log loss are better; higher accuracy and ROC AUC are better. The
report checks game identities, outcomes, model probabilities, and quote times before scoring.
It also gives a 95% paired bootstrap interval for the market's advantage on each metric.
Positive advantage means the market performed better; an interval crossing zero means the
sample does not clearly distinguish the two. The bootstrap resamples whole game dates,
keeping games from the same date together. Use `--bootstrap-resamples` and `--bootstrap-seed`
to control the reproducible calculation.
The odds-covered sample is smaller and drawn from a specific date range, so its scores
should not be treated as the model's full-season performance. Running this command uses
saved local files and makes no odds API requests.

### The Odds API route

The historical odds job uses [The Odds API's historical NBA endpoint](https://the-odds-api.com/liveapi/guides/v4/), which requires a plan with historical access. It requests `basketball_nba`, the `h2h` market, and American odds at explicit UTC snapshot times. The provider returns the closest snapshot at or before the requested time. A key is read from `ODDS_API_KEY` and is never written to the output.

```powershell
$env:ODDS_API_KEY = "your-key"
python -m nba_predictor.odds.historical --snapshot 2024-10-22T20:00:00Z --snapshot 2024-10-23T00:00:00Z
```

The command writes `data/raw/historical_moneylines.csv` and a companion `_unmatched.csv`. Each matched row contains the NBA game ID, provider event ID, actual snapshot time, market update time, scheduled tip time, both team codes, bookmaker, and both prices. It accepts only complete two-way quotes with both timestamps before the provider's tip time. Matching uses the provider's tip date in Eastern time plus the exact home and away teams; zero or multiple NBA game matches go to the unmatched file. Review unmatched rows before using the data. The existing `games.csv` does not have an authoritative tip timestamp, so the provider's scheduled tip is the time boundary; a corrected tip time from an NBA schedule feed would strengthen this check.

Backtest with **out-of-sample** predictions, for example the existing final holdout predictions:

```powershell
python -m nba_predictor.odds.backtest --predictions artifacts/final_holdout_corrected/holdout_predictions.csv --hours-before-tip 1
```

This saves `artifacts/odds/value_backtest.csv`. It selects the latest eligible snapshot per game and bookmaker, uses the model's probability and the offered price to choose a positive expected-value side, and records realized net return for a one-unit stake. ROI in the command output is net return divided by the number of bets. Coverage depends on the paid feed, chosen snapshots, and the prediction season. Do not use predictions from a model trained on the same games as an historical profitability estimate.

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

Next, run the scheduled-game prediction job daily and mount its output into the API container.
Then build the frontend against `GET /games/upcoming`, add prediction freshness monitoring,
and version a retrained model after preserving the completed 2025-26 evaluation.
