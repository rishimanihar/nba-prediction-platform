# NBA prediction platform: Codex handoff

Read `README.md` before changing the pipeline. Explain each step in plain language; the project owner is learning Codex CLI.

## Current project

- Python package under `src/nba_predictor`, with a basketball-only pregame model and FastAPI `/predict`, `/value`, and `/games/upcoming` endpoints.
- The frozen production model was trained through the 2024-25 NBA season. The 2025-26 games are reserved for out-of-sample predictions and odds backtesting.
- Historical NBA moneyline ingestion, game matching, batch checkpoints, value backtesting, and diagnostics are implemented under `src/nba_predictor/odds`.
- OddsPapi reads its key from `ODDSPAPI_API_KEY`. Never print or commit the key. Raw CSVs and most generated artifacts are ignored by Git.
- `nba_predictor.predict_upcoming` uses the NBA season schedule and completed history to generate today's scoreless-fixture predictions in `artifacts/upcoming_predictions.json`. The API serves only a snapshot for the current Eastern date and loaded model version.
- `scripts/run_daily_predictions.ps1` refreshes the snapshot with retries and a dated log. `scripts/install_daily_predictions_task.ps1` registers a 9:00 AM local Windows task; run that installer from the owner's PowerShell session, never from the Codex sandbox account.

## Current evaluation state

- The 2025-26 local games file contains 1,230 completed games. Frozen-model out-of-sample predictions cover all 1,230.
- The saved FanDuel backtest through March 1 contains eligible pregame quotes for 287 games. `nba_predictor.evaluation.outcome_comparison` scores the full season and compares model and no-vig market on the same priced games, including paired game-date bootstrap intervals.
- The current 287-game sample favors the market on ROC AUC, Brier score, and log loss; the accuracy interval includes zero. Treat these as held-out observations, not a training set or a claim of profitable betting.
- OddsPapi batch checkpoints and raw quotes are local and ignored by Git. Missing historical odds remain missing; never treat them as valid quotes.

## Local setup and useful checks

- Run commands from this repository root. The Python virtual environment is `.venv`; on PowerShell activate with `.\.venv\Scripts\Activate.ps1` if needed.
- The package is installed in editable mode when the environment has been set up: `python -m pip install -e ".[dev,training]"`.
- Check code with `pytest` and `ruff check .` when making relevant changes.
- Before any odds API calls, confirm the intended date range, bookmaker, expected request cost, and that `ODDSPAPI_API_KEY` is available in the current shell. Do not store the key in this file.
