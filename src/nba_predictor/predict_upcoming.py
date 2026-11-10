"""Generate today's NBA predictions from the season schedule and completed games."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from nba_predictor.data.nba_api_ingest import (
    fetch_season,
    normalize_schedule_fixtures,
    normalize_schedule_games,
)
from nba_predictor.features.team_form import REQUIRED_COLUMNS, build_fixture_features
from nba_predictor.models.feature_sets import LEAN_LOGISTIC_FEATURE_COLUMNS


def generate_predictions(
    historical_games: pd.DataFrame,
    season_schedule: pd.DataFrame,
    season: str,
    game_date: date,
    model: object,
    metadata: dict,
    generated_at: datetime | None = None,
) -> dict:
    """Use only results from dates before game_date to predict scheduled games."""
    missing = REQUIRED_COLUMNS - set(historical_games.columns)
    if missing:
        raise ValueError(f"Historical games are missing columns: {', '.join(sorted(missing))}")
    if historical_games.empty:
        raise ValueError("Historical games cannot be empty")
    if season_schedule.empty:
        raise ValueError("The season schedule is empty")
    if metadata.get("feature_columns") != LEAN_LOGISTIC_FEATURE_COLUMNS:
        raise ValueError("Model metadata does not match the lean feature set")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Model must classify home losses and wins as 0 and 1")
    history = historical_games.copy()
    history["game_id"] = history["game_id"].astype(str).str.zfill(10)
    history_dates = pd.to_datetime(history["game_date"], errors="raise")
    if history_dates.isna().any() or history_dates.dt.date.ge(game_date).any():
        raise ValueError("Historical games must all precede the prediction date")
    if (
        history["game_id"].duplicated().any()
        or history[["home_score", "away_score"]].isna().any().any()
    ):
        raise ValueError("Historical games need unique IDs and completed scores")
    if season in set(history["season"].astype(str)):
        raise ValueError(
            "Pass previous seasons as history; current-season results come from the schedule"
        )
    trained_seasons = set(metadata["artifact_trained_seasons"])
    if not trained_seasons.issubset(set(history["season"].astype(str))):
        raise ValueError("Historical games do not cover the model's training seasons")

    scheduled = normalize_schedule_fixtures(season_schedule, season, game_date.isoformat())
    completed = normalize_schedule_games(season_schedule, season)
    completed = completed.loc[pd.to_datetime(completed["game_date"]).dt.date.lt(game_date)]
    current_ids = season_schedule["gameId"].astype(str).str.zfill(10)
    current_status = pd.to_numeric(season_schedule["gameStatus"], errors="coerce")
    current_dates = pd.to_datetime(season_schedule["gameDate"], errors="raise").dt.date
    expected_completed = (
        current_ids.str.startswith("002") & current_status.eq(3) & current_dates.lt(game_date)
    ).sum()
    if len(completed) != expected_completed:
        raise ValueError("The schedule has completed games with missing scores")
    all_completed = pd.concat([history, completed], ignore_index=True)
    if all_completed["game_id"].duplicated().any():
        raise ValueError("Historical games overlap current-season results")
    if scheduled.empty:
        game_predictions: list[dict] = []
    else:
        features = build_fixture_features(all_completed, scheduled)
        probabilities = model.predict_proba(features[LEAN_LOGISTIC_FEATURE_COLUMNS])[:, 1]
        if (
            not np.isfinite(probabilities).all()
            or not np.logical_and(0 <= probabilities, probabilities <= 1).all()
        ):
            raise ValueError("Model returned invalid probabilities")
        game_predictions = [
            {
                "game_id": str(row.game_id),
                "game_date": row.game_date.date().isoformat(),
                "season": row.season,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_win_probability": float(probability),
                "away_win_probability": float(1 - probability),
                "predicted_winner": row.home_team if probability >= 0.5 else row.away_team,
            }
            for row, probability in zip(
                features.itertuples(index=False), probabilities, strict=True
            )
        ]
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must have a timezone")
    return {
        "as_of_date": game_date.isoformat(),
        "generated_at": timestamp.isoformat(),
        "model_name": str(metadata["model"]),
        "model_trained_at": str(metadata["trained_at"]),
        "games": game_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="Season label, such as 2026-27")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=datetime.now(ZoneInfo("America/New_York")).date(),
        help="NBA calendar date to score; defaults to today in Eastern time",
    )
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--schedule-csv", type=Path, help="Use a saved schedule instead of NBA.com")
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/home_win_lean_logistic.joblib")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("artifacts/metrics_lean_logistic.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/upcoming_predictions.json"))
    args = parser.parse_args()
    if args.schedule_csv is None and args.date != datetime.now(ZoneInfo("America/New_York")).date():
        parser.error("Live schedule predictions must use today's Eastern date")
    schedule = (
        pd.read_csv(args.schedule_csv, dtype={"gameId": str})
        if args.schedule_csv
        else fetch_season(args.season)
    )
    history = pd.concat(
        [pd.read_csv(path, dtype={"game_id": str}) for path in args.history], ignore_index=True
    )
    snapshot = generate_predictions(
        history,
        schedule,
        args.season,
        args.date,
        joblib.load(args.model),
        json.loads(args.metadata.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Saved {len(snapshot['games'])} scheduled-game predictions to {args.output}")


if __name__ == "__main__":
    main()
