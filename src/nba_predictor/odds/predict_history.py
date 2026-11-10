"""Score a later NBA season with the frozen, earlier-season model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from nba_predictor.features.team_form import build_pregame_features
from nba_predictor.models.feature_sets import LEAN_LOGISTIC_FEATURE_COLUMNS


def score_later_season(
    training_games: pd.DataFrame,
    target_games: pd.DataFrame,
    model: object,
    metadata: dict,
) -> pd.DataFrame:
    """Build pregame features chronologically and score games outside training seasons."""
    target_seasons = target_games["season"].astype(str).unique().tolist()
    if len(target_seasons) != 1:
        raise ValueError("Target games must contain exactly one season")
    target_season = target_seasons[0]
    trained_seasons = set(metadata["artifact_trained_seasons"])
    if target_season in trained_seasons:
        raise ValueError("Target season was included in model training")
    if set(training_games["season"].astype(str)) != trained_seasons:
        raise ValueError("Training games do not match the model's recorded training seasons")
    if metadata["feature_columns"] != LEAN_LOGISTIC_FEATURE_COLUMNS:
        raise ValueError("Model feature columns do not match the approved feature set")
    if pd.to_datetime(training_games["game_date"]).max() >= pd.to_datetime(
        target_games["game_date"]
    ).min():
        raise ValueError("Training games must precede the target season")
    if set(training_games["game_id"].astype(str)) & set(target_games["game_id"].astype(str)):
        raise ValueError("Training and target games overlap")
    all_games = pd.concat([training_games, target_games], ignore_index=True)
    appearances = pd.concat([
        all_games[["game_date", "home_team"]].rename(columns={"home_team": "team"}),
        all_games[["game_date", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    if appearances.duplicated(["game_date", "team"]).any():
        raise ValueError("A team has multiple games on one date; tip times are needed")
    features = build_pregame_features(all_games)
    target_rows = features.loc[features["season"].eq(target_season)].copy()
    probabilities = model.predict_proba(target_rows[LEAN_LOGISTIC_FEATURE_COLUMNS])[:, 1]
    return pd.DataFrame({
        "model": "lean_logistic",
        "target_season": target_season,
        "game_id": target_rows["game_id"].astype(str).to_numpy(),
        "game_date": target_rows["game_date"].dt.date.astype(str).to_numpy(),
        "home_team": target_rows["home_team"].to_numpy(),
        "away_team": target_rows["away_team"].to_numpy(),
        "home_win": target_rows["home_win"].to_numpy(),
        "home_win_probability": probabilities,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-games", type=Path, default=Path("data/raw/games.csv"))
    parser.add_argument("--target-games", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/home_win_lean_logistic.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/metrics_lean_logistic.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/odds/season_predictions.csv"))
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    predictions = score_later_season(
        pd.read_csv(args.training_games, dtype={"game_id": str}),
        pd.read_csv(args.target_games, dtype={"game_id": str}),
        joblib.load(args.model),
        metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} out-of-sample predictions to {args.output}")


if __name__ == "__main__":
    main()
