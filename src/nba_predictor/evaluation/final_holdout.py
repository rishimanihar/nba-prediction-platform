"""Evaluate the frozen lean logistic model once on the reserved final season."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.evaluation.walk_forward import bootstrap_metric_intervals, probability_metrics
from nba_predictor.features.team_form import build_pregame_features
from nba_predictor.models.feature_sets import LEAN_LOGISTIC_FEATURE_COLUMNS
from nba_predictor.models.logistic import build_logistic_pipeline

MODEL_NAME = "lean_logistic"


def _season_order(features: pd.DataFrame) -> list[str]:
    return features.groupby("season", sort=False)["game_date"].min().sort_values().index.tolist()


def evaluate_final_holdout(
    features: pd.DataFrame, holdout_season: str
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train on seasons before the holdout and evaluate the frozen feature set once."""
    seasons = _season_order(features)
    if holdout_season not in seasons:
        available = ", ".join(seasons)
        raise ValueError(f"Holdout season {holdout_season!r} not found. Available seasons: {available}")

    holdout_index = seasons.index(holdout_season)
    if holdout_index < 2:
        raise ValueError("At least two seasons before the final holdout are required.")
    if holdout_index != len(seasons) - 1:
        raise ValueError("The final holdout must be the latest season in the input data.")

    train_seasons = seasons[:holdout_index]
    train_rows = features[features["season"].isin(train_seasons)]
    holdout_rows = features[features["season"] == holdout_season]
    if train_rows["home_win"].nunique() < 2 or holdout_rows["home_win"].nunique() < 2:
        raise ValueError("Training and holdout rows must both contain home wins and home losses.")

    model = build_logistic_pipeline()
    model.fit(train_rows[LEAN_LOGISTIC_FEATURE_COLUMNS], train_rows["home_win"])
    probabilities = np.clip(
        model.predict_proba(holdout_rows[LEAN_LOGISTIC_FEATURE_COLUMNS])[:, 1],
        1e-6,
        1.0 - 1e-6,
    )
    metrics: dict[str, object] = {
        "model": MODEL_NAME,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "holdout_season": holdout_season,
        "train_through_season": train_seasons[-1],
        "training_seasons": train_seasons,
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "feature_count": len(LEAN_LOGISTIC_FEATURE_COLUMNS),
        "feature_columns": LEAN_LOGISTIC_FEATURE_COLUMNS,
        **probability_metrics(holdout_rows["home_win"], probabilities),
    }
    predictions = pd.DataFrame(
        {
            "model": MODEL_NAME,
            "holdout_season": holdout_season,
            "game_id": holdout_rows["game_id"].to_numpy(),
            "game_date": holdout_rows["game_date"].to_numpy(),
            "home_team": holdout_rows["home_team"].to_numpy(),
            "away_team": holdout_rows["away_team"].to_numpy(),
            "home_win": holdout_rows["home_win"].to_numpy(),
            "home_win_probability": probabilities,
        }
    )
    return metrics, predictions


def save_final_holdout_report(
    metrics: dict[str, object],
    predictions: pd.DataFrame,
    output_dir: Path,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    """Save the final report, refusing to replace an existing evaluation."""
    output_paths = (
        output_dir / "holdout_metrics.json",
        output_dir / "holdout_metric_intervals.csv",
        output_dir / "holdout_predictions.csv",
    )
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(f"Final holdout artifacts already exist: {existing}")

    intervals = bootstrap_metric_intervals(predictions, n_resamples=bootstrap_resamples)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths[0].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    intervals.to_csv(output_paths[1], index=False)
    predictions.to_csv(output_paths[2], index=False)
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to completed historical games CSV.")
    parser.add_argument("--holdout-season", required=True, help="Reserved final evaluation season.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_holdout"))
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    args = parser.parse_args()

    features = build_pregame_features(pd.read_csv(args.input))
    metrics, predictions = evaluate_final_holdout(features, args.holdout_season)
    intervals = save_final_holdout_report(
        metrics,
        predictions,
        args.output_dir,
        args.bootstrap_resamples,
    )
    print(json.dumps(metrics, indent=2))
    print("\n95% bootstrap metric intervals")
    print(intervals.to_string(index=False))
    print(f"\nSaved final holdout artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
