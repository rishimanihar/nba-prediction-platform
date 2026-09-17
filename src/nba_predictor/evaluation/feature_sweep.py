"""Compare purposeful logistic-regression feature sets with walk-forward evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.evaluation.walk_forward import (
    bootstrap_candidate_deltas,
    bootstrap_metric_intervals,
    probability_metrics,
)
from nba_predictor.features.team_form import build_pregame_features
from nba_predictor.models.feature_sets import FEATURE_SWEEP_GROUPS, FEATURE_SWEEP_SETS
from nba_predictor.models.logistic import build_logistic_pipeline

CANDIDATE_MODEL = "lean_logistic"


def _season_order(features: pd.DataFrame) -> list[str]:
    return features.groupby("season", sort=False)["game_date"].min().sort_values().index.tolist()


def _predict_logistic(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    model = build_logistic_pipeline()
    model.fit(train_rows[feature_columns], train_rows["home_win"])
    return model.predict_proba(test_rows[feature_columns])[:, 1]


def evaluate_feature_sweep(
    features: pd.DataFrame, holdout_season: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate curated feature combinations without using the reserved holdout."""
    seasons = _season_order(features)
    if holdout_season not in seasons:
        available = ", ".join(seasons)
        raise ValueError(f"Holdout season {holdout_season!r} not found. Available seasons: {available}")

    holdout_index = seasons.index(holdout_season)
    if holdout_index < 3:
        raise ValueError("At least three seasons before the holdout are required for feature-sweep evaluation.")

    fold_metrics: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for validation_index in range(2, holdout_index):
        validation_season = seasons[validation_index]
        train_seasons = seasons[:validation_index]
        train_rows = features[features["season"].isin(train_seasons)]
        test_rows = features[features["season"] == validation_season]
        for model_name, feature_columns in FEATURE_SWEEP_SETS.items():
            probabilities = _predict_logistic(train_rows, test_rows, feature_columns)
            metrics = probability_metrics(test_rows["home_win"], probabilities)
            fold_metrics.append(
                {
                    "model": model_name,
                    "validation_season": validation_season,
                    "train_through_season": train_seasons[-1],
                    "feature_groups": ",".join(FEATURE_SWEEP_GROUPS[model_name]),
                    "feature_count": len(feature_columns),
                    "train_rows": len(train_rows),
                    "test_rows": len(test_rows),
                    **metrics,
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "validation_season": validation_season,
                        "game_id": test_rows["game_id"].to_numpy(),
                        "home_win": test_rows["home_win"].to_numpy(),
                        "home_win_probability": probabilities,
                    }
                )
            )

    return pd.DataFrame(fold_metrics), pd.concat(prediction_rows, ignore_index=True)


def save_feature_sweep_report(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    holdout_season: str,
    output_dir: Path,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    """Save sweep results and paired evidence for the current lean candidate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        fold_metrics.groupby("model", sort=False)[["accuracy", "roc_auc", "log_loss", "brier_score"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(column).strip("_") for column in summary.columns.to_flat_index()]
    intervals = bootstrap_metric_intervals(predictions, n_resamples=bootstrap_resamples)
    baseline_models = tuple(model_name for model_name in FEATURE_SWEEP_SETS if model_name != CANDIDATE_MODEL)
    deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model=CANDIDATE_MODEL,
        baseline_models=baseline_models,
        n_resamples=bootstrap_resamples,
    )

    fold_metrics.to_csv(output_dir / "sweep_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "sweep_summary.csv", index=False)
    predictions.to_csv(output_dir / "sweep_predictions.csv", index=False)
    intervals.to_csv(output_dir / "sweep_metric_intervals.csv", index=False)
    deltas.to_csv(output_dir / "sweep_deltas.csv", index=False)
    (output_dir / "sweep_metadata.json").write_text(
        json.dumps(
            {
                "holdout_season": holdout_season,
                "holdout_evaluated": False,
                "candidate_model": CANDIDATE_MODEL,
                "feature_sweep_groups": FEATURE_SWEEP_GROUPS,
                "feature_sweep_sets": FEATURE_SWEEP_SETS,
                "bootstrap_resamples": bootstrap_resamples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to completed historical games CSV.")
    parser.add_argument("--holdout-season", required=True, help="Season to reserve from model selection.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/feature_sweep"))
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    args = parser.parse_args()

    features = build_pregame_features(pd.read_csv(args.input))
    fold_metrics, predictions = evaluate_feature_sweep(features, args.holdout_season)
    summary = save_feature_sweep_report(
        fold_metrics,
        predictions,
        args.holdout_season,
        args.output_dir,
        args.bootstrap_resamples,
    )
    print(summary.to_string(index=False))
    print(f"\nSaved feature sweep artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
