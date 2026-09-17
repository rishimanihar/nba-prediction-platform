"""Measure the predictive contribution of NBA feature groups with walk-forward ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.evaluation.walk_forward import bootstrap_candidate_deltas, probability_metrics
from nba_predictor.features.team_form import build_pregame_features
from nba_predictor.models.feature_sets import ENHANCED_FEATURE_COLUMNS, FEATURE_GROUPS
from nba_predictor.models.logistic import build_logistic_pipeline

FULL_MODEL_NAME = "full_logistic"


def feature_columns_without(group_name: str) -> list[str]:
    """Return the enhanced feature set with one named group removed."""
    if group_name not in FEATURE_GROUPS:
        available = ", ".join(FEATURE_GROUPS)
        raise ValueError(f"Unknown feature group {group_name!r}. Available groups: {available}")
    excluded_columns = set(FEATURE_GROUPS[group_name])
    return [column for column in ENHANCED_FEATURE_COLUMNS if column not in excluded_columns]


def _season_order(features: pd.DataFrame) -> list[str]:
    return features.groupby("season", sort=False)["game_date"].min().sort_values().index.tolist()


def _predict_logistic(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    model = build_logistic_pipeline()
    model.fit(train_rows[feature_columns], train_rows["home_win"])
    return model.predict_proba(test_rows[feature_columns])[:, 1]


def evaluate_feature_ablations(features: pd.DataFrame, holdout_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the complete model and one group-removed model for each feature group."""
    seasons = _season_order(features)
    if holdout_season not in seasons:
        available = ", ".join(seasons)
        raise ValueError(f"Holdout season {holdout_season!r} not found. Available seasons: {available}")

    holdout_index = seasons.index(holdout_season)
    if holdout_index < 3:
        raise ValueError("At least three seasons before the holdout are required for ablation evaluation.")

    model_feature_sets = {FULL_MODEL_NAME: ENHANCED_FEATURE_COLUMNS}
    model_feature_sets.update(
        {f"without_{group_name}": feature_columns_without(group_name) for group_name in FEATURE_GROUPS}
    )
    fold_metrics: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for validation_index in range(2, holdout_index):
        validation_season = seasons[validation_index]
        train_seasons = seasons[:validation_index]
        train_rows = features[features["season"].isin(train_seasons)]
        test_rows = features[features["season"] == validation_season]
        for model_name, feature_columns in model_feature_sets.items():
            probabilities = _predict_logistic(train_rows, test_rows, feature_columns)
            metrics = probability_metrics(test_rows["home_win"], probabilities)
            fold_metrics.append(
                {
                    "model": model_name,
                    "validation_season": validation_season,
                    "train_through_season": train_seasons[-1],
                    "features_removed": model_name.removeprefix("without_") if model_name != FULL_MODEL_NAME else "",
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


def extract_coefficients(features: pd.DataFrame, holdout_season: str) -> pd.DataFrame:
    """Fit the approved model before the holdout and export its standardized coefficients."""
    train_rows = features[features["season"] != holdout_season]
    model = build_logistic_pipeline()
    model.fit(train_rows[ENHANCED_FEATURE_COLUMNS], train_rows["home_win"])
    coefficients = model.named_steps["classifier"].coef_[0]
    coefficient_frame = pd.DataFrame(
        {
            "feature": ENHANCED_FEATURE_COLUMNS,
            "standardized_coefficient": coefficients,
            "odds_ratio_per_standard_deviation": np.exp(coefficients),
        }
    )
    coefficient_frame["association"] = np.where(
        coefficient_frame["standardized_coefficient"] >= 0, "increases_home_win_probability", "decreases_home_win_probability"
    )
    return coefficient_frame.assign(abs_coefficient=coefficient_frame["standardized_coefficient"].abs()).sort_values(
        "abs_coefficient", ascending=False
    )


def save_ablation_report(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    holdout_season: str,
    output_dir: Path,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    """Save feature-group ablations and coefficient evidence without touching the holdout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        fold_metrics.groupby("model", sort=False)[["accuracy", "roc_auc", "log_loss", "brier_score"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(column).strip("_") for column in summary.columns.to_flat_index()]
    ablation_models = tuple(f"without_{group_name}" for group_name in FEATURE_GROUPS)
    ablation_deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model=FULL_MODEL_NAME,
        baseline_models=ablation_models,
        n_resamples=bootstrap_resamples,
    )

    fold_metrics.to_csv(output_dir / "ablation_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    predictions.to_csv(output_dir / "ablation_predictions.csv", index=False)
    ablation_deltas.to_csv(output_dir / "ablation_deltas.csv", index=False)
    coefficients.to_csv(output_dir / "feature_coefficients.csv", index=False)
    (output_dir / "ablation_metadata.json").write_text(
        json.dumps(
            {
                "holdout_season": holdout_season,
                "holdout_evaluated": False,
                "feature_groups": FEATURE_GROUPS,
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
    parser.add_argument("--holdout-season", required=True, help="Season to reserve from analysis.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ablation"))
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    args = parser.parse_args()

    features = build_pregame_features(pd.read_csv(args.input))
    fold_metrics, predictions = evaluate_feature_ablations(features, args.holdout_season)
    coefficients = extract_coefficients(features, args.holdout_season)
    summary = save_ablation_report(
        fold_metrics,
        predictions,
        coefficients,
        args.holdout_season,
        args.output_dir,
        args.bootstrap_resamples,
    )
    print(summary.to_string(index=False))
    print(f"\nSaved feature ablation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
