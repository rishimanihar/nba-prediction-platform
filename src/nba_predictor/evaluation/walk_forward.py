"""Evaluate NBA home-win models with chronological, season-by-season validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from nba_predictor.features.team_form import build_pregame_features
from nba_predictor.models.feature_sets import (
    BASELINE_FEATURE_COLUMNS,
    ENHANCED_FEATURE_COLUMNS,
    LEAN_LOGISTIC_FEATURE_COLUMNS,
)
from nba_predictor.models.gradient_boosting import build_hist_gradient_boosting_classifier
from nba_predictor.models.logistic import build_logistic_pipeline
from nba_predictor.models.xgboost import build_xgboost_classifier

PROBABILITY_EPSILON = 1e-6
METRIC_COLUMNS = ("accuracy", "roc_auc", "log_loss", "brier_score")
MODEL_ORDER = [
    "always_home",
    "historical_home_rate",
    "elo_only",
    "logistic_baseline",
    "logistic_enhanced",
    "logistic_lean",
    "hist_gradient_boosting",
    "xgboost",
]
LOGISTIC_COMPARISON_BASELINES = ("historical_home_rate", "elo_only", "logistic_baseline")


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def probability_metrics(target: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate classification and probability-quality metrics from one prediction set."""
    clipped_probabilities = _clip_probabilities(probabilities)
    predictions = (clipped_probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, clipped_probabilities)),
        "log_loss": float(log_loss(target, clipped_probabilities)),
        "brier_score": float(brier_score_loss(target, clipped_probabilities)),
    }


def _bootstrap_indices(rng: np.random.Generator, target: np.ndarray, n_resamples: int) -> list[np.ndarray]:
    """Return bootstrap samples that retain both outcome classes for ROC-AUC."""
    samples: list[np.ndarray] = []
    while len(samples) < n_resamples:
        indices = rng.integers(0, len(target), size=len(target))
        if np.unique(target[indices]).size == 2:
            samples.append(indices)
    return samples


def bootstrap_metric_intervals(
    predictions: pd.DataFrame, n_resamples: int = 2_000, random_seed: int = 42
) -> pd.DataFrame:
    """Calculate percentile bootstrap intervals for each model's out-of-fold metrics."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    rng = np.random.default_rng(random_seed)
    interval_rows: list[dict[str, float | int | str]] = []
    for model_name, model_predictions in predictions.groupby("model", sort=False):
        target = model_predictions["home_win"].to_numpy()
        probabilities = model_predictions["home_win_probability"].to_numpy()
        point_estimates = probability_metrics(target, probabilities)
        metric_samples = {metric: [] for metric in METRIC_COLUMNS}

        for indices in _bootstrap_indices(rng, target, n_resamples):
            sampled_metrics = probability_metrics(target[indices], probabilities[indices])
            for metric, value in sampled_metrics.items():
                metric_samples[metric].append(value)

        for metric, point_estimate in point_estimates.items():
            lower, upper = np.quantile(metric_samples[metric], [0.025, 0.975])
            interval_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "point_estimate": point_estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": 0.95,
                    "n_bootstrap_resamples": n_resamples,
                }
            )
    return pd.DataFrame(interval_rows)


def bootstrap_candidate_deltas(
    predictions: pd.DataFrame,
    candidate_model: str,
    baseline_models: tuple[str, ...],
    n_resamples: int = 2_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Calculate paired bootstrap intervals for a candidate model against fixed baselines."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    wide_predictions = predictions.pivot(
        index=["validation_season", "game_id", "home_win"],
        columns="model",
        values="home_win_probability",
    )
    target = wide_predictions.index.get_level_values("home_win").to_numpy()
    requested_models = (candidate_model, *baseline_models)
    if any(model not in wide_predictions.columns for model in requested_models):
        raise ValueError("Predictions are missing one or more comparison baselines.")

    rng = np.random.default_rng(random_seed)
    bootstrap_indices = _bootstrap_indices(rng, target, n_resamples)
    interval_rows: list[dict[str, float | int | str | bool]] = []
    candidate_probabilities = wide_predictions[candidate_model].to_numpy()
    candidate_metrics = probability_metrics(target, candidate_probabilities)
    for baseline_name in baseline_models:
        baseline_probabilities = wide_predictions[baseline_name].to_numpy()
        baseline_metrics = probability_metrics(target, baseline_probabilities)
        delta_samples = {metric: [] for metric in METRIC_COLUMNS}

        for indices in bootstrap_indices:
            candidate_sample = probability_metrics(target[indices], candidate_probabilities[indices])
            baseline_sample = probability_metrics(target[indices], baseline_probabilities[indices])
            for metric in METRIC_COLUMNS:
                delta_samples[metric].append(candidate_sample[metric] - baseline_sample[metric])

        for metric in METRIC_COLUMNS:
            lower, upper = np.quantile(delta_samples[metric], [0.025, 0.975])
            higher_is_better = metric in {"accuracy", "roc_auc"}
            supports_candidate = lower > 0 if higher_is_better else upper < 0
            interval_rows.append(
                {
                    "candidate_model": candidate_model,
                    "baseline_model": baseline_name,
                    "metric": metric,
                    "delta_candidate_minus_baseline": candidate_metrics[metric] - baseline_metrics[metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "higher_is_better": higher_is_better,
                    "ci_supports_candidate": supports_candidate,
                    "confidence_level": 0.95,
                    "n_bootstrap_resamples": n_resamples,
                }
            )
    return pd.DataFrame(interval_rows)


def _season_order(features: pd.DataFrame) -> list[str]:
    return features.groupby("season", sort=False)["game_date"].min().sort_values().index.tolist()


def _model_predictors() -> dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]]:
    def always_home(_: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        return np.ones(len(test_rows))

    def historical_home_rate(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        return np.full(len(test_rows), train_rows["home_win"].mean())

    def elo_only(_: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        return test_rows["elo_home_win_probability"].to_numpy()

    def logistic_baseline(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        model = build_logistic_pipeline()
        model.fit(train_rows[BASELINE_FEATURE_COLUMNS], train_rows["home_win"])
        return model.predict_proba(test_rows[BASELINE_FEATURE_COLUMNS])[:, 1]

    def logistic_enhanced(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        model = build_logistic_pipeline()
        model.fit(train_rows[ENHANCED_FEATURE_COLUMNS], train_rows["home_win"])
        return model.predict_proba(test_rows[ENHANCED_FEATURE_COLUMNS])[:, 1]

    def logistic_lean(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        model = build_logistic_pipeline()
        model.fit(train_rows[LEAN_LOGISTIC_FEATURE_COLUMNS], train_rows["home_win"])
        return model.predict_proba(test_rows[LEAN_LOGISTIC_FEATURE_COLUMNS])[:, 1]

    def hist_gradient_boosting(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        model = build_hist_gradient_boosting_classifier()
        model.fit(train_rows[ENHANCED_FEATURE_COLUMNS], train_rows["home_win"])
        return model.predict_proba(test_rows[ENHANCED_FEATURE_COLUMNS])[:, 1]

    def xgboost(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> np.ndarray:
        model = build_xgboost_classifier()
        model.fit(train_rows[ENHANCED_FEATURE_COLUMNS], train_rows["home_win"])
        return model.predict_proba(test_rows[ENHANCED_FEATURE_COLUMNS])[:, 1]

    return {
        "always_home": always_home,
        "historical_home_rate": historical_home_rate,
        "elo_only": elo_only,
        "logistic_baseline": logistic_baseline,
        "logistic_enhanced": logistic_enhanced,
        "logistic_lean": logistic_lean,
        "hist_gradient_boosting": hist_gradient_boosting,
        "xgboost": xgboost,
    }


def evaluate_walk_forward(features: pd.DataFrame, holdout_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-fold metrics and out-of-fold predictions before the reserved holdout."""
    seasons = _season_order(features)
    if holdout_season not in seasons:
        available = ", ".join(seasons)
        raise ValueError(f"Holdout season {holdout_season!r} not found. Available seasons: {available}")

    holdout_index = seasons.index(holdout_season)
    if holdout_index < 3:
        raise ValueError("At least three seasons before the holdout are required for walk-forward evaluation.")

    predictors = _model_predictors()
    fold_metrics: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for validation_index in range(2, holdout_index):
        validation_season = seasons[validation_index]
        train_seasons = seasons[:validation_index]
        train_rows = features[features["season"].isin(train_seasons)]
        test_rows = features[features["season"] == validation_season]

        for model_name in MODEL_ORDER:
            probabilities = _clip_probabilities(predictors[model_name](train_rows, test_rows))
            metrics = probability_metrics(test_rows["home_win"], probabilities)
            fold_metrics.append(
                {
                    "model": model_name,
                    "validation_season": validation_season,
                    "train_through_season": train_seasons[-1],
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
                        "game_date": test_rows["game_date"].to_numpy(),
                        "home_win": test_rows["home_win"].to_numpy(),
                        "home_win_probability": probabilities,
                    }
                )
            )

    return pd.DataFrame(fold_metrics), pd.concat(prediction_rows, ignore_index=True)


def build_calibration_data(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create ten-bin calibration points from the out-of-fold predictions for each model."""
    calibration_rows: list[dict[str, float | int | str]] = []
    for model_name, model_predictions in predictions.groupby("model", sort=False):
        observed_rate, mean_probability = calibration_curve(
            model_predictions["home_win"],
            model_predictions["home_win_probability"],
            n_bins=10,
            strategy="uniform",
        )
        for bin_index, (predicted, observed) in enumerate(zip(mean_probability, observed_rate, strict=True), start=1):
            calibration_rows.append(
                {
                    "model": model_name,
                    "bin": bin_index,
                    "mean_predicted_probability": predicted,
                    "observed_home_win_rate": observed,
                }
            )
    return pd.DataFrame(calibration_rows)


def save_calibration_plot(calibration_data: pd.DataFrame, output_path: Path) -> None:
    """Save a reliability diagram for model probabilities across validation folds."""
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="perfect calibration")
    for model_name in MODEL_ORDER:
        model_data = calibration_data[calibration_data["model"] == model_name]
        axis.plot(
            model_data["mean_predicted_probability"],
            model_data["observed_home_win_rate"],
            marker="o",
            label=model_name,
        )
    axis.set(
        xlabel="Mean predicted home-win probability",
        ylabel="Observed home-win rate",
        title="Walk-forward calibration by model",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_evaluation_report(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    holdout_season: str,
    output_dir: Path,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    """Persist complete evaluation evidence without touching the final holdout season."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        fold_metrics.groupby("model", sort=False)[["accuracy", "roc_auc", "log_loss", "brier_score"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(column).strip("_") for column in summary.columns.to_flat_index()]
    calibration_data = build_calibration_data(predictions)
    metric_intervals = bootstrap_metric_intervals(predictions, n_resamples=bootstrap_resamples)
    enhanced_deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model="logistic_enhanced",
        baseline_models=LOGISTIC_COMPARISON_BASELINES,
        n_resamples=bootstrap_resamples,
    )
    gradient_boosting_deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model="hist_gradient_boosting",
        baseline_models=("logistic_enhanced",),
        n_resamples=bootstrap_resamples,
    )
    lean_deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model="logistic_lean",
        baseline_models=("logistic_enhanced",),
        n_resamples=bootstrap_resamples,
    )
    xgboost_deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model="xgboost",
        baseline_models=("logistic_enhanced",),
        n_resamples=bootstrap_resamples,
    )

    fold_metrics.to_csv(output_dir / "walk_forward_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "walk_forward_summary.csv", index=False)
    predictions.to_csv(output_dir / "walk_forward_predictions.csv", index=False)
    calibration_data.to_csv(output_dir / "calibration_points.csv", index=False)
    metric_intervals.to_csv(output_dir / "bootstrap_metric_intervals.csv", index=False)
    enhanced_deltas.to_csv(output_dir / "bootstrap_enhanced_deltas.csv", index=False)
    lean_deltas.to_csv(output_dir / "bootstrap_lean_logistic_deltas.csv", index=False)
    gradient_boosting_deltas.to_csv(output_dir / "bootstrap_hist_gradient_boosting_deltas.csv", index=False)
    xgboost_deltas.to_csv(output_dir / "bootstrap_xgboost_deltas.csv", index=False)
    save_calibration_plot(calibration_data, output_dir / "calibration.png")
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "holdout_season": holdout_season,
                "evaluated_validation_seasons": sorted(fold_metrics["validation_season"].unique().tolist()),
                "holdout_evaluated": False,
                "models": MODEL_ORDER,
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
    parser.add_argument("--holdout-season", required=True, help="Season to reserve from all model selection.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    args = parser.parse_args()

    features = build_pregame_features(pd.read_csv(args.input))
    fold_metrics, predictions = evaluate_walk_forward(features, args.holdout_season)
    summary = save_evaluation_report(
        fold_metrics,
        predictions,
        args.holdout_season,
        args.output_dir,
        args.bootstrap_resamples,
    )
    print(summary.to_string(index=False))
    print(f"\nSaved walk-forward evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
