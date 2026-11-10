"""Compare frozen-model outcomes with a pregame bookmaker on the same games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def _metrics(outcomes: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(outcomes, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(outcomes, probabilities)),
        "brier_score": float(brier_score_loss(outcomes, probabilities)),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
    }


def paired_market_advantage(
    outcomes: np.ndarray,
    model_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    game_dates: np.ndarray,
    n_resamples: int = 2_000,
    seed: int = 42,
) -> dict:
    """Estimate 95% intervals for market advantage, resampling whole game dates."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    dates = np.asarray(game_dates)
    if len(dates) != len(outcomes) or pd.isna(dates).any():
        raise ValueError("Each priced game needs a game date")
    groups = [np.flatnonzero(dates == date) for date in np.unique(dates)]
    model_metrics = _metrics(outcomes, model_probabilities)
    market_metrics = _metrics(outcomes, market_probabilities)
    higher_is_better = ("accuracy", "roc_auc")

    def advantage(model: dict[str, float], market: dict[str, float]) -> dict[str, float]:
        return {
            metric: (market[metric] - model[metric])
            if metric in higher_is_better
            else (model[metric] - market[metric])
            for metric in model
        }

    point = advantage(model_metrics, market_metrics)
    samples = {metric: [] for metric in point}
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(samples["accuracy"]) < n_resamples:
        attempts += 1
        if attempts > 20 * n_resamples:
            raise ValueError("Too few bootstrap samples contain both outcome classes")
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        sampled_outcomes = outcomes[indices]
        if np.unique(sampled_outcomes).size < 2:
            continue
        draw = advantage(
            _metrics(sampled_outcomes, model_probabilities[indices]),
            _metrics(sampled_outcomes, market_probabilities[indices]),
        )
        for metric, value in draw.items():
            samples[metric].append(value)
    return {
        "method": "paired game-date cluster bootstrap, percentile 95% interval",
        "n_game_dates": len(groups),
        "n_resamples": n_resamples,
        "seed": seed,
        "positive_means": "market performed better",
        "metrics": {
            metric: {
                "estimate": point[metric],
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
            }
            for metric, values in samples.items()
        },
    }


def compare_outcomes(
    games: pd.DataFrame,
    predictions: pd.DataFrame,
    backtest: pd.DataFrame,
    metadata: dict,
    hours_before_tip: float = 1,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 42,
) -> dict:
    """Score the full target season, then compare model and market on matched games."""
    if hours_before_tip < 0:
        raise ValueError("hours_before_tip must be nonnegative")
    required = {
        "games": {
            "game_id",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        },
        "predictions": {
            "game_id",
            "game_date",
            "target_season",
            "home_team",
            "away_team",
            "home_win",
            "home_win_probability",
            "model",
        },
        "backtest": {
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_win",
            "home_win_probability",
            "home_decimal_odds",
            "away_decimal_odds",
            "snapshot_at",
            "market_updated_at",
            "commence_time",
            "bookmaker_key",
        },
    }
    for name, frame in (("games", games), ("predictions", predictions), ("backtest", backtest)):
        missing = required[name] - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {', '.join(sorted(missing))}")
        if frame.empty or frame["game_id"].isna().any():
            raise ValueError(f"{name} must have game IDs and at least one row")
        if frame["game_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} has duplicate game IDs")

    season = games["season"].astype(str).unique()
    if (
        len(season) != 1
        or predictions["target_season"].astype(str).unique().tolist() != season.tolist()
    ):
        raise ValueError("Games and predictions must cover the same single target season")
    if season[0] in metadata["artifact_trained_seasons"]:
        raise ValueError("Target season was included in model training")
    if predictions["model"].nunique() != 1 or backtest["bookmaker_key"].nunique() != 1:
        raise ValueError("Use one model and one bookmaker per comparison")
    if set(games["game_id"].astype(str)) != set(predictions["game_id"].astype(str)):
        raise ValueError("Predictions must cover exactly the supplied games")

    source = games.copy()
    source["game_id"] = source["game_id"].astype(str)
    predicted = predictions.copy()
    predicted["game_id"] = predicted["game_id"].astype(str)
    matched = predicted.merge(
        source, on="game_id", suffixes=("_pred", "_game"), validate="one_to_one"
    )
    for field in ("game_date", "home_team", "away_team"):
        if not matched[f"{field}_pred"].astype(str).eq(matched[f"{field}_game"].astype(str)).all():
            raise ValueError(f"Prediction {field} does not match game data")
    home_score = pd.to_numeric(matched["home_score"], errors="coerce")
    away_score = pd.to_numeric(matched["away_score"], errors="coerce")
    if home_score.isna().any() or away_score.isna().any() or home_score.eq(away_score).any():
        raise ValueError("Games require completed, non-tied scores")
    outcomes = pd.to_numeric(matched["home_win"], errors="coerce").to_numpy()
    if not np.isin(outcomes, [0, 1]).all() or not np.array_equal(
        outcomes, (home_score > away_score).to_numpy()
    ):
        raise ValueError("Prediction outcomes do not match game scores")
    model_probability = pd.to_numeric(matched["home_win_probability"], errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(model_probability).all()
        or not np.logical_and(0 <= model_probability, model_probability <= 1).all()
    ):
        raise ValueError("Model probabilities must be in [0, 1]")

    quoted = backtest.copy()
    quoted["game_id"] = quoted["game_id"].astype(str)
    if not set(quoted["game_id"]).issubset(set(predicted["game_id"])):
        raise ValueError("Backtest contains games outside the predicted season")
    for field in ("snapshot_at", "market_updated_at", "commence_time"):
        quoted[field] = pd.to_datetime(quoted[field], utc=True, format="ISO8601", errors="coerce")
    if quoted[["snapshot_at", "market_updated_at", "commence_time"]].isna().any().any():
        raise ValueError("Backtest quote timestamps are missing or invalid")
    cutoff = quoted["commence_time"] - pd.Timedelta(hours=hours_before_tip)
    if (quoted["snapshot_at"] > cutoff).any() or (
        quoted["market_updated_at"] > quoted["snapshot_at"]
    ).any():
        raise ValueError("Backtest contains a quote unavailable at the pregame cutoff")
    priced = quoted.merge(
        predicted, on="game_id", suffixes=("_quote", "_pred"), validate="one_to_one"
    )
    for field in ("game_date", "home_team", "away_team", "home_win"):
        if not priced[f"{field}_quote"].astype(str).eq(priced[f"{field}_pred"].astype(str)).all():
            raise ValueError(f"Backtest {field} does not match predictions")
    quote_probability = pd.to_numeric(
        priced["home_win_probability_quote"], errors="coerce"
    ).to_numpy(dtype=float)
    prediction_probability = pd.to_numeric(
        priced["home_win_probability_pred"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.allclose(quote_probability, prediction_probability, rtol=0, atol=1e-12):
        raise ValueError("Backtest model probabilities do not match saved predictions")
    home_odds = pd.to_numeric(priced["home_decimal_odds"], errors="coerce").to_numpy(dtype=float)
    away_odds = pd.to_numeric(priced["away_decimal_odds"], errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(home_odds).all()
        or not np.isfinite(away_odds).all()
        or (home_odds <= 1).any()
        or (away_odds <= 1).any()
    ):
        raise ValueError("Backtest requires valid two-way decimal odds")
    home_implied, away_implied = 1 / home_odds, 1 / away_odds
    market_probability = home_implied / (home_implied + away_implied)
    priced_outcomes = priced["home_win_pred"].to_numpy(dtype=int)
    market_advantage = paired_market_advantage(
        priced_outcomes,
        prediction_probability,
        market_probability,
        priced["game_date_pred"].to_numpy(),
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )

    return {
        "target_season": season[0],
        "model": str(predicted["model"].iloc[0]),
        "trained_seasons": metadata["artifact_trained_seasons"],
        "full_season": {
            "games": len(matched),
            "home_win_rate": float(np.mean(outcomes)),
            "model": _metrics(outcomes, model_probability),
        },
        "priced_games": {
            "games": len(priced),
            "bookmaker": str(priced["bookmaker_key"].iloc[0]),
            "hours_before_tip": hours_before_tip,
            "first_game_date": str(priced["game_date_pred"].min()),
            "last_game_date": str(priced["game_date_pred"].max()),
            "model": _metrics(priced_outcomes, prediction_probability),
            "market_no_vig": _metrics(priced_outcomes, market_probability),
            "paired_market_advantage": market_advantage,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--backtest", type=Path, required=True)
    parser.add_argument(
        "--metadata", type=Path, default=Path("artifacts/metrics_lean_logistic.json")
    )
    parser.add_argument("--hours-before-tip", type=float, default=1)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/odds/outcome_comparison_2025_26.json")
    )
    args = parser.parse_args()
    report = compare_outcomes(
        pd.read_csv(args.games, dtype={"game_id": str}),
        pd.read_csv(args.predictions, dtype={"game_id": str}),
        pd.read_csv(args.backtest, dtype={"game_id": str}),
        json.loads(args.metadata.read_text(encoding="utf-8")),
        args.hours_before_tip,
        args.bootstrap_resamples,
        args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
