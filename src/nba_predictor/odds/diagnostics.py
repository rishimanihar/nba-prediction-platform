"""Report calibration and selected-bet outcomes without tuning on test games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def diagnose_backtest(backtest: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Summarize one bookmaker's fixed-rule backtest with prespecified bins."""
    required = {
        "game_id", "home_win", "home_win_probability", "home_decimal_odds",
        "away_decimal_odds", "home_ev", "away_ev", "bet_side", "net_return",
    }
    missing = required - set(backtest.columns)
    if missing:
        raise ValueError(f"Backtest missing columns: {', '.join(sorted(missing))}")
    if backtest["game_id"].astype(str).duplicated().any():
        raise ValueError("Diagnostics require one selected bookmaker quote per game")
    if backtest.empty:
        raise ValueError("Backtest is empty")
    frame = backtest.copy()
    outcomes = frame["home_win"].to_numpy(dtype=int)
    model_probability = frame["home_win_probability"].to_numpy(dtype=float)
    home_implied = 1 / frame["home_decimal_odds"].to_numpy(dtype=float)
    away_implied = 1 / frame["away_decimal_odds"].to_numpy(dtype=float)
    market_probability = home_implied / (home_implied + away_implied)
    if not np.isin(outcomes, [0, 1]).all() or not np.isfinite(model_probability).all():
        raise ValueError("Invalid outcomes or probabilities")
    if not np.logical_and(model_probability >= 0, model_probability <= 1).all():
        raise ValueError("Model probability outside [0, 1]")
    if not np.logical_and(home_implied > 0, away_implied > 0).all():
        raise ValueError("Invalid decimal odds")

    def brier(probability: np.ndarray) -> float:
        return float(np.mean((probability - outcomes) ** 2))

    def log_loss(probability: np.ndarray) -> float:
        clipped = np.clip(probability, 1e-9, 1 - 1e-9)
        return float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped)))

    bets = frame.loc[frame["bet_side"].ne("none")].copy()
    bets["selected_ev"] = np.where(
        bets["bet_side"].eq("home"), bets["home_ev"], bets["away_ev"]
    )
    if not bets["selected_ev"].gt(0).all():
        raise ValueError("Selected bets must have positive model-estimated value")
    wins = (bets["net_return"] > 0).sum()
    summary: dict[str, float | int] = {
        "games": len(frame), "bets": len(bets), "winning_bets": int(wins),
        "net_units": float(bets["net_return"].sum()),
        "roi_per_unit": float(bets["net_return"].mean()) if len(bets) else float("nan"),
        "mean_selected_estimated_ev": float(bets["selected_ev"].mean()) if len(bets) else float("nan"),
        "model_brier": brier(model_probability), "market_brier": brier(market_probability),
        "model_log_loss": log_loss(model_probability),
        "market_log_loss": log_loss(market_probability),
    }
    frame["probability_bin"] = pd.cut(
        frame["home_win_probability"], bins=[0, .2, .4, .6, .8, 1], include_lowest=True
    )
    frame["market_home_probability"] = market_probability
    calibration = frame.groupby("probability_bin", observed=False).agg(
        games=("game_id", "size"), mean_model_probability=("home_win_probability", "mean"),
        mean_market_probability=("market_home_probability", "mean"),
        observed_home_win_rate=("home_win", "mean"),
    ).reset_index()
    calibration["probability_bin"] = calibration["probability_bin"].astype(str)
    bets["estimated_ev_bin"] = pd.cut(
        bets["selected_ev"], bins=[0, .05, .10, .20, float("inf")], include_lowest=True
    )
    value_bins = bets.groupby("estimated_ev_bin", observed=False).agg(
        bets=("game_id", "size"), mean_estimated_ev=("selected_ev", "mean"),
        realized_roi=("net_return", "mean"), net_units=("net_return", "sum"),
    ).reset_index()
    value_bins["estimated_ev_bin"] = value_bins["estimated_ev_bin"].astype(str)
    return summary, calibration, value_bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/odds/diagnostics"))
    args = parser.parse_args()
    summary, calibration, value_bins = diagnose_backtest(
        pd.read_csv(args.backtest, dtype={"game_id": str})
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    calibration.to_csv(args.output_dir / "calibration.csv", index=False)
    value_bins.to_csv(args.output_dir / "estimated_value_bins.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
