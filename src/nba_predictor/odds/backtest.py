"""Join historical quotes to out-of-sample predictions for a simple value backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba_predictor.odds.value import expected_value_per_unit


def build_backtest(
    quotes: pd.DataFrame, predictions: pd.DataFrame, hours_before_tip: float = 1.0,
    model: str = "lean_logistic",
) -> pd.DataFrame:
    """Use the latest available pre-cutoff quote per game/bookmaker."""
    if hours_before_tip < 0:
        raise ValueError("hours_before_tip must be nonnegative")
    required_quotes = {
        "game_id", "snapshot_at", "market_updated_at", "commence_time", "bookmaker_key",
        "home_team", "away_team", "home_moneyline", "away_moneyline",
    }
    required_predictions = {"game_id", "model", "home_win_probability", "home_win"}
    for name, frame, required in (
        ("Quotes", quotes, required_quotes), ("Predictions", predictions, required_predictions)
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {', '.join(sorted(missing))}")
    selected_predictions = predictions.loc[predictions["model"].eq(model)].copy()
    if selected_predictions["game_id"].astype(str).duplicated().any():
        raise ValueError("Predictions contain duplicate game IDs for the selected model")
    frame = quotes.copy()
    for column in ("snapshot_at", "market_updated_at", "commence_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601", errors="raise")
    cutoff = frame["commence_time"] - pd.to_timedelta(hours_before_tip, unit="h")
    frame = frame.loc[
        frame["snapshot_at"].le(cutoff)
        & frame["market_updated_at"].le(frame["snapshot_at"])
    ].copy()
    # Stable sorting makes ties reproducible and keeps the newest market update.
    frame = frame.sort_values(["snapshot_at", "market_updated_at"], kind="stable")
    frame = frame.drop_duplicates(["game_id", "bookmaker_key"], keep="last")
    frame["game_id"] = frame["game_id"].astype(str)
    selected_predictions["game_id"] = selected_predictions["game_id"].astype(str)
    joined = frame.merge(
        selected_predictions[["game_id", "home_win_probability", "home_win"]],
        on="game_id", how="inner", validate="one_to_one" if frame["bookmaker_key"].nunique() == 1 else "many_to_one",
    )
    if joined.empty:
        return joined.assign(
            home_ev=pd.Series(dtype=float), away_ev=pd.Series(dtype=float),
            bet_side=pd.Series(dtype=str), net_return=pd.Series(dtype=float)
        )
    if not joined["home_win_probability"].between(0, 1).all():
        raise ValueError("Prediction probability outside [0, 1]")
    if {"home_decimal_odds", "away_decimal_odds"} <= set(joined.columns):
        if not joined[["home_decimal_odds", "away_decimal_odds"]].gt(1).all().all():
            raise ValueError("Decimal odds must exceed 1")
        joined["home_ev"] = joined["home_win_probability"] * joined["home_decimal_odds"] - 1
        joined["away_ev"] = (1 - joined["home_win_probability"]) * joined[
            "away_decimal_odds"
        ] - 1
    else:
        joined["home_ev"] = [
            expected_value_per_unit(p, odds) for p, odds in zip(
                joined["home_win_probability"], joined["home_moneyline"], strict=True
            )
        ]
        joined["away_ev"] = [
            expected_value_per_unit(1 - p, odds) for p, odds in zip(
                joined["home_win_probability"], joined["away_moneyline"], strict=True
            )
        ]
    joined["bet_side"] = "none"
    positive = joined[["home_ev", "away_ev"]].max(axis=1).gt(0)
    joined.loc[positive & joined["home_ev"].ge(joined["away_ev"]), "bet_side"] = "home"
    joined.loc[positive & joined["away_ev"].gt(joined["home_ev"]), "bet_side"] = "away"
    joined["net_return"] = 0.0
    home_win = joined["home_win"].astype(int).eq(1)
    home_bet = joined["bet_side"].eq("home")
    away_bet = joined["bet_side"].eq("away")
    joined.loc[home_bet & ~home_win | away_bet & home_win, "net_return"] = -1.0
    for side, winning in (("home", home_bet & home_win), ("away", away_bet & ~home_win)):
        if f"{side}_decimal_odds" in joined:
            joined.loc[winning, "net_return"] = joined.loc[winning, f"{side}_decimal_odds"] - 1
        else:
            joined.loc[winning, "net_return"] = [
                odds / 100 if odds > 0 else 100 / abs(odds)
                for odds in joined.loc[winning, f"{side}_moneyline"]
            ]
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", type=Path, default=Path("data/raw/historical_moneylines.csv"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/odds/value_backtest.csv"))
    parser.add_argument("--hours-before-tip", type=float, default=1.0)
    parser.add_argument("--model", default="lean_logistic")
    args = parser.parse_args()
    result = build_backtest(
        pd.read_csv(args.quotes, dtype={"game_id": str}),
        pd.read_csv(args.predictions, dtype={"game_id": str}),
        args.hours_before_tip, args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    bets = result.loc[result["bet_side"].ne("none")]
    roi = bets["net_return"].mean() if len(bets) else float("nan")
    print(f"Saved {len(result)} priced games; {len(bets)} bets; ROI per unit staked: {roi:.3f}")


if __name__ == "__main__":
    main()
