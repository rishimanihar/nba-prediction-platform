import pandas as pd
import pytest

from nba_predictor.odds.diagnostics import diagnose_backtest


def test_diagnostics_use_fair_market_probability_and_fixed_bet_results() -> None:
    rows = pd.DataFrame([
        {
            "game_id": "1", "home_win": 1, "home_win_probability": .6,
            "home_decimal_odds": 2.0, "away_decimal_odds": 2.0,
            "home_ev": .2, "away_ev": -.2, "bet_side": "home", "net_return": 1.0,
        },
        {
            "game_id": "2", "home_win": 0, "home_win_probability": .7,
            "home_decimal_odds": 1.5, "away_decimal_odds": 2.8,
            "home_ev": .05, "away_ev": -.16, "bet_side": "home", "net_return": -1.0,
        },
    ])
    summary, calibration, value_bins = diagnose_backtest(rows)
    assert summary["games"] == summary["bets"] == 2
    assert summary["winning_bets"] == 1
    assert summary["net_units"] == pytest.approx(0)
    assert summary["model_brier"] == pytest.approx((.4**2 + .7**2) / 2)
    assert calibration["games"].sum() == 2
    assert value_bins["bets"].sum() == 2
