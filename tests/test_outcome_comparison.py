"""Checks for season coverage and fair same-game market comparisons."""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.evaluation.outcome_comparison import (
    compare_outcomes,
    paired_market_advantage,
)


def _inputs():
    games = pd.DataFrame(
        [
            {
                "game_id": "01",
                "game_date": "2025-10-21",
                "season": "2025-26",
                "home_team": "A",
                "away_team": "B",
                "home_score": 110,
                "away_score": 100,
            },
            {
                "game_id": "02",
                "game_date": "2025-10-22",
                "season": "2025-26",
                "home_team": "C",
                "away_team": "D",
                "home_score": 90,
                "away_score": 100,
            },
            {
                "game_id": "03",
                "game_date": "2025-10-23",
                "season": "2025-26",
                "home_team": "E",
                "away_team": "F",
                "home_score": 105,
                "away_score": 99,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "game_id": row.game_id,
                "game_date": row.game_date,
                "target_season": "2025-26",
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_win": int(row.home_score > row.away_score),
                "home_win_probability": probability,
                "model": "lean_logistic",
            }
            for row, probability in zip(games.itertuples(), [0.8, 0.3, 0.6], strict=True)
        ]
    )
    backtest = predictions.iloc[:2][
        ["game_id", "game_date", "home_team", "away_team", "home_win", "home_win_probability"]
    ].copy()
    backtest["home_decimal_odds"] = [1.5, 2.0]
    backtest["away_decimal_odds"] = [3.0, 2.0]
    backtest["snapshot_at"] = [
        "2025-10-21 17:00:00.493000+00:00",
        "2025-10-22 17:00:00+00:00",
    ]
    backtest["market_updated_at"] = backtest["snapshot_at"]
    backtest["commence_time"] = ["2025-10-21T20:00:00Z", "2025-10-22T20:00:00Z"]
    backtest["bookmaker_key"] = "fanduel"
    return games, predictions, backtest, {"artifact_trained_seasons": ["2024-25"]}


def test_compares_full_season_and_same_priced_games() -> None:
    report = compare_outcomes(*_inputs(), bootstrap_resamples=20)
    assert report["full_season"]["games"] == 3
    assert report["priced_games"]["games"] == 2
    assert report["full_season"]["model"]["accuracy"] == pytest.approx(1)
    assert report["priced_games"]["market_no_vig"]["accuracy"] == pytest.approx(0.5)
    assert report["priced_games"]["model"]["brier_score"] == pytest.approx((0.2**2 + 0.3**2) / 2)
    advantage = report["priced_games"]["paired_market_advantage"]
    assert advantage["n_game_dates"] == 2
    assert advantage["metrics"]["brier_score"]["estimate"] == pytest.approx(
        report["priced_games"]["model"]["brier_score"]
        - report["priced_games"]["market_no_vig"]["brier_score"]
    )


def test_paired_intervals_are_reproducible_and_zero_for_identical_predictions() -> None:
    outcomes = np.array([1, 0, 1, 0])
    probabilities = np.array([0.8, 0.3, 0.6, 0.4])
    dates = np.array(["day1", "day1", "day2", "day2"])
    first = paired_market_advantage(outcomes, probabilities, probabilities, dates, 30, 7)
    second = paired_market_advantage(outcomes, probabilities, probabilities, dates, 30, 7)
    assert first == second
    assert first["n_game_dates"] == 2
    for metric in first["metrics"].values():
        assert metric == {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_rejects_mismatched_result_or_late_quote() -> None:
    games, predictions, backtest, metadata = _inputs()
    predictions.loc[0, "home_win"] = 0
    with pytest.raises(ValueError, match="outcomes"):
        compare_outcomes(games, predictions, backtest, metadata)
    predictions.loc[0, "home_win"] = 1
    backtest.loc[0, "snapshot_at"] = "2025-10-21T19:30:00Z"
    with pytest.raises(ValueError, match="cutoff"):
        compare_outcomes(games, predictions, backtest, metadata)


def test_rejects_training_on_target_season() -> None:
    games, predictions, backtest, metadata = _inputs()
    metadata["artifact_trained_seasons"].append("2025-26")
    with pytest.raises(ValueError, match="included in model training"):
        compare_outcomes(games, predictions, backtest, metadata)
