"""Convert two-way moneylines into market probabilities and estimated value."""

from __future__ import annotations

from dataclasses import dataclass


def american_odds_to_decimal(american_odds: float) -> float:
    """Convert a valid American moneyline to decimal odds."""
    if -100.0 < american_odds < 100.0:
        raise ValueError("American odds must be at most -100 or at least +100.")
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    return 1.0 + 100.0 / abs(american_odds)


def implied_probability(american_odds: float) -> float:
    """Return the break-even probability implied by an offered moneyline."""
    return 1.0 / american_odds_to_decimal(american_odds)


@dataclass(frozen=True)
class TwoWayMarket:
    home_raw_probability: float
    away_raw_probability: float
    home_fair_probability: float
    away_fair_probability: float
    overround: float


def remove_two_way_vig(home_moneyline: float, away_moneyline: float) -> TwoWayMarket:
    """Remove a two-way market's vig by proportionally normalizing implied probabilities."""
    home_raw = implied_probability(home_moneyline)
    away_raw = implied_probability(away_moneyline)
    probability_sum = home_raw + away_raw
    return TwoWayMarket(
        home_raw_probability=home_raw,
        away_raw_probability=away_raw,
        home_fair_probability=home_raw / probability_sum,
        away_fair_probability=away_raw / probability_sum,
        overround=probability_sum - 1.0,
    )


def expected_value_per_unit(model_probability: float, american_odds: float) -> float:
    """Estimate net profit per unit staked using the model probability."""
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("Model probability must be between 0 and 1.")
    decimal_odds = american_odds_to_decimal(american_odds)
    return model_probability * (decimal_odds - 1.0) - (1.0 - model_probability)
