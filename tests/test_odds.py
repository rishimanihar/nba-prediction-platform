import pytest

from nba_predictor.odds.value import (
    american_odds_to_decimal,
    expected_value_per_unit,
    implied_probability,
    remove_two_way_vig,
)


def test_american_moneylines_convert_to_decimal_odds() -> None:
    assert american_odds_to_decimal(150) == pytest.approx(2.5)
    assert american_odds_to_decimal(-200) == pytest.approx(1.5)
    assert implied_probability(150) == pytest.approx(0.4)
    assert implied_probability(-200) == pytest.approx(2.0 / 3.0)


def test_invalid_american_moneyline_is_rejected() -> None:
    with pytest.raises(ValueError):
        american_odds_to_decimal(50)


def test_two_way_vig_removal_produces_probabilities_that_sum_to_one() -> None:
    market = remove_two_way_vig(-120, 110)

    assert market.home_raw_probability + market.away_raw_probability > 1.0
    assert market.home_fair_probability + market.away_fair_probability == pytest.approx(1.0)
    assert market.overround > 0.0


def test_expected_value_uses_the_actual_offered_price() -> None:
    assert expected_value_per_unit(0.60, -110) == pytest.approx(0.1454545)
    assert expected_value_per_unit(0.50, -110) < 0.0
