import io
import json
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from nba_predictor.odds.backtest import build_backtest
from nba_predictor.odds.historical import fetch_snapshot, match_quotes_to_games, normalize_snapshot


def _snapshot() -> dict:
    return {
        "timestamp": "2024-10-22T20:00:00Z",
        "data": [{
            "id": "event-1", "sport_key": "basketball_nba",
            "commence_time": "2024-10-23T00:00:00Z",
            "home_team": "Boston Celtics", "away_team": "New York Knicks",
            "bookmakers": [{
                "key": "book-a", "title": "Book A",
                "markets": [{
                    "key": "h2h", "last_update": "2024-10-22T19:55:00Z",
                    "outcomes": [
                        {"name": "New York Knicks", "price": 110},
                        {"name": "Boston Celtics", "price": -120},
                    ],
                }],
            }],
        }],
    }


def test_fetch_requests_historical_nba_h2h_at_explicit_time(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        assert timeout == 30
        url = urlsplit(request.full_url)
        query = parse_qs(url.query)
        assert url.path.endswith("/historical/sports/basketball_nba/odds")
        assert query["markets"] == ["h2h"]
        assert query["oddsFormat"] == ["american"]
        assert query["date"] == ["2024-10-22T20:00:00Z"]
        return io.BytesIO(json.dumps(_snapshot()).encode())

    monkeypatch.setattr("nba_predictor.odds.historical.urlopen", fake_urlopen)
    assert fetch_snapshot("2024-10-22T20:00:00Z", "fake-key")["timestamp"] == (
        "2024-10-22T20:00:00Z"
    )


def test_snapshot_keeps_provenance_and_matches_eastern_game_date() -> None:
    quotes = normalize_snapshot(_snapshot())
    games = pd.DataFrame([{
        "game_id": "0022400061", "game_date": "2024-10-22",
        "home_team": "BOS", "away_team": "NYK",
    }])
    matched, unmatched = match_quotes_to_games(quotes, games)

    assert len(matched) == 1
    assert unmatched.empty
    assert matched.iloc[0]["game_id"] == "0022400061"
    assert matched.iloc[0]["home_moneyline"] == -120
    assert matched.iloc[0]["snapshot_at"] == "2024-10-22T20:00:00Z"


def test_live_or_incomplete_quotes_are_excluded() -> None:
    payload = _snapshot()
    payload["timestamp"] = "2024-10-23T00:01:00Z"
    assert normalize_snapshot(payload).empty
    payload["timestamp"] = "2024-10-22T20:00:00Z"
    payload["data"][0]["bookmakers"][0]["markets"][0]["outcomes"].pop()
    assert normalize_snapshot(payload).empty


def test_ambiguous_game_match_is_left_unmatched() -> None:
    quotes = normalize_snapshot(_snapshot())
    games = pd.DataFrame([{
        "game_id": game_id, "game_date": "2024-10-22",
        "home_team": "BOS", "away_team": "NYK",
    } for game_id in ("a", "b")])
    matched, unmatched = match_quotes_to_games(quotes, games)
    assert matched.empty
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["match_status"] == "ambiguous_game"


def test_backtest_uses_latest_eligible_snapshot_once() -> None:
    quotes = normalize_snapshot(_snapshot()).assign(game_id="0022400061")
    newer = quotes.copy()
    newer["snapshot_at"] = "2024-10-22T23:30:00.123Z"
    newer["market_updated_at"] = "2024-10-22T23:25:00.123Z"
    newer["home_moneyline"] = 150
    predictions = pd.DataFrame([{
        "game_id": "0022400061", "model": "lean_logistic",
        "home_win_probability": 0.6, "home_win": 1,
    }])
    result = build_backtest(pd.concat([quotes, newer]), predictions, hours_before_tip=1)
    assert len(result) == 1
    assert result.iloc[0]["home_moneyline"] == -120
    assert result.iloc[0]["bet_side"] == "home"
    assert result.iloc[0]["net_return"] == pytest.approx(100 / 120)
