import io
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd
import pytest

from nba_predictor.odds.backtest import build_backtest
from nba_predictor.odds.oddspapi import (
    OddsPapiRequestError,
    ingest_day,
    normalize_fixture_history,
    request_json,
)


def test_http_error_reports_provider_code_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(request: object, timeout: int) -> None:
        assert "User-agent" in request.headers
        raise HTTPError(
            request.full_url, 403, "Forbidden", {},
            io.BytesIO(b'{"error":{"code":"RESTRICTED_ACCESS"}}'),
        )

    monkeypatch.setattr("nba_predictor.odds.oddspapi.urlopen", forbidden)
    with pytest.raises(RuntimeError, match=r"HTTP 403 \(RESTRICTED_ACCESS\)") as error:
        request_json("account", {"apiKey": "private-key"})
    assert "private-key" not in str(error.value)


def test_rate_limit_retries_after_provider_delay_without_exposing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    delays: list[float] = []

    def rate_limited_once(request: object, timeout: int) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url, 429, "Too Many Requests", {},
                io.BytesIO(b'{"error":{"code":"RATE_LIMITED","retryMs":420}}'),
            )
        return io.BytesIO(b'{"status":"ok"}')

    monkeypatch.setattr("nba_predictor.odds.oddspapi.urlopen", rate_limited_once)
    monkeypatch.setattr("nba_predictor.odds.oddspapi.time.sleep", delays.append)

    assert request_json("fixtures", {"apiKey": "private-key"}) == {"status": "ok"}
    assert calls == 2
    assert delays == pytest.approx([0.62])


def test_quota_exhaustion_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def quota_exceeded(request: object, timeout: int) -> None:
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url, 429, "Too Many Requests", {},
            io.BytesIO(b'{"error":{"code":"REQUEST_LIMIT_EXCEEDED"}}'),
        )

    monkeypatch.setattr("nba_predictor.odds.oddspapi.urlopen", quota_exceeded)
    with pytest.raises(OddsPapiRequestError, match="REQUEST_LIMIT_EXCEEDED"):
        request_json("fixtures", {"apiKey": "private-key"})
    assert calls == 1


def test_history_pairs_active_home_and_away_at_each_line_move() -> None:
    fixture = {
        "fixtureId": "fixture-1", "tournamentSlug": "nba",
        "participant1Name": "Boston Celtics", "participant2Name": "New York Knicks",
        "startTime": "2026-02-02T00:00:00Z",
    }
    history = {
        "fixtureId": "fixture-1", "bookmakers": {"fanduel": {"markets": {"111": {"outcomes": {
            "111": {"players": {"0": [
                {"createdAt": "2026-02-01T20:00:00Z", "price": 1.8, "active": True},
                {"createdAt": "2026-02-01T22:00:00Z", "price": 1.9, "active": True},
            ]}},
            "112": {"players": {"0": [
                {"createdAt": "2026-02-01T20:05:00Z", "price": 2.1, "active": True},
                {"createdAt": "2026-02-01T23:00:00Z", "price": 2.0, "active": False},
            ]}},
        }}}}},
    }
    quotes = normalize_fixture_history(fixture, history)
    assert len(quotes) == 2
    assert quotes.iloc[0]["snapshot_at"] == "2026-02-01T20:05:00Z"
    assert quotes.iloc[1]["home_decimal_odds"] == pytest.approx(1.9)
    assert quotes.iloc[1]["away_decimal_odds"] == pytest.approx(2.1)

    predictions = pd.DataFrame([{
        "game_id": "0022500001", "model": "lean_logistic",
        "home_win_probability": 0.6, "home_win": 1,
    }])
    quotes["game_id"] = "0022500001"
    result = build_backtest(quotes, predictions, hours_before_tip=1)
    assert len(result) == 1
    assert result.iloc[0]["home_ev"] == pytest.approx(0.14)
    assert result.iloc[0]["net_return"] == pytest.approx(0.9)


def test_ingest_day_fetches_history_when_completed_fixture_has_odds_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = {
        "fixtureId": "fixture-2", "tournamentId": 132, "hasOdds": False,
        "participant1Name": "Boston Celtics", "participant2Name": "New York Knicks",
        "startTime": "2026-01-16T00:00:00Z",
    }
    history = {
        "fixtureId": "fixture-2", "bookmakers": {"fanduel": {"markets": {"111": {"outcomes": {
            "111": {"players": {"0": [{
                "createdAt": "2026-01-15T20:00:00Z", "price": 1.8, "active": True,
            }]}},
            "112": {"players": {"0": [{
                "createdAt": "2026-01-15T20:01:00Z", "price": 2.1, "active": True,
            }]}},
        }}}}},
    }
    calls = []

    def fake_request(endpoint: str, params: dict) -> object:
        calls.append(endpoint)
        return [fixture] if endpoint == "fixtures" else history

    monkeypatch.setattr("nba_predictor.odds.oddspapi.request_json", fake_request)
    monkeypatch.setattr("nba_predictor.odds.oddspapi.time.sleep", lambda seconds: None)
    games_path = tmp_path / "games.csv"
    output_path = tmp_path / "odds.csv"
    pd.DataFrame([{
        "game_id": "0022500001", "game_date": "2026-01-15",
        "home_team": "BOS", "away_team": "NYK",
    }]).to_csv(games_path, index=False)
    matched, unmatched = ingest_day("2026-01-16", ["fanduel"], games_path, output_path, "key")
    assert calls == ["fixtures", "historical-odds"]
    assert len(matched) == 1 and unmatched.empty


def test_ingest_day_records_missing_history_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = [
        {
            "fixtureId": fixture_id, "tournamentId": 132,
            "participant1Name": "Boston Celtics", "participant2Name": "New York Knicks",
            "startTime": "2026-02-11T23:00:00Z",
        }
        for fixture_id in ("missing", "available")
    ]
    requested: list[str] = []

    def fake_request(endpoint: str, params: dict) -> object:
        if endpoint == "fixtures":
            return fixtures
        requested.append(params["fixtureId"])
        if params["fixtureId"] == "missing":
            raise OddsPapiRequestError(endpoint, 404, "NOT_FOUND")
        return {"fixtureId": "available", "bookmakers": {}}

    monkeypatch.setattr("nba_predictor.odds.oddspapi.request_json", fake_request)
    monkeypatch.setattr("nba_predictor.odds.oddspapi.time.sleep", lambda seconds: None)
    games_path = tmp_path / "games.csv"
    output_path = tmp_path / "odds.csv"
    pd.DataFrame([{
        "game_id": "0022500001", "game_date": "2026-02-11",
        "home_team": "BOS", "away_team": "NYK",
    }]).to_csv(games_path, index=False)
    matched, unmatched = ingest_day("2026-02-11", ["fanduel"], games_path, output_path, "key")
    missing = pd.read_csv(tmp_path / "odds_missing_history.csv")
    assert requested == ["missing", "available"]
    assert matched.empty and unmatched.empty
    assert missing["fixture_id"].tolist() == ["missing"]
    assert missing["provider_code"].tolist() == ["NOT_FOUND"]


def test_ingest_day_saves_empty_checkpoint_when_no_fixtures_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(endpoint: str, params: dict) -> object:
        assert endpoint == "fixtures"
        raise OddsPapiRequestError(endpoint, 404, "FIXTURE_NOT_FOUND")

    monkeypatch.setattr("nba_predictor.odds.oddspapi.request_json", fake_request)
    monkeypatch.setattr("nba_predictor.odds.oddspapi.time.sleep", lambda seconds: None)
    games_path = tmp_path / "games.csv"
    output_path = tmp_path / "odds.csv"
    pd.DataFrame([{
        "game_id": "0022500001", "game_date": "2026-02-14",
        "home_team": "BOS", "away_team": "NYK",
    }]).to_csv(games_path, index=False)

    matched, unmatched = ingest_day("2026-02-14", ["fanduel"], games_path, output_path, "key")

    assert matched.empty and unmatched.empty
    assert pd.read_csv(output_path).empty
    assert pd.read_csv(tmp_path / "odds_unmatched.csv").empty
    assert pd.read_csv(tmp_path / "odds_missing_history.csv").empty


def test_ingest_day_preserves_other_fixture_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_request(endpoint: str, params: dict) -> object:
        raise OddsPapiRequestError(endpoint, 404, "NOT_FOUND")

    monkeypatch.setattr("nba_predictor.odds.oddspapi.request_json", fake_request)
    with pytest.raises(OddsPapiRequestError, match=r"fixtures returned HTTP 404 \(NOT_FOUND\)"):
        ingest_day("2026-02-14", ["fanduel"], tmp_path / "games.csv", tmp_path / "odds.csv", "key")
