from pathlib import Path

import pandas as pd
import pytest

from nba_predictor.odds.oddspapi_batch import collect_range


def test_batch_saves_daily_files_and_resumes_without_refetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_ingest(
        day: str, bookmakers: list[str], games_path: Path, output_path: Path, api_key: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append(day)
        matched = pd.DataFrame([{
            "provider_event_id": day, "snapshot_at": f"{day}T18:00:00Z",
            "market_updated_at": f"{day}T18:00:00Z", "commence_time": f"{day}T23:00:00Z",
            "home_team": "BOS", "away_team": "NYK", "bookmaker_key": "fanduel",
            "bookmaker_title": "fanduel", "home_moneyline": -120, "away_moneyline": 110,
            "home_decimal_odds": 1.83, "away_decimal_odds": 2.1,
            "game_id": day, "game_date": day,
        }])
        unmatched = pd.DataFrame(columns=matched.columns)
        matched.to_csv(output_path, index=False)
        unmatched.to_csv(output_path.with_name(output_path.stem + "_unmatched.csv"), index=False)
        return matched, unmatched

    monkeypatch.setattr("nba_predictor.odds.oddspapi_batch.ingest_day", fake_ingest)
    kwargs = {
        "start_date": "2026-01-16", "end_date": "2026-01-17", "bookmakers": ["fanduel"],
        "games_path": tmp_path / "games.csv", "daily_dir": tmp_path / "days",
        "output_path": tmp_path / "combined.csv", "api_key": "key",
    }
    first = collect_range(**kwargs)
    second = collect_range(**kwargs)
    assert calls == ["2026-01-16", "2026-01-17"]
    assert len(first) == len(second) == 2
    assert second["game_id"].tolist() == ["2026-01-16", "2026-01-17"]
    with pytest.raises(ValueError, match="different inputs"):
        collect_range(**(kwargs | {"bookmakers": ["draftkings"]}))


def test_batch_checkpoints_an_empty_fixture_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_ingest(
        day: str, bookmakers: list[str], games_path: Path, output_path: Path, api_key: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append(day)
        matched = pd.DataFrame(columns=[
            "provider_event_id", "snapshot_at", "commence_time", "bookmaker_key", "game_id",
        ])
        unmatched = matched.copy()
        matched.to_csv(output_path, index=False)
        unmatched.to_csv(output_path.with_name(output_path.stem + "_unmatched.csv"), index=False)
        return matched, unmatched

    monkeypatch.setattr("nba_predictor.odds.oddspapi_batch.ingest_day", fake_ingest)
    kwargs = {
        "start_date": "2026-02-14", "end_date": "2026-02-14", "bookmakers": ["fanduel"],
        "games_path": tmp_path / "games.csv", "daily_dir": tmp_path / "days",
        "output_path": tmp_path / "combined.csv", "api_key": "key",
    }

    assert collect_range(**kwargs).empty
    assert collect_range(**kwargs).empty
    assert calls == ["2026-02-14"]
    assert (tmp_path / "days" / "2026-02-14.json").exists()
