"""Upcoming predictions use only completed games before the scoring date."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_predictor.models.feature_sets import LEAN_LOGISTIC_FEATURE_COLUMNS
from nba_predictor.predict_upcoming import generate_predictions, write_snapshot


class FixedModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        assert list(features.columns) == LEAN_LOGISTIC_FEATURE_COLUMNS
        return np.tile([0.3, 0.7], (len(features), 1))


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    history = pd.DataFrame(
        [["0022400001", "2024-10-21", "2024-25", "BOS", "NYK", 110, 100]],
        columns=[
            "game_id",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ],
    )
    schedule = pd.DataFrame(
        [
            ["0022500001", "2025-10-20", 3, "BOS", "MIA", 115, 105],
            ["0022500002", "2025-10-21", 1, "BOS", "NYK", None, None],
            ["0022500003", "2025-10-22", 3, "BOS", "CHI", 90, 120],
        ],
        columns=[
            "gameId",
            "gameDate",
            "gameStatus",
            "homeTeam_teamTricode",
            "awayTeam_teamTricode",
            "homeTeam_score",
            "awayTeam_score",
        ],
    )
    metadata = {
        "model": "test_lean_logistic",
        "trained_at": "2025-09-01T00:00:00+00:00",
        "feature_columns": LEAN_LOGISTIC_FEATURE_COLUMNS,
        "artifact_trained_seasons": ["2024-25"],
    }
    return history, schedule, metadata


def test_future_results_cannot_change_scheduled_prediction() -> None:
    history, schedule, metadata = _inputs()
    when = datetime(2025, 10, 21, tzinfo=UTC)
    snapshot = generate_predictions(
        history, schedule, "2025-26", date(2025, 10, 21), FixedModel(), metadata, when
    )
    without_future = generate_predictions(
        history, schedule.iloc[:2], "2025-26", date(2025, 10, 21), FixedModel(), metadata, when
    )
    assert snapshot == without_future
    assert snapshot["games"] == [
        {
            "game_id": "0022500002",
            "game_date": "2025-10-21",
            "season": "2025-26",
            "home_team": "BOS",
            "away_team": "NYK",
            "home_win_probability": 0.7,
            "away_win_probability": pytest.approx(0.3),
            "predicted_winner": "BOS",
        }
    ]


def test_rejects_history_from_prediction_date() -> None:
    history, schedule, metadata = _inputs()
    history.loc[0, "game_date"] = "2025-10-21"
    with pytest.raises(ValueError, match="precede"):
        generate_predictions(
            history, schedule, "2025-26", date(2025, 10, 21), FixedModel(), metadata
        )


def test_no_scheduled_games_produces_empty_snapshot() -> None:
    history, schedule, metadata = _inputs()
    snapshot = generate_predictions(
        history, schedule, "2025-26", date(2025, 10, 23), FixedModel(), metadata
    )
    assert snapshot["games"] == []


def test_empty_schedule_is_an_error_instead_of_an_off_day() -> None:
    history, schedule, metadata = _inputs()
    with pytest.raises(ValueError, match="schedule is empty"):
        generate_predictions(
            history, schedule.iloc[:0], "2025-26", date(2025, 10, 21), FixedModel(), metadata
        )


def test_snapshot_write_preserves_previous_file_when_serialization_fails(tmp_path: Path) -> None:
    output = tmp_path / "upcoming.json"
    write_snapshot({"games": ["first"]}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"games": ["first"]}
    with pytest.raises(TypeError):
        write_snapshot({"games": {"not JSON serializable"}}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"games": ["first"]}
    assert list(tmp_path.glob("*.tmp")) == []
