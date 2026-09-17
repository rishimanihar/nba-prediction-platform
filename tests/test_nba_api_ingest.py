import pandas as pd
import pytest

from nba_predictor.data.nba_api_ingest import (
    normalize_schedule_games,
    season_label,
    validate_season_games,
)


def test_season_label() -> None:
    assert season_label(2018) == "2018-19"


def test_normalize_schedule_games_keeps_neutral_site_game() -> None:
    schedule_games = pd.DataFrame(
        [
            ["0022400147", "2024-11-02", 3, "WAS", "MIA", 98, 118, True],
            ["0062400001", "2024-12-17", 3, "MIL", "OKC", 97, 81, True],
            ["0022400999", "2025-04-14", 1, "BOS", "NYK", None, None, False],
        ],
        columns=[
            "gameId",
            "gameDate",
            "gameStatus",
            "homeTeam_teamTricode",
            "awayTeam_teamTricode",
            "homeTeam_score",
            "awayTeam_score",
            "isNeutral",
        ],
    )

    normalized = normalize_schedule_games(schedule_games, "2024-25")

    assert normalized.to_dict("records") == [
        {
            "game_id": "0022400147",
            "game_date": "2024-11-02",
            "season": "2024-25",
            "home_team": "WAS",
            "away_team": "MIA",
            "home_score": 98,
            "away_score": 118,
        }
    ]


def test_validate_season_games_rejects_incomplete_configured_season() -> None:
    incomplete_games = pd.DataFrame(
        [["0022400147", "WAS", "MIA"]], columns=["game_id", "home_team", "away_team"]
    )

    with pytest.raises(ValueError, match="expected 1230"):
        validate_season_games(incomplete_games, "2024-25")
