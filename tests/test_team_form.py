import pandas as pd

from nba_predictor.features.team_form import ELO_BASE_RATING, build_pregame_features


def test_features_are_calculated_before_the_current_game() -> None:
    games = pd.DataFrame(
        [
            ["1", "2024-10-22", "2024", "BOS", "NYK", 100, 90],
            ["2", "2024-10-24", "2024", "BOS", "WAS", 80, 110],
        ],
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

    features = build_pregame_features(games)
    second_game = features.iloc[1]

    assert second_game["home_win_pct_10"] == 1.0
    assert second_game["home_avg_margin_10"] == 10.0
    assert second_game["home_rest_days"] == 1.0
    assert second_game["home_win"] == 0


def test_enhanced_features_use_only_prior_games() -> None:
    games = pd.DataFrame(
        [
            ["1", "2024-10-22", "2024-25", "BOS", "NYK", 100, 90],
            ["2", "2024-10-23", "2024-25", "BOS", "WAS", 80, 110],
        ],
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

    features = build_pregame_features(games)
    second_game = features.iloc[1]

    assert second_game["home_win_pct_5"] == 1.0
    assert second_game["home_avg_points_allowed_10"] == 90.0
    assert second_game["home_home_win_pct_10"] == 1.0
    assert second_game["home_back_to_back"] == 1
    assert second_game["home_games_last_7_days"] == 1.0
    assert second_game["home_elo"] > ELO_BASE_RATING
    assert second_game["home_win"] == 0


def test_elo_regresses_toward_average_when_a_new_season_starts() -> None:
    games = pd.DataFrame(
        [
            ["1", "2024-10-22", "2024-25", "BOS", "NYK", 100, 90],
            ["2", "2025-10-22", "2025-26", "BOS", "NYK", 100, 90],
        ],
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

    features = build_pregame_features(games)

    assert ELO_BASE_RATING < features.iloc[1]["home_elo"] < features.iloc[0]["home_elo"] + 20
    assert features.iloc[1]["home_win_pct_10"] == 0.5
