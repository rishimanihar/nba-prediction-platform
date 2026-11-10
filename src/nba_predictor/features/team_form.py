"""Build leakage-safe pre-game form, schedule, and Elo features for NBA games."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

REQUIRED_COLUMNS = {
    "game_id",
    "game_date",
    "season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}
ELO_BASE_RATING = 1_500.0
ELO_HOME_COURT_ADVANTAGE = 100.0
ELO_K_FACTOR = 20.0
ELO_SEASON_CARRYOVER = 0.75


def _team_summary(
    history: list[dict[str, object]], window: int, venue: bool | None = None
) -> dict[str, float]:
    """Summarize only games played before the game being featurized."""
    eligible_games = (
        history if venue is None else [game for game in history if game["is_home"] == venue]
    )
    recent = eligible_games[-window:]
    if not recent:
        return {
            "win_pct": 0.5,
            "avg_margin": 0.0,
            "avg_points": 110.0,
            "avg_points_allowed": 110.0,
        }

    frame = pd.DataFrame(recent)
    return {
        "win_pct": float(frame["won"].mean()),
        "avg_margin": float(frame["margin"].mean()),
        "avg_points": float(frame["points"].mean()),
        "avg_points_allowed": float(frame["points_allowed"].mean()),
    }


def _rest_days(
    last_game_date: dict[str, pd.Timestamp], team: str, game_date: pd.Timestamp
) -> float:
    previous_date = last_game_date.get(team)
    if previous_date is None:
        return 7.0
    return float(max((game_date - previous_date).days - 1, 0))


def _games_in_last_days(
    history: list[dict[str, object]], game_date: pd.Timestamp, days: int
) -> float:
    """Count earlier games in a rolling calendar window, excluding the current game."""
    return float(
        sum(0 < (game_date - pd.Timestamp(game["game_date"])).days <= days for game in history)
    )


def elo_home_win_probability(home_elo: float, away_elo: float) -> float:
    """Return the pregame Elo probability after applying home-court advantage."""
    adjusted_difference = home_elo + ELO_HOME_COURT_ADVANTAGE - away_elo
    return 1.0 / (1.0 + 10.0 ** (-adjusted_difference / 400.0))


def _update_elo(home_elo: float, away_elo: float, home_won: bool) -> tuple[float, float]:
    expected_home_win = elo_home_win_probability(home_elo, away_elo)
    rating_change = ELO_K_FACTOR * (float(home_won) - expected_home_win)
    return home_elo + rating_change, away_elo - rating_change


def _regress_elo_for_new_season(elo_ratings: dict[str, float]) -> dict[str, float]:
    """Retain most team-strength signal without treating a new roster as unchanged."""
    return {
        team: ELO_BASE_RATING + ELO_SEASON_CARRYOVER * (rating - ELO_BASE_RATING)
        for team, rating in elo_ratings.items()
    }


def build_pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """Return one model row per completed game using only information available pre-tip-off.

    Team form resets at each season boundary. Elo ratings carry forward after regressing
    toward league average, preserving long-term signal without assuming unchanged rosters.
    """
    return _build_feature_rows(games, include_unplayed=False)


def build_fixture_features(completed_games: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Score scheduled matchups from strictly earlier completed games."""
    fixture_columns = REQUIRED_COLUMNS - {"home_score", "away_score"}
    missing_history = REQUIRED_COLUMNS - set(completed_games.columns)
    if missing_history:
        raise ValueError(
            f"Completed games are missing required columns: {', '.join(sorted(missing_history))}"
        )
    missing = fixture_columns - set(fixtures.columns)
    if missing:
        raise ValueError(f"Fixtures are missing required columns: {', '.join(sorted(missing))}")
    if fixtures.empty or completed_games.empty:
        raise ValueError("Completed games and fixtures must both be nonempty")
    fixture_ids = fixtures["game_id"].astype(str)
    history_ids = completed_games["game_id"].astype(str)
    if fixture_ids.duplicated().any() or history_ids.duplicated().any():
        raise ValueError("Game IDs must be unique")
    if set(fixture_ids) & set(history_ids):
        raise ValueError("Fixtures overlap completed games")
    fixture_dates = pd.to_datetime(fixtures["game_date"], errors="raise")
    history_dates = pd.to_datetime(completed_games["game_date"], errors="raise")
    if fixture_dates.isna().any() or history_dates.isna().any():
        raise ValueError("Game dates cannot be missing")
    if history_dates.max() >= fixture_dates.min():
        raise ValueError("Completed games must be strictly earlier than every fixture")
    if completed_games[["home_score", "away_score"]].isna().any().any():
        raise ValueError("Completed history contains missing scores")
    fixture_rows = fixtures[list(fixture_columns)].copy()
    fixture_rows["home_score"] = pd.NA
    fixture_rows["away_score"] = pd.NA
    all_rows = pd.concat([completed_games, fixture_rows], ignore_index=True)
    features = _build_feature_rows(all_rows, include_unplayed=True)
    return features.loc[features["game_id"].isin(set(fixture_ids))].reset_index(drop=True)


def _build_feature_rows(games: pd.DataFrame, include_unplayed: bool) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(games.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Games data is missing required columns: {missing}")

    frame = games.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise")
    if not include_unplayed:
        frame = frame.dropna(subset=["home_score", "away_score"])
    frame = frame.sort_values(["game_date", "game_id"], kind="stable")

    team_history: dict[str, list[dict[str, object]]] = defaultdict(list)
    last_game_date: dict[str, pd.Timestamp] = {}
    elo_ratings: dict[str, float] = defaultdict(lambda: ELO_BASE_RATING)
    active_season: str | None = None
    feature_rows: list[dict[str, object]] = []

    for game in frame.itertuples(index=False):
        completed = pd.notna(game.home_score) and pd.notna(game.away_score)
        season = str(game.season)
        if active_season is not None and season != active_season:
            team_history.clear()
            last_game_date.clear()
            elo_ratings = defaultdict(
                lambda: ELO_BASE_RATING, _regress_elo_for_new_season(elo_ratings)
            )
        active_season = season

        home = str(game.home_team)
        away = str(game.away_team)
        game_date = pd.Timestamp(game.game_date)
        home_form_5 = _team_summary(team_history[home], window=5)
        away_form_5 = _team_summary(team_history[away], window=5)
        home_form_10 = _team_summary(team_history[home], window=10)
        away_form_10 = _team_summary(team_history[away], window=10)
        home_home_form_10 = _team_summary(team_history[home], window=10, venue=True)
        away_road_form_10 = _team_summary(team_history[away], window=10, venue=False)
        home_rest_days = _rest_days(last_game_date, home, game_date)
        away_rest_days = _rest_days(last_game_date, away, game_date)
        home_elo = elo_ratings[home]
        away_elo = elo_ratings[away]

        feature_rows.append(
            {
                "game_id": str(game.game_id),
                "game_date": game_date,
                "season": season,
                "home_team": home,
                "away_team": away,
                "home_win_pct_5": home_form_5["win_pct"],
                "away_win_pct_5": away_form_5["win_pct"],
                "home_win_pct_10": home_form_10["win_pct"],
                "away_win_pct_10": away_form_10["win_pct"],
                "home_avg_margin_10": home_form_10["avg_margin"],
                "away_avg_margin_10": away_form_10["avg_margin"],
                "home_avg_points_10": home_form_10["avg_points"],
                "away_avg_points_10": away_form_10["avg_points"],
                "home_avg_points_allowed_10": home_form_10["avg_points_allowed"],
                "away_avg_points_allowed_10": away_form_10["avg_points_allowed"],
                "home_home_win_pct_10": home_home_form_10["win_pct"],
                "away_road_win_pct_10": away_road_form_10["win_pct"],
                "home_rest_days": home_rest_days,
                "away_rest_days": away_rest_days,
                "home_back_to_back": int(home_rest_days == 0),
                "away_back_to_back": int(away_rest_days == 0),
                "home_games_last_7_days": _games_in_last_days(
                    team_history[home], game_date, days=7
                ),
                "away_games_last_7_days": _games_in_last_days(
                    team_history[away], game_date, days=7
                ),
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_difference": home_elo - away_elo,
                "elo_home_win_probability": elo_home_win_probability(home_elo, away_elo),
                "home_win": int(float(game.home_score) > float(game.away_score))
                if completed
                else None,
                "home_point_diff": (
                    float(game.home_score) - float(game.away_score) if completed else None
                ),
            }
        )

        if not completed:
            continue
        home_margin = float(game.home_score) - float(game.away_score)
        home_won = home_margin > 0
        home_points = float(game.home_score)
        away_points = float(game.away_score)
        team_history[home].append(
            {
                "game_date": game_date,
                "is_home": True,
                "won": float(home_won),
                "margin": home_margin,
                "points": home_points,
                "points_allowed": away_points,
            }
        )
        team_history[away].append(
            {
                "game_date": game_date,
                "is_home": False,
                "won": float(not home_won),
                "margin": -home_margin,
                "points": away_points,
                "points_allowed": home_points,
            }
        )
        elo_ratings[home], elo_ratings[away] = _update_elo(home_elo, away_elo, home_won)
        last_game_date[home] = game_date
        last_game_date[away] = game_date

    return pd.DataFrame(feature_rows)
