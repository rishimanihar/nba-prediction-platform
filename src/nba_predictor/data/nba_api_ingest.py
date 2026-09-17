"""Fetch completed NBA regular-season games and normalize them for model training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

SCHEDULE_COLUMNS = {
    "gameId",
    "gameDate",
    "gameStatus",
    "homeTeam_teamTricode",
    "awayTeam_teamTricode",
    "homeTeam_score",
    "awayTeam_score",
}
NORMALIZED_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]
EXPECTED_SEASON_GAME_COUNTS = {
    "2018-19": 1_230,
    "2019-20": 1_059,
    "2020-21": 1_080,
    "2021-22": 1_230,
    "2022-23": 1_230,
    "2023-24": 1_230,
    "2024-25": 1_230,
}


def season_label(start_year: int) -> str:
    """Convert 2018 to NBA's 2018-19 season label."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def normalize_schedule_games(schedule_games: pd.DataFrame, season: str) -> pd.DataFrame:
    """Normalize completed regular-season schedule rows with explicit home and away teams."""
    missing_columns = SCHEDULE_COLUMNS - set(schedule_games.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"NBA API response is missing expected columns: {missing}")

    game_ids = schedule_games["gameId"].astype(str).str.zfill(10)
    game_status = pd.to_numeric(schedule_games["gameStatus"], errors="coerce")
    home_scores = pd.to_numeric(schedule_games["homeTeam_score"], errors="coerce")
    away_scores = pd.to_numeric(schedule_games["awayTeam_score"], errors="coerce")
    completed_regular_season = schedule_games[
        game_ids.str.startswith("002")
        & game_status.eq(3)
        & home_scores.notna()
        & away_scores.notna()
    ].copy()
    completed_regular_season["game_id"] = game_ids.loc[completed_regular_season.index]
    completed_regular_season["game_date"] = pd.to_datetime(
        completed_regular_season["gameDate"]
    ).dt.date.astype(str)
    completed_regular_season["season"] = season
    completed_regular_season["home_team"] = completed_regular_season[
        "homeTeam_teamTricode"
    ].astype(str)
    completed_regular_season["away_team"] = completed_regular_season[
        "awayTeam_teamTricode"
    ].astype(str)
    completed_regular_season["home_score"] = home_scores.loc[
        completed_regular_season.index
    ].astype(int)
    completed_regular_season["away_score"] = away_scores.loc[
        completed_regular_season.index
    ].astype(int)
    return completed_regular_season[NORMALIZED_COLUMNS].reset_index(drop=True)


def validate_season_games(games: pd.DataFrame, season: str) -> None:
    """Fail when a configured completed season is incomplete or contains duplicate games."""
    duplicate_ids = games.loc[games["game_id"].duplicated(), "game_id"].tolist()
    if duplicate_ids:
        raise ValueError(f"Season {season} contains duplicate game IDs: {duplicate_ids}")

    expected_count = EXPECTED_SEASON_GAME_COUNTS.get(season)
    if expected_count is not None and len(games) != expected_count:
        team_counts = pd.concat([games["home_team"], games["away_team"]]).value_counts()
        deficient_teams = ", ".join(
            f"{team}:{count}" for team, count in team_counts.sort_index().items() if count < 82
        )
        raise ValueError(
            f"Season {season} has {len(games)} games; expected {expected_count}. "
            f"Team game counts below 82: {deficient_teams or 'none'}"
        )


def fetch_season(season: str) -> pd.DataFrame:
    """Request one regular season from NBA.com through nba_api."""
    from nba_api.stats.endpoints import scheduleleaguev2

    response = scheduleleaguev2.ScheduleLeagueV2(
        league_id="00",
        season=season,
        timeout=60,
    )
    return response.season_games.get_data_frame()


def ingest_seasons(start_year: int, end_year: int, output_path: Path) -> pd.DataFrame:
    """Fetch inclusive season range, normalize rows, and write a deterministic CSV."""
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")

    normalized_seasons: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        season = season_label(year)
        print(f"Fetching {season}...")
        season_games = normalize_schedule_games(fetch_season(season), season)
        validate_season_games(season_games, season)
        normalized_seasons.append(season_games)
        if year < end_year:
            time.sleep(0.8)

    games = pd.concat(normalized_seasons, ignore_index=True)
    games = games.sort_values(["game_date", "game_id"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(output_path, index=False)
    print(f"Wrote {len(games):,} completed games to {output_path}")
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, required=True, help="First season's start year.")
    parser.add_argument("--end-season", type=int, required=True, help="Last season's start year.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/games.csv"))
    args = parser.parse_args()
    ingest_seasons(args.start_season, args.end_season, args.output)


if __name__ == "__main__":
    main()
