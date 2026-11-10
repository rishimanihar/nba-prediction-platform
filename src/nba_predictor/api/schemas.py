"""Validated request and response contracts for the prediction API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    home_team: str = Field(pattern=r"^[A-Z]{2,4}$")
    away_team: str = Field(pattern=r"^[A-Z]{2,4}$")
    game_date: date
    home_win_pct_10: float = Field(ge=0.0, le=1.0)
    away_win_pct_10: float = Field(ge=0.0, le=1.0)
    home_avg_margin_10: float
    away_avg_margin_10: float
    home_avg_points_10: float = Field(ge=0.0)
    away_avg_points_10: float = Field(ge=0.0)
    home_avg_points_allowed_10: float = Field(ge=0.0)
    away_avg_points_allowed_10: float = Field(ge=0.0)
    home_home_win_pct_10: float = Field(ge=0.0, le=1.0)
    away_road_win_pct_10: float = Field(ge=0.0, le=1.0)
    home_rest_days: float = Field(ge=0.0)
    away_rest_days: float = Field(ge=0.0)
    home_games_last_7_days: int = Field(ge=0, le=7)
    away_games_last_7_days: int = Field(ge=0, le=7)
    home_elo: float = Field(gt=0.0)
    away_elo: float = Field(gt=0.0)

    @model_validator(mode="after")
    def teams_must_differ(self) -> Self:
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must differ")
        return self


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    game_date: date
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str
    model_name: str
    model_trained_at: str


class UpcomingGamePrediction(BaseModel):
    game_id: str
    game_date: date
    season: str
    home_team: str
    away_team: str
    home_win_probability: float = Field(ge=0.0, le=1.0)
    away_win_probability: float = Field(ge=0.0, le=1.0)
    predicted_winner: str

    @model_validator(mode="after")
    def probabilities_and_winner_must_agree(self) -> Self:
        if abs(self.home_win_probability + self.away_win_probability - 1.0) > 1e-6:
            raise ValueError("home and away probabilities must sum to one")
        expected = self.home_team if self.home_win_probability >= 0.5 else self.away_team
        if self.predicted_winner != expected:
            raise ValueError("predicted winner must match the probabilities")
        return self


class UpcomingGamesResponse(BaseModel):
    as_of_date: date
    generated_at: datetime
    model_name: str
    model_trained_at: str
    games: list[UpcomingGamePrediction]

    @model_validator(mode="after")
    def game_dates_must_match_snapshot(self) -> Self:
        if self.generated_at.tzinfo is None:
            raise ValueError("generation time needs a timezone")
        if any(game.game_date != self.as_of_date for game in self.games):
            raise ValueError("game dates must match the snapshot date")
        return self


class ValueRequest(PredictionRequest):
    home_moneyline: int
    away_moneyline: int
    bookmaker: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("home_moneyline", "away_moneyline")
    @classmethod
    def moneyline_must_be_valid(cls, value: int) -> int:
        if -100 < value < 100:
            raise ValueError("moneyline must be at most -100 or at least +100")
        return value


class ValueResponse(BaseModel):
    home_team: str
    away_team: str
    game_date: date
    bookmaker: str | None
    home_moneyline: int
    away_moneyline: int
    sportsbook_overround: float
    home_model_probability: float
    away_model_probability: float
    home_market_fair_probability: float
    away_market_fair_probability: float
    home_model_edge: float
    away_model_edge: float
    home_expected_value_per_unit: float
    away_expected_value_per_unit: float
    higher_estimated_value_side: str | None
    model_name: str
    model_trained_at: str


class HealthResponse(BaseModel):
    status: str
    model_name: str


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str
    trained_rows: int
    trained_seasons: list[str]
    feature_count: int
    feature_columns: list[str]
    final_holdout_season: str
    final_holdout_rows: int
    final_holdout_metrics: dict[str, float]
