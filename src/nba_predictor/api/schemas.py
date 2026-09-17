"""Validated request and response contracts for the prediction API."""

from __future__ import annotations

from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
