"""FastAPI application that serves the approved NBA home-win model."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from nba_predictor.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ValueRequest,
    ValueResponse,
)
from nba_predictor.features.team_form import elo_home_win_probability
from nba_predictor.models.feature_sets import LEAN_LOGISTIC_FEATURE_COLUMNS
from nba_predictor.odds.value import expected_value_per_unit, remove_two_way_vig

DEFAULT_MODEL_PATH = Path("artifacts/home_win_lean_logistic.joblib")
DEFAULT_METADATA_PATH = Path("artifacts/metrics_lean_logistic.json")
DEFAULT_FINAL_EVALUATION_PATH = Path("artifacts/final_holdout_corrected/holdout_metrics.json")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise RuntimeError(f"Required artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_artifacts(metadata: dict[str, object], final_evaluation: dict[str, object]) -> None:
    if metadata.get("excluded_holdout_season") is not None:
        raise RuntimeError(
            "The model artifact excludes the final holdout. Retrain on all seasons before serving."
        )
    if metadata.get("feature_columns") != LEAN_LOGISTIC_FEATURE_COLUMNS:
        raise RuntimeError("Model metadata does not match the approved lean feature set.")
    if final_evaluation.get("feature_columns") != LEAN_LOGISTIC_FEATURE_COLUMNS:
        raise RuntimeError("Final evaluation metadata does not match the approved lean feature set.")


def _feature_frame(payload: PredictionRequest) -> pd.DataFrame:
    derived_columns = {
        "home_back_to_back",
        "away_back_to_back",
        "elo_difference",
        "elo_home_win_probability",
    }
    independent_columns = set(LEAN_LOGISTIC_FEATURE_COLUMNS) - derived_columns
    values = payload.model_dump(include=independent_columns)
    values["home_back_to_back"] = int(payload.home_rest_days == 0)
    values["away_back_to_back"] = int(payload.away_rest_days == 0)
    values["elo_difference"] = payload.home_elo - payload.away_elo
    values["elo_home_win_probability"] = elo_home_win_probability(
        payload.home_elo, payload.away_elo
    )
    return pd.DataFrame([values], columns=LEAN_LOGISTIC_FEATURE_COLUMNS)


def _home_win_probability(payload: PredictionRequest, request: Request) -> float:
    return float(request.app.state.model.predict_proba(_feature_frame(payload))[0, 1])


def create_app(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    final_evaluation_path: Path | None = None,
) -> FastAPI:
    resolved_model_path = model_path or Path(os.getenv("NBA_MODEL_PATH", DEFAULT_MODEL_PATH))
    resolved_metadata_path = metadata_path or Path(
        os.getenv("NBA_MODEL_METADATA_PATH", DEFAULT_METADATA_PATH)
    )
    resolved_final_evaluation_path = final_evaluation_path or Path(
        os.getenv("NBA_FINAL_EVALUATION_PATH", DEFAULT_FINAL_EVALUATION_PATH)
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        metadata = _read_json(resolved_metadata_path)
        final_evaluation = _read_json(resolved_final_evaluation_path)
        _validate_artifacts(metadata, final_evaluation)
        application.state.model = joblib.load(resolved_model_path)
        application.state.metadata = metadata
        application.state.final_evaluation = final_evaluation
        yield

    application = FastAPI(
        title="NBA Home-Win Prediction API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_name=str(request.app.state.metadata["model"]),
        )

    @application.get("/model", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        metadata = request.app.state.metadata
        final_evaluation = request.app.state.final_evaluation
        return ModelInfoResponse(
            model_name=str(metadata["model"]),
            trained_at=str(metadata["trained_at"]),
            trained_rows=int(metadata["artifact_trained_rows"]),
            trained_seasons=list(metadata["artifact_trained_seasons"]),
            feature_count=int(metadata["feature_count"]),
            feature_columns=list(metadata["feature_columns"]),
            final_holdout_season=str(final_evaluation["holdout_season"]),
            final_holdout_rows=int(final_evaluation["holdout_rows"]),
            final_holdout_metrics={
                metric: float(final_evaluation[metric])
                for metric in ("accuracy", "roc_auc", "log_loss", "brier_score")
            },
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        probability = _home_win_probability(payload, request)
        metadata = request.app.state.metadata
        return PredictionResponse(
            home_team=payload.home_team,
            away_team=payload.away_team,
            game_date=payload.game_date,
            home_win_probability=probability,
            away_win_probability=1.0 - probability,
            predicted_winner=payload.home_team if probability >= 0.5 else payload.away_team,
            model_name=str(metadata["model"]),
            model_trained_at=str(metadata["trained_at"]),
        )

    @application.post("/value", response_model=ValueResponse)
    def value(payload: ValueRequest, request: Request) -> ValueResponse:
        home_probability = _home_win_probability(payload, request)
        away_probability = 1.0 - home_probability
        market = remove_two_way_vig(payload.home_moneyline, payload.away_moneyline)
        home_ev = expected_value_per_unit(home_probability, payload.home_moneyline)
        away_ev = expected_value_per_unit(away_probability, payload.away_moneyline)
        higher_estimated_value_side: str | None = None
        if max(home_ev, away_ev) > 0.0:
            higher_estimated_value_side = (
                payload.home_team if home_ev >= away_ev else payload.away_team
            )
        metadata = request.app.state.metadata
        return ValueResponse(
            home_team=payload.home_team,
            away_team=payload.away_team,
            game_date=payload.game_date,
            bookmaker=payload.bookmaker,
            home_moneyline=payload.home_moneyline,
            away_moneyline=payload.away_moneyline,
            sportsbook_overround=market.overround,
            home_model_probability=home_probability,
            away_model_probability=away_probability,
            home_market_fair_probability=market.home_fair_probability,
            away_market_fair_probability=market.away_fair_probability,
            home_model_edge=home_probability - market.home_fair_probability,
            away_model_edge=away_probability - market.away_fair_probability,
            home_expected_value_per_unit=home_ev,
            away_expected_value_per_unit=away_ev,
            higher_estimated_value_side=higher_estimated_value_side,
            model_name=str(metadata["model"]),
            model_trained_at=str(metadata["trained_at"]),
        )

    return application


app = create_app()
