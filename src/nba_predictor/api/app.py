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
)
from nba_predictor.features.team_form import elo_home_win_probability
from nba_predictor.models.feature_sets import LEAN_LOGISTIC_FEATURE_COLUMNS

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
    values = payload.model_dump(exclude={"home_team", "away_team", "game_date"})
    values["home_back_to_back"] = int(payload.home_rest_days == 0)
    values["away_back_to_back"] = int(payload.away_rest_days == 0)
    values["elo_difference"] = payload.home_elo - payload.away_elo
    values["elo_home_win_probability"] = elo_home_win_probability(
        payload.home_elo, payload.away_elo
    )
    return pd.DataFrame([values], columns=LEAN_LOGISTIC_FEATURE_COLUMNS)


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
        probability = float(
            request.app.state.model.predict_proba(_feature_frame(payload))[0, 1]
        )
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

    return application


app = create_app()
