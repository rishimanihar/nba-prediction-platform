import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nba_predictor.api.app import create_app
from nba_predictor.models.feature_sets import LEAN_LOGISTIC_FEATURE_COLUMNS
from nba_predictor.models.logistic import build_logistic_pipeline


def _prediction_payload() -> dict[str, object]:
    return {
        "home_team": "BOS",
        "away_team": "NYK",
        "game_date": "2026-10-20",
        "home_win_pct_10": 0.7,
        "away_win_pct_10": 0.5,
        "home_avg_margin_10": 6.0,
        "away_avg_margin_10": 1.0,
        "home_avg_points_10": 116.0,
        "away_avg_points_10": 112.0,
        "home_avg_points_allowed_10": 108.0,
        "away_avg_points_allowed_10": 111.0,
        "home_home_win_pct_10": 0.8,
        "away_road_win_pct_10": 0.4,
        "home_rest_days": 2.0,
        "away_rest_days": 1.0,
        "home_games_last_7_days": 3,
        "away_games_last_7_days": 4,
        "home_elo": 1580.0,
        "away_elo": 1510.0,
    }


def _write_test_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    training_rows = pd.DataFrame(
        [
            [0.0] * len(LEAN_LOGISTIC_FEATURE_COLUMNS),
            [1.0] * len(LEAN_LOGISTIC_FEATURE_COLUMNS),
            [2.0] * len(LEAN_LOGISTIC_FEATURE_COLUMNS),
            [3.0] * len(LEAN_LOGISTIC_FEATURE_COLUMNS),
        ],
        columns=LEAN_LOGISTIC_FEATURE_COLUMNS,
    )
    model = build_logistic_pipeline()
    model.fit(training_rows, [0, 0, 1, 1])
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    evaluation_path = tmp_path / "evaluation.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model": "test_lean_logistic",
                "trained_at": "2026-09-17T00:00:00+00:00",
                "artifact_trained_rows": 8_289,
                "artifact_trained_seasons": ["2023-24", "2024-25"],
                "feature_count": len(LEAN_LOGISTIC_FEATURE_COLUMNS),
                "feature_columns": LEAN_LOGISTIC_FEATURE_COLUMNS,
                "excluded_holdout_season": None,
            }
        ),
        encoding="utf-8",
    )
    evaluation_path.write_text(
        json.dumps(
            {
                "holdout_season": "2024-25",
                "holdout_rows": 1_230,
                "feature_columns": LEAN_LOGISTIC_FEATURE_COLUMNS,
                "accuracy": 0.6699,
                "roc_auc": 0.7280,
                "log_loss": 0.6058,
                "brier_score": 0.2095,
            }
        ),
        encoding="utf-8",
    )
    return model_path, metadata_path, evaluation_path


def test_api_serves_health_model_info_and_prediction(tmp_path: Path) -> None:
    paths = _write_test_artifacts(tmp_path)
    application = create_app(*paths)

    with TestClient(application) as client:
        health = client.get("/health")
        model_info = client.get("/model")
        prediction = client.post("/predict", json=_prediction_payload())

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_name": "test_lean_logistic"}
    assert model_info.status_code == 200
    assert model_info.json()["trained_rows"] == 8_289
    assert model_info.json()["final_holdout_rows"] == 1_230
    assert prediction.status_code == 200
    assert prediction.json()["predicted_winner"] in {"BOS", "NYK"}
    assert prediction.json()["home_win_probability"] + prediction.json()[
        "away_win_probability"
    ] == pytest.approx(1.0)


def test_api_rejects_the_same_home_and_away_team(tmp_path: Path) -> None:
    paths = _write_test_artifacts(tmp_path)
    application = create_app(*paths)
    payload = _prediction_payload()
    payload["away_team"] = payload["home_team"]

    with TestClient(application) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_api_compares_model_probability_with_a_two_way_moneyline(tmp_path: Path) -> None:
    paths = _write_test_artifacts(tmp_path)
    application = create_app(*paths)
    payload = _prediction_payload() | {
        "home_moneyline": -120,
        "away_moneyline": 110,
        "bookmaker": "Example Sportsbook",
    }

    with TestClient(application) as client:
        response = client.post("/value", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["home_market_fair_probability"] + result[
        "away_market_fair_probability"
    ] == pytest.approx(1.0)
    assert result["home_model_edge"] + result["away_model_edge"] == pytest.approx(0.0)
    assert result["bookmaker"] == "Example Sportsbook"
