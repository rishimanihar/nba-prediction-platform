"""Train the approved lean logistic NBA home-win model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from nba_predictor.features.team_form import build_pregame_features
from nba_predictor.models.feature_sets import LEAN_LOGISTIC_FEATURE_COLUMNS
from nba_predictor.models.logistic import build_logistic_pipeline

MODEL_NAME = "logistic_regression_home_win_lean_features"


def train(
    input_path: Path, artifact_dir: Path, holdout_season: str | None = None
) -> dict[str, object]:
    games = pd.read_csv(input_path)
    features = build_pregame_features(games)
    if len(features) < 20:
        raise ValueError("At least 20 completed games are required to train the model.")

    development_rows = features
    if holdout_season is not None:
        if holdout_season not in set(features["season"]):
            raise ValueError(f"Holdout season {holdout_season!r} is not present in the input data.")
        development_rows = features[features["season"] != holdout_season]
    if len(development_rows) < 20:
        raise ValueError("At least 20 development rows are required after reserving the holdout season.")

    split_index = int(len(development_rows) * 0.8)
    evaluation_train_rows = development_rows.iloc[:split_index]
    evaluation_test_rows = development_rows.iloc[split_index:]
    if evaluation_train_rows["home_win"].nunique() < 2 or evaluation_test_rows["home_win"].nunique() < 2:
        raise ValueError("Both chronological splits must contain home wins and home losses.")

    evaluation_model = build_logistic_pipeline()
    evaluation_model.fit(evaluation_train_rows[LEAN_LOGISTIC_FEATURE_COLUMNS], evaluation_train_rows["home_win"])
    probabilities = evaluation_model.predict_proba(evaluation_test_rows[LEAN_LOGISTIC_FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Refit after evaluation so the artifact uses all available development seasons.
    artifact_model = build_logistic_pipeline()
    artifact_model.fit(development_rows[LEAN_LOGISTIC_FEATURE_COLUMNS], development_rows["home_win"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_model, artifact_dir / "home_win_lean_logistic.joblib")
    metrics: dict[str, object] = {
        "model": MODEL_NAME,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_count": len(LEAN_LOGISTIC_FEATURE_COLUMNS),
        "feature_columns": LEAN_LOGISTIC_FEATURE_COLUMNS,
        "artifact_trained_rows": len(development_rows),
        "artifact_trained_seasons": sorted(development_rows["season"].unique().tolist()),
        "excluded_holdout_season": holdout_season,
        "evaluation_train_rows": len(evaluation_train_rows),
        "evaluation_test_rows": len(evaluation_test_rows),
        "accuracy": round(float(accuracy_score(evaluation_test_rows["home_win"], predictions)), 4),
        "roc_auc": round(float(roc_auc_score(evaluation_test_rows["home_win"], probabilities)), 4),
        "log_loss": round(float(log_loss(evaluation_test_rows["home_win"], probabilities)), 4),
        "brier_score": round(float(brier_score_loss(evaluation_test_rows["home_win"], probabilities)), 4),
    }
    (artifact_dir / "metrics_lean_logistic.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to historical completed games CSV.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--holdout-season", help="Optional season to exclude from the model artifact.")
    args = parser.parse_args()
    metrics = train(args.input, args.artifacts, args.holdout_season)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
