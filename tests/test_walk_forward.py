import numpy as np
import pandas as pd

from nba_predictor.evaluation.feature_ablation import (
    evaluate_feature_ablations,
    feature_columns_without,
)
from nba_predictor.evaluation.feature_sweep import evaluate_feature_sweep
from nba_predictor.evaluation.final_holdout import evaluate_final_holdout
from nba_predictor.evaluation.walk_forward import (
    bootstrap_candidate_deltas,
    bootstrap_metric_intervals,
    evaluate_walk_forward,
    probability_metrics,
)
from nba_predictor.features.team_form import build_pregame_features
from nba_predictor.models.feature_sets import (
    FEATURE_GROUPS,
    FEATURE_SWEEP_SETS,
    LEAN_LOGISTIC_FEATURE_COLUMNS,
)


def _sample_games() -> pd.DataFrame:
    rows: list[list[object]] = []
    game_number = 1
    for season_start in range(2018, 2023):
        season = f"{season_start}-{str(season_start + 1)[-2:]}"
        for day, home_score, away_score in [(1, 110, 100), (3, 95, 105), (5, 112, 101), (7, 98, 104)]:
            rows.append(
                [
                    str(game_number),
                    f"{season_start}-10-{day:02d}",
                    season,
                    "BOS" if game_number % 2 else "NYK",
                    "NYK" if game_number % 2 else "BOS",
                    home_score,
                    away_score,
                ]
            )
            game_number += 1
    return pd.DataFrame(
        rows,
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


def test_probability_metrics_reward_correct_probabilities() -> None:
    target = pd.Series([1, 0, 1, 0])

    metrics = probability_metrics(target, np.array([0.9, 0.1, 0.8, 0.2]))

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] < 0.2
    assert metrics["brier_score"] < 0.05


def test_walk_forward_excludes_the_reserved_holdout_season() -> None:
    features = build_pregame_features(_sample_games())

    metrics, predictions = evaluate_walk_forward(features, holdout_season="2022-23")

    assert set(metrics["validation_season"]) == {"2020-21", "2021-22"}
    assert "2022-23" not in predictions["validation_season"].unique()
    assert set(metrics["model"]) == {
        "always_home",
        "historical_home_rate",
        "elo_only",
        "logistic_baseline",
        "logistic_enhanced",
        "logistic_lean",
        "hist_gradient_boosting",
        "xgboost",
    }


def test_bootstrap_reports_include_model_intervals_and_paired_deltas() -> None:
    features = build_pregame_features(_sample_games())
    _, predictions = evaluate_walk_forward(features, holdout_season="2022-23")

    intervals = bootstrap_metric_intervals(predictions, n_resamples=10)
    deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model="logistic_enhanced",
        baseline_models=("historical_home_rate", "elo_only", "logistic_baseline"),
        n_resamples=10,
    )

    assert set(intervals["model"]) == set(predictions["model"])
    assert set(intervals["metric"]) == {"accuracy", "roc_auc", "log_loss", "brier_score"}
    assert set(deltas["baseline_model"]) == {
        "historical_home_rate",
        "elo_only",
        "logistic_baseline",
    }
    assert deltas["ci_supports_candidate"].dtype == bool


def test_gradient_boosting_is_compared_directly_with_logistic_enhanced() -> None:
    features = build_pregame_features(_sample_games())
    _, predictions = evaluate_walk_forward(features, holdout_season="2022-23")

    deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model="hist_gradient_boosting",
        baseline_models=("logistic_enhanced",),
        n_resamples=10,
    )

    assert set(deltas["candidate_model"]) == {"hist_gradient_boosting"}
    assert set(deltas["baseline_model"]) == {"logistic_enhanced"}


def test_xgboost_is_compared_directly_with_logistic_enhanced() -> None:
    features = build_pregame_features(_sample_games())
    _, predictions = evaluate_walk_forward(features, holdout_season="2022-23")

    deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model="xgboost",
        baseline_models=("logistic_enhanced",),
        n_resamples=10,
    )

    assert set(deltas["candidate_model"]) == {"xgboost"}
    assert set(deltas["baseline_model"]) == {"logistic_enhanced"}


def test_lean_logistic_is_compared_directly_with_logistic_enhanced() -> None:
    features = build_pregame_features(_sample_games())
    _, predictions = evaluate_walk_forward(features, holdout_season="2022-23")

    deltas = bootstrap_candidate_deltas(
        predictions,
        candidate_model="logistic_lean",
        baseline_models=("logistic_enhanced",),
        n_resamples=10,
    )

    assert set(deltas["candidate_model"]) == {"logistic_lean"}
    assert set(deltas["baseline_model"]) == {"logistic_enhanced"}


def test_feature_ablation_removes_only_the_requested_group() -> None:
    without_elo = feature_columns_without("elo")

    assert not set(FEATURE_GROUPS["elo"]) & set(without_elo)
    assert "home_win_pct_10" in without_elo


def test_feature_ablation_excludes_the_reserved_holdout_season() -> None:
    features = build_pregame_features(_sample_games())

    metrics, predictions = evaluate_feature_ablations(features, holdout_season="2022-23")

    assert set(metrics["validation_season"]) == {"2020-21", "2021-22"}
    assert "2022-23" not in predictions["validation_season"].unique()
    assert {"full_logistic", "without_elo"}.issubset(set(metrics["model"]))


def test_feature_sweep_defines_the_current_lean_set_without_short_term_form() -> None:
    assert FEATURE_SWEEP_SETS["lean_logistic"] == LEAN_LOGISTIC_FEATURE_COLUMNS
    assert "home_win_pct_5" not in FEATURE_SWEEP_SETS["lean_logistic"]
    assert "away_win_pct_5" not in FEATURE_SWEEP_SETS["lean_logistic"]


def test_feature_sweep_excludes_the_reserved_holdout_season() -> None:
    features = build_pregame_features(_sample_games())

    metrics, predictions = evaluate_feature_sweep(features, holdout_season="2022-23")

    assert set(metrics["validation_season"]) == {"2020-21", "2021-22"}
    assert "2022-23" not in predictions["validation_season"].unique()
    assert set(metrics["model"]) == set(FEATURE_SWEEP_SETS)


def test_final_holdout_uses_only_earlier_seasons_for_training() -> None:
    features = build_pregame_features(_sample_games())

    metrics, predictions = evaluate_final_holdout(features, holdout_season="2022-23")

    assert metrics["training_seasons"] == ["2018-19", "2019-20", "2020-21", "2021-22"]
    assert metrics["train_through_season"] == "2021-22"
    assert metrics["holdout_rows"] == 4
    assert set(predictions["holdout_season"]) == {"2022-23"}
    assert len(predictions) == 4
