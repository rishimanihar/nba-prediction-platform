"""XGBoost pipeline for regularized NBA home-win forecasts."""

from xgboost import XGBClassifier


def build_xgboost_classifier() -> XGBClassifier:
    """Create a conservative XGBoost challenger for the logistic champion."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
    )
