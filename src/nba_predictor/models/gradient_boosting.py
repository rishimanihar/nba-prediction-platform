"""Gradient-boosted tree pipeline for nonlinear NBA home-win forecasts."""

from sklearn.ensemble import HistGradientBoostingClassifier


def build_hist_gradient_boosting_classifier() -> HistGradientBoostingClassifier:
    """Create a regularized first tree-based challenger for the logistic champion."""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
