"""Named, versioned feature sets used for fair model comparisons."""

BASELINE_FEATURE_COLUMNS = [
    "home_win_pct_10",
    "away_win_pct_10",
    "home_avg_margin_10",
    "away_avg_margin_10",
    "home_avg_points_10",
    "away_avg_points_10",
    "home_rest_days",
    "away_rest_days",
]

ENHANCED_FEATURE_COLUMNS = [
    "home_win_pct_5",
    "away_win_pct_5",
    "home_win_pct_10",
    "away_win_pct_10",
    "home_avg_margin_10",
    "away_avg_margin_10",
    "home_avg_points_10",
    "away_avg_points_10",
    "home_avg_points_allowed_10",
    "away_avg_points_allowed_10",
    "home_home_win_pct_10",
    "away_road_win_pct_10",
    "home_rest_days",
    "away_rest_days",
    "home_back_to_back",
    "away_back_to_back",
    "home_games_last_7_days",
    "away_games_last_7_days",
    "home_elo",
    "away_elo",
    "elo_difference",
    "elo_home_win_probability",
]

FEATURE_GROUPS = {
    "short_term_form": (
        "home_win_pct_5",
        "away_win_pct_5",
    ),
    "medium_term_form": (
        "home_win_pct_10",
        "away_win_pct_10",
        "home_avg_margin_10",
        "away_avg_margin_10",
        "home_avg_points_10",
        "away_avg_points_10",
        "home_avg_points_allowed_10",
        "away_avg_points_allowed_10",
    ),
    "venue_form": (
        "home_home_win_pct_10",
        "away_road_win_pct_10",
    ),
    "schedule": (
        "home_rest_days",
        "away_rest_days",
        "home_back_to_back",
        "away_back_to_back",
        "home_games_last_7_days",
        "away_games_last_7_days",
    ),
    "elo": (
        "home_elo",
        "away_elo",
        "elo_difference",
        "elo_home_win_probability",
    ),
}

LEAN_LOGISTIC_FEATURE_COLUMNS = [
    column
    for column in ENHANCED_FEATURE_COLUMNS
    if column not in {"home_win_pct_5", "away_win_pct_5"}
]


def _feature_columns_for_groups(group_names: tuple[str, ...]) -> list[str]:
    """Return group columns in the stable enhanced-feature order."""
    selected_columns = {
        column for group_name in group_names for column in FEATURE_GROUPS[group_name]
    }
    return [column for column in ENHANCED_FEATURE_COLUMNS if column in selected_columns]


# Purposeful combinations, selected after the group-removal ablation. The short-term
# five-game win rates are omitted because they did not improve the fuller model.
FEATURE_SWEEP_GROUPS = {
    "elo_only": ("elo",),
    "elo_venue": ("elo", "venue_form"),
    "elo_medium_term_form": ("elo", "medium_term_form"),
    "elo_medium_term_venue": ("elo", "medium_term_form", "venue_form"),
    "elo_medium_term_schedule": ("elo", "medium_term_form", "schedule"),
    "lean_logistic": ("elo", "medium_term_form", "venue_form", "schedule"),
}

FEATURE_SWEEP_SETS = {
    model_name: _feature_columns_for_groups(group_names)
    for model_name, group_names in FEATURE_SWEEP_GROUPS.items()
}
