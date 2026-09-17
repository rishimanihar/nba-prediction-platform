param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$base = $BaseUrl.TrimEnd("/")

$health = Invoke-RestMethod -Uri "$base/health" -Method Get
if ($health.status -ne "ok") {
    throw "Health check returned unexpected status: $($health.status)"
}

$model = Invoke-RestMethod -Uri "$base/model" -Method Get
if ($model.trained_rows -ne 8289) {
    throw "Expected a model trained on 8289 rows, received $($model.trained_rows)."
}
if ($model.final_holdout_rows -ne 1230) {
    throw "Expected 1230 final holdout rows, received $($model.final_holdout_rows)."
}

$payload = @{
    home_team = "BOS"
    away_team = "NYK"
    game_date = "2026-10-20"
    home_win_pct_10 = 0.7
    away_win_pct_10 = 0.5
    home_avg_margin_10 = 6.0
    away_avg_margin_10 = 1.0
    home_avg_points_10 = 116.0
    away_avg_points_10 = 112.0
    home_avg_points_allowed_10 = 108.0
    away_avg_points_allowed_10 = 111.0
    home_home_win_pct_10 = 0.8
    away_road_win_pct_10 = 0.4
    home_rest_days = 2.0
    away_rest_days = 1.0
    home_games_last_7_days = 3
    away_games_last_7_days = 4
    home_elo = 1580.0
    away_elo = 1510.0
} | ConvertTo-Json

$prediction = Invoke-RestMethod `
    -Uri "$base/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $payload

$probabilitySum = $prediction.home_win_probability + $prediction.away_win_probability
if ([math]::Abs($probabilitySum - 1.0) -gt 0.000000001) {
    throw "Prediction probabilities do not sum to 1.0."
}

Write-Host "API smoke test passed." -ForegroundColor Green
Write-Host "Model: $($model.model_name)"
Write-Host "Predicted winner: $($prediction.predicted_winner)"
Write-Host "Home win probability: $($prediction.home_win_probability)"
