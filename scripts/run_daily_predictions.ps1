param(
    [string]$Season = '2026-27',
    [string[]]$HistoryFiles = @('data/raw/games.csv', 'data/raw/games_2025_26.csv'),
    [string]$GameDate = '',
    [string]$ScheduleCsv = '',
    [string]$SnapshotPath = 'artifacts/upcoming_predictions.json',
    [ValidateRange(1, 5)][int]$MaxAttempts = 3,
    [ValidateRange(0, 600)][int]$RetryDelaySeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python was not found: $pythonPath"
}

$easternZone = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$easternToday = [TimeZoneInfo]::ConvertTimeFromUtc([datetime]::UtcNow, $easternZone).ToString('yyyy-MM-dd')
$targetDate = if ($GameDate) { $GameDate } else { $easternToday }
$parsedDate = [datetime]::MinValue
if (-not [datetime]::TryParseExact(
    $targetDate, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::None, [ref]$parsedDate
)) {
    throw "GameDate must have the form YYYY-MM-DD: $targetDate"
}

Push-Location $projectRoot
try {
    foreach ($historyFile in $HistoryFiles) {
        if (-not (Test-Path -LiteralPath $historyFile -PathType Leaf)) {
            throw "Historical games file was not found: $historyFile"
        }
    }
    if ($ScheduleCsv -and -not (Test-Path -LiteralPath $ScheduleCsv -PathType Leaf)) {
        throw "Saved schedule was not found: $ScheduleCsv"
    }
    $logDirectory = Join-Path $projectRoot 'artifacts\logs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $logPath = Join-Path $logDirectory "upcoming_$targetDate.log"

    function Write-RunLog([string]$message) {
        $line = '{0} {1}' -f [datetime]::UtcNow.ToString('o'), $message
        Add-Content -LiteralPath $logPath -Value $line
        Write-Host $line
    }

    $arguments = @('-m', 'nba_predictor.predict_upcoming', '--season', $Season,
        '--date', $targetDate, '--history') + $HistoryFiles + @('--output', $SnapshotPath)
    if ($ScheduleCsv) {
        $arguments += @('--schedule-csv', $ScheduleCsv)
    }
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-RunLog "Attempt $attempt of $MaxAttempts for $Season on $targetDate"
        $commandOutput = @()
        $exitCode = 1
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $commandOutput = & $pythonPath @arguments 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousPreference
        }
        foreach ($line in $commandOutput) {
            Write-RunLog ([string]$line)
        }
        if ($exitCode -eq 0) {
            Write-RunLog "Prediction refresh succeeded: $SnapshotPath"
            exit 0
        }
        Write-RunLog "Prediction refresh failed with exit code $exitCode"
        if ($attempt -lt $MaxAttempts) {
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
    throw "Prediction refresh failed after $MaxAttempts attempts; see $logPath"
}
finally {
    Pop-Location
}
