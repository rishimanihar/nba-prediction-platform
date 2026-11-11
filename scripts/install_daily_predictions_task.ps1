[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidatePattern('^([01][0-9]|2[0-3]):[0-5][0-9]$')]
    [string]$At = '09:00',
    [switch]$RunNow
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$taskName = 'NBA Predictor Daily Predictions'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runnerPath = Join-Path $projectRoot 'scripts\run_daily_predictions.ps1'
$pwshPath = 'C:\Program Files\PowerShell\7\pwsh.exe'
if (-not (Test-Path -LiteralPath $runnerPath -PathType Leaf)) {
    throw "Prediction runner was not found: $runnerPath"
}
if (-not (Test-Path -LiteralPath $pwshPath -PathType Leaf)) {
    throw "PowerShell 7 was not found: $pwshPath"
}
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$timeOfDay = [TimeSpan]::Parse($At, [Globalization.CultureInfo]::InvariantCulture)
Write-Host "Task: $taskName"
Write-Host "Runs as: $currentUser (while signed in)"
Write-Host "Daily time: $At local Windows time"
Write-Host "Runner: $runnerPath"
if (-not $PSCmdlet.ShouldProcess($taskName, 'Register daily prediction refresh')) {
    return
}
if ($currentUser -match 'CodexSandboxOffline') {
    throw 'Run this installer from your own PowerShell terminal so the task belongs to your Windows account.'
}
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw "Task already exists: $taskName. Review it in Task Scheduler before replacing it."
}
$action = New-ScheduledTaskAction -Execute $pwshPath `
    -Argument ('-NoProfile -NonInteractive -File "{0}"' -f $runnerPath) `
    -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::Today.Add($timeOfDay))
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description 'Refresh the NBA pregame prediction snapshot every morning.' | Out-Null
if ($RunNow) {
    Start-ScheduledTask -TaskName $taskName
    Write-Host 'Started an initial refresh. It may take several minutes; check artifacts\logs.'
}
Get-ScheduledTask -TaskName $taskName | Get-ScheduledTaskInfo |
    Select-Object LastRunTime, NextRunTime, LastTaskResult
