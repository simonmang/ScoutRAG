[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8000,
    [switch]$EnableDenseRetrieval,
    [switch]$NoBrowser,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$scoutProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$scoutPython = Join-Path $scoutProjectRoot ".venv\Scripts\python.exe"
$scoutProfiles = Join-Path $scoutProjectRoot "data\processed\bundesliga-2023-2024\player_season_profiles.parquet"
$scoutEvidence = Join-Path $scoutProjectRoot "data\processed\bundesliga-2023-2024\player_metric_evidence.parquet"

foreach ($scoutRequiredPath in @($scoutPython, $scoutProfiles, $scoutEvidence)) {
    if (-not (Test-Path -LiteralPath $scoutRequiredPath)) {
        throw "ScoutRAG-Datei fehlt: $scoutRequiredPath"
    }
}

$scoutDenseMode = if ($EnableDenseRetrieval) { "true" } else { "false" }
$env:SCOUTRAG_ANSWER_MODE = "template"
$env:SCOUTRAG_ENABLE_DENSE_RETRIEVAL = $scoutDenseMode
$env:SCOUTRAG_ENVIRONMENT = "development"
$env:SCOUTRAG_LOCAL_FILES_ONLY = "true"

if ($Check) {
    Write-Host "ScoutRAG ist startbereit." -ForegroundColor Green
    Write-Host "Python: $scoutPython"
    Write-Host "Dense Retrieval: $scoutDenseMode"
    exit 0
}

$scoutDashboardUrl = "http://127.0.0.1:$Port"
Write-Host ""
Write-Host "ScoutRAG startet lokal auf $scoutDashboardUrl" -ForegroundColor Green
Write-Host "Zum Beenden dieses Fenster schließen oder Strg+C drücken."
Write-Host ""

if (-not $NoBrowser) {
    Start-Process $scoutDashboardUrl
}

Push-Location $scoutProjectRoot
try {
    & $scoutPython -m uvicorn scoutrag.main:app `
        --host 127.0.0.1 `
        --port $Port `
        --reload
}
finally {
    Pop-Location
}
