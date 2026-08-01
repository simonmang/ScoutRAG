[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8000,
    [switch]$EnableDenseRetrieval,
    [switch]$OpenAI,
    [ValidateSet("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")]
    [string]$OpenAIModel = "gpt-5.6-luna",
    [switch]$NoBrowser,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$scoutProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$scoutPython = Join-Path $scoutProjectRoot ".venv\Scripts\python.exe"

foreach ($scoutRequiredPath in @($scoutPython)) {
    if (-not (Test-Path -LiteralPath $scoutRequiredPath)) {
        throw "ScoutRAG-Datei fehlt: $scoutRequiredPath"
    }
}

Push-Location $scoutProjectRoot
try {
    $scoutProfiles = & $scoutPython -c "from scoutrag.config import Settings; print(Settings().profiles_path)"
    $scoutEvidence = & $scoutPython -c "from scoutrag.config import Settings; print(Settings().metric_evidence_path)"
}
finally {
    Pop-Location
}

foreach ($scoutDataPath in @($scoutProfiles, $scoutEvidence)) {
    $scoutResolvedDataPath = Join-Path $scoutProjectRoot $scoutDataPath
    if (-not (Test-Path -LiteralPath $scoutResolvedDataPath)) {
        throw "ScoutRAG-Datendatei fehlt: $scoutResolvedDataPath"
    }
}

$scoutDenseMode = if ($EnableDenseRetrieval) { "true" } else { "false" }
$scoutAnswerMode = if ($OpenAI) { "openai" } else { "template" }
$scoutTemporaryApiKey = $false

if ($OpenAI) {
    & $scoutPython -c "import openai" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "OpenAI-Unterstuetzung fehlt. Installiere: .venv\Scripts\python.exe -m pip install -e `".[llm]`""
    }

    if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
        $scoutEnvPath = Join-Path $scoutProjectRoot ".env"
        if (Test-Path -LiteralPath $scoutEnvPath) {
            $scoutKeyLine = Get-Content -LiteralPath $scoutEnvPath |
                Where-Object { $_ -match "^\s*OPENAI_API_KEY\s*=" } |
                Select-Object -Last 1
            if ($null -ne $scoutKeyLine) {
                $scoutApiKey = ($scoutKeyLine -split "=", 2)[1].Trim().Trim('"').Trim("'")
                $scoutPlaceholders = @(
                    "HIER_DEINEN_API_KEY_EINFUEGEN",
                    "replace-with-your-key"
                )
                if (
                    -not [string]::IsNullOrWhiteSpace($scoutApiKey) -and
                    $scoutApiKey -notin $scoutPlaceholders
                ) {
                    $env:OPENAI_API_KEY = $scoutApiKey
                    $scoutApiKey = $null
                    $scoutTemporaryApiKey = $true
                    Write-Host "API-Key wurde aus der lokalen .env-Datei geladen." -ForegroundColor Cyan
                }
            }
        }
    }

    if (-not $Check -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
        throw "Trage deinen Key in die Datei .env ein: OPENAI_API_KEY=dein-key"
    }

    $env:SCOUTRAG_OPENAI_MODEL = $OpenAIModel
}

$env:SCOUTRAG_ANSWER_MODE = $scoutAnswerMode
$env:SCOUTRAG_ENABLE_DENSE_RETRIEVAL = $scoutDenseMode
$env:SCOUTRAG_ENVIRONMENT = "development"
$env:SCOUTRAG_LOCAL_FILES_ONLY = "true"

if ($Check) {
    Write-Host "ScoutRAG ist startbereit." -ForegroundColor Green
    Write-Host "Python: $scoutPython"
    Write-Host "Dense Retrieval: $scoutDenseMode"
    Write-Host "Antwortmodus: $scoutAnswerMode"
    if ($OpenAI) {
        $scoutKeyPresent = -not [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)
        Write-Host "OpenAI-Modell: $OpenAIModel"
        Write-Host "API-Key in Umgebung vorhanden: $scoutKeyPresent"
    }
    exit 0
}

$scoutDashboardUrl = "http://127.0.0.1:$Port"
Write-Host ""
Write-Host "ScoutRAG startet lokal auf $scoutDashboardUrl" -ForegroundColor Green
Write-Host "Antwortmodus: $scoutAnswerMode"
if ($OpenAI) {
    Write-Host "OpenAI-Modell: $OpenAIModel"
}
Write-Host "Zum Beenden dieses Fenster schließen oder Strg+C drücken."
Write-Host ""

$scoutLocationPushed = $false
try {
    if (-not $NoBrowser) {
        Start-Process $scoutDashboardUrl
    }

    Push-Location $scoutProjectRoot
    $scoutLocationPushed = $true
    & $scoutPython -m uvicorn scoutrag.main:app `
        --host 127.0.0.1 `
        --port $Port `
        --reload
}
finally {
    if ($scoutLocationPushed) {
        Pop-Location
    }
    if ($scoutTemporaryApiKey) {
        Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    }
}
