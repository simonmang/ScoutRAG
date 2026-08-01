[CmdletBinding()]
param(
    [switch]$Download,
    [switch]$Build,
    [ValidateSet("top5", "top5_second", "scouting", "scouting_second")]
    [string[]]$Groups = @("top5", "top5_second", "scouting", "scouting_second"),
    [ValidateRange(0, 2100)]
    [int]$SeasonStartYear = 0,
    [ValidateRange(20, 7500)]
    [int]$RequestBudgetPerLeague = 250
)

$ErrorActionPreference = "Stop"
$scoutProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$scoutDataCli = Join-Path $scoutProjectRoot ".venv\Scripts\scoutrag-data.exe"
$scoutCatalogPath = Join-Path $scoutProjectRoot "config\scouting_leagues.json"

if (-not $Download -and -not $Build) {
    $Download = $true
    $Build = $true
}

foreach ($scoutRequiredPath in @($scoutDataCli, $scoutCatalogPath)) {
    if (-not (Test-Path -LiteralPath $scoutRequiredPath)) {
        throw "ScoutRAG-Datei fehlt: $scoutRequiredPath"
    }
}

$scoutCatalog = Get-Content -LiteralPath $scoutCatalogPath -Raw |
    ConvertFrom-Json
$scoutEntries = @(
    $scoutCatalog.entries |
        Where-Object { $_.group -in $Groups }
)
if ($scoutEntries.Count -eq 0) {
    throw "Der gewählte Katalog enthält keine Ligen."
}
$scoutSelectedYear = if ($SeasonStartYear -gt 0) {
    $SeasonStartYear
}
else {
    [int]$scoutEntries[0].season_start_year
}
$scoutNextYear = $scoutSelectedYear + 1
$scoutDatasetName = "scouting-$scoutSelectedYear-$scoutNextYear"
$scoutRawRoot = Join-Path $scoutProjectRoot "data\raw\$scoutDatasetName"
$scoutProcessedRoot = Join-Path $scoutProjectRoot "data\processed\$scoutDatasetName"

Push-Location $scoutProjectRoot
try {
    foreach ($scoutLeague in $scoutEntries) {
        $scoutSeasonName = if ($scoutLeague.calendar_year -eq $true) {
            "$scoutSelectedYear"
        }
        else {
            "$scoutSelectedYear/$scoutNextYear"
        }
        $scoutRawPath = Join-Path $scoutRawRoot "$($scoutLeague.slug).json"
        $scoutOutputPath = Join-Path $scoutProcessedRoot $scoutLeague.slug
        Write-Host ""
        Write-Host (
            "ScoutRAG: {0} ({1}, Saison {2})" -f
            $scoutLeague.competition_name,
            $scoutLeague.country,
            $scoutSeasonName
        ) -ForegroundColor Cyan

        if ($Download) {
            & $scoutDataCli api-football-fixture-sync `
                --league-id $scoutLeague.league_id `
                --season $scoutSelectedYear `
                --output $scoutRawPath `
                --request-budget $RequestBudgetPerLeague `
                --max-player-pages 80 `
                --throttle-seconds 0.25
            if ($LASTEXITCODE -ne 0) {
                throw "Download fehlgeschlagen: $($scoutLeague.slug)"
            }
        }

        if ($Build) {
            & $scoutDataCli api-football-fixture-build `
                --input $scoutRawPath `
                --output $scoutOutputPath `
                --competition-name $scoutLeague.competition_name `
                --round-prefix $scoutLeague.round_prefix `
                --season-name $scoutSeasonName `
                --include-same-league-postseason
            if ($LASTEXITCODE -ne 0) {
                throw "Build fehlgeschlagen: $($scoutLeague.slug)"
            }
        }
    }

    if ($Build) {
        & $scoutDataCli api-football-fixture-merge `
            --input $scoutProcessedRoot `
            --output (Join-Path $scoutProcessedRoot "combined") `
            --season $scoutSelectedYear
        if ($LASTEXITCODE -ne 0) {
            throw "Zusammenführen des Scouting-Datensatzes fehlgeschlagen."
        }
    }
}
finally {
    Pop-Location
}
