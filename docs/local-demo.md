# Local dashboard and portfolio demo

ScoutRAG ships a compact, model-free local demonstration. It contains only the generated
`PlayerSeasonProfile` and `PlayerMetricEvidence` artifacts required by the application—no raw
events, downloaded embedding model, API credential, or public web service.

## Demo data

The committed Bundesliga 2023/2024 snapshot contains:

- 373 season-specific player profiles
- 11,563 source-linked metric-evidence records
- 34 StatsBomb Open Data matches
- 21 Bayern Munich profiles from two observed matches
- full-season Leverkusen coverage from the available upstream partition

This is intentionally not described as a complete Bundesliga dataset. Governance exposes low
minutes, incomplete club coverage, unavailable percentiles, and unsuitable comparison groups.
Bayern examples demonstrate transparent `LIMITED` behavior; statistically sufficient rankings
can only use source-covered comparison groups.

The snapshot hashes and warnings are recorded in:

- `data/processed/bundesliga-2023-2024/manifest.json`
- `data/processed/bundesliga-2023-2024/validation_report.json`

## Windows one-click start

Double-click `start_dashboard.cmd` in the repository root. It calls the PowerShell launcher,
checks Python and both demo artifacts, selects the local model-free configuration, opens the
browser, and starts FastAPI.

The direct PowerShell equivalent is:

```powershell
.\start_dashboard.ps1
```

Useful options:

```powershell
.\start_dashboard.ps1 -Check
.\start_dashboard.ps1 -NoBrowser -Port 8080
.\start_dashboard.ps1 -EnableDenseRetrieval
```

`-EnableDenseRetrieval` expects the configured sentence-transformer model to exist in the local
cache. The standard local demo deliberately avoids that dependency.

## Docker

Build and run the same model-free image used by the deployment blueprint:

```powershell
docker build --tag scoutrag:1.0.0 .
docker run --rm --publish 8000:8000 scoutrag:1.0.0
```

Open `http://127.0.0.1:8000`. The image runs as an unprivileged user and defaults to:

```text
SCOUTRAG_ENABLE_DENSE_RETRIEVAL=false
SCOUTRAG_ANSWER_MODE=template
SCOUTRAG_LOCAL_FILES_ONLY=true
```

Exact, structured, and BM25 retrieval, fusion, governance, API, dashboard, trace, and safe answers
remain active. Dense retrieval and external generation stay optional because a portfolio demo
should not require model downloads, API credentials, or paid calls.
