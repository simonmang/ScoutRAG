# Local dashboard demo

ScoutRAG runs entirely locally against a dataset you generate yourself with your own
[API-Football](https://www.api-football.com/) key — see [data notes](../data/README.md) for the
full generation workflow. Nothing derived from that provider is ever committed or shipped, since
their terms prohibit redistributing the data to third parties.

## Generate a dataset

```powershell
# 1. Copy .env.example to .env and add your key:
#    API_FOOTBALL_KEY=...
# 2. Generate at least one league/season (a free plan is enough for one league):
.\sync_scouting_universe.ps1 -Groups top5 -SeasonStartYear 2025
.\sync_scouting_universe.ps1 -Build -Groups top5 -SeasonStartYear 2025
```

`sync_scouting_universe.ps1` without arguments builds the full 26-league, multi-season catalog,
which needs a paid plan for its request volume. See [data notes](../data/README.md) for the
complete set of options.

## Windows one-click start

Double-click `start_dashboard.cmd` in the repository root. It calls the PowerShell launcher,
checks Python and both generated data artifacts, opens the browser, and starts FastAPI.

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
cache; the first run downloads it. `start_dashboard_ai.cmd` additionally loads `OPENAI_API_KEY`
from `.env` into the running process and starts in grounded-answer mode.

## Docker

The Docker image ships with a small, entirely invented three-player fixture
(`scripts/build_synthetic_ci_fixture.py`) baked in, used only to prove the packaged application
starts and serves a governed result end to end — it is not real football data and is not a
demo dataset:

```powershell
docker build --tag scoutrag:1.0.1 .
docker run --rm --publish 8000:8000 scoutrag:1.0.1
```

Open `http://127.0.0.1:8000`. The image runs as an unprivileged user and defaults to:

```text
SCOUTRAG_ENABLE_DENSE_RETRIEVAL=false
SCOUTRAG_ANSWER_MODE=template
SCOUTRAG_LOCAL_FILES_ONLY=true
```

To run the container against your own generated dataset instead, mount it and override the
profile/evidence paths, for example:

```powershell
docker run --rm --publish 8000:8000 `
  --volume "${PWD}/data/processed/scouting-2025-2026:/app/data/processed/scouting-2025-2026:ro" `
  --env SCOUTRAG_PROFILES_PATH=data/processed/scouting-2025-2026/combined/player_season_profiles.parquet `
  --env SCOUTRAG_METRIC_EVIDENCE_PATH=data/processed/scouting-2025-2026/combined/player_metric_evidence.parquet `
  scoutrag:1.0.1
```
