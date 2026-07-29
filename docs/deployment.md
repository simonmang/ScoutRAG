# Deployment and portfolio demo

ScoutRAG ships a compact, model-free demonstration image. It contains only the generated
`PlayerSeasonProfile` and `PlayerMetricEvidence` artifacts required by the application—no raw
events and no downloaded embedding model.

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

## Render Blueprint

`render.yaml` describes one free Docker web service in the Frankfurt region with `/health` as its
HTTP health check. Auto-deployment is disabled for copies created through the public Deploy to
Render button.

[Deploy ScoutRAG to Render](https://render.com/deploy?repo=https://github.com/simonmang/ScoutRAG)

After reviewing the Blueprint, approve it in your Render account. Render builds the checked-in
Dockerfile and assigns an `onrender.com` URL. Free instances can sleep while inactive, so the
first request after inactivity may take longer.

Do not enable `SCOUTRAG_ANSWER_MODE=openai` on a public demo unless an explicit spending limit,
request throttling, and a securely stored `OPENAI_API_KEY` are configured.
