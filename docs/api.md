# Phase 8 API and explainability dashboard

## One pipeline, three transport contracts

Phase 8 exposes the governed pipeline without duplicating retrieval logic:

```text
HTTP request
→ validated RetrievalRequest
→ GovernedRetrievalPipeline
→ RecommendationEvidencePack
→ complete or compact HTTP projection
```

The pipeline is loaded lazily on the first retrieval request. Missing local Parquet artifacts
produce an explicit `503 Service Unavailable` response with the data-build command rather than a
partially initialized search result.

## Run locally

Build or reuse the Phase 3 artifacts, then start FastAPI:

```powershell
scoutrag-data build
uvicorn scoutrag.main:app --reload
```

The default paths can be changed through the `SCOUTRAG_` environment variables documented in
`.env.example`. For an offline run with an existing dense model cache:

```powershell
$env:SCOUTRAG_LOCAL_FILES_ONLY = "true"
uvicorn scoutrag.main:app
```

Set `SCOUTRAG_ENABLE_DENSE_RETRIEVAL=false` for a model-free exact, structured, and BM25 run.

## Retrieve a complete Evidence Pack

```http
POST /api/v1/retrieve
Content-Type: application/json

{
  "query": "Zeige das Profil von Aleksandar Pavlović",
  "result_count": 10
}
```

The response is the canonical `RecommendationEvidencePack`: analyzed query, verdict, ranked
candidates, metric evidence, limitations, missing evidence, retrieval trace, and runtime metrics.
This endpoint is the preferred integration contract for evaluation and audit consumers.

## Compact search

```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "Bayern-Sechser mit mindestens 100 Minuten",
  "result_count": 5
}
```

The compact response retains the governance verdict, Evidence Quality Score, player provenance,
warnings, missing evidence, and total runtime. It is only a projection of `/retrieve`; it never
uses a second search implementation.

## Safe answer projection

```http
POST /api/v1/answer
Content-Type: application/json

{
  "evidence_pack": {
    "...": "the complete validated /retrieve response"
  }
}
```

The default uses `TemplateAnswerGenerator`, not an LLM. `SUFFICIENT` and `LIMITED` packs can cite
only their returned player IDs and stored requested metrics. `INSUFFICIENT`, `CONFLICTING`, and
`OUT_OF_SCOPE` packs abstain and cite no players.

Keeping `/answer` separate makes answer behavior testable without rerunning retrieval and makes
the optional Phase 10 backend replaceable. Every response now includes:

- `generation_mode`: `template`, `grounded_model`, or `safe_fallback`
- `grounding.validation_passed` and a non-probabilistic grounding score
- cited Fact IDs and claim counts
- validation violations and whether fallback was used

Enable the optional structured OpenAI adapter with:

```powershell
python -m pip install -e ".[llm]"
$env:OPENAI_API_KEY = "..."
$env:SCOUTRAG_ANSWER_MODE = "openai"
uvicorn scoutrag.main:app
```

Unsafe verdicts never invoke it. Invalid or unavailable model output is replaced by the
deterministic answer and marked `safe_fallback`; see `docs/answer-generation.md`.

## Dashboard

The dashboard at `/` calls `/api/v1/retrieve` and shows:

- query intent, applied filters, and strategies used
- governance verdict and all ten separately scored evidence factors
- ranked player cards with retrieval provenance
- season-specific metric evidence and source references
- limitations, missing evidence, and warnings
- candidate counts and per-stage timings from the retrieval trace
- the optional deterministic answer only after the Evidence Pack is visible

Static dashboard assets are served by the same application at `/assets`. The dashboard contains
no independent ranking or governance implementation.
