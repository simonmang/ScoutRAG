# ScoutRAG architecture

## Design goal

ScoutRAG should retrieve suitable football players while making the limits of its evidence as
visible as its recommendations. The system is decomposed so that improvements to recall cannot
silently change governance, and generation cannot invent new domain facts.

## Component diagram

```mermaid
flowchart LR
    subgraph Input
        Q[User query]
        QA[QueryAnalyzer]
        QP[(QueryProfile)]
        Q --> QA --> QP
    end

    subgraph Recall["Recall layer"]
        EX[Exact]
        ST[Structured]
        BM[BM25 / TF-IDF]
        DE[Dense Bi-Encoder]
        FU[Normalized Fusion]
        QP --> EX & ST & BM & DE
        EX & ST & BM & DE --> FU
    end

    subgraph Ranking["Ranking layer"]
        CA[(Broad candidate pool)]
        CE[PlayerReranker]
        FU --> CA --> CE
    end

    subgraph Governance["Evidence governance"]
        ME[(Metric evidence)]
        GO[RecommendationGovernor]
        CE --> GO
        ME --> GO
        GO --> EP[(RecommendationEvidencePack)]
    end

    subgraph Consumers
        RA[Retrieve API]
        DA[Dashboard]
        AN[AnswerGenerator]
        AA[Answer API]
        EP --> RA
        EP --> DA
        EP --> AN --> AA
    end
```

## Request sequence

```mermaid
sequenceDiagram
    actor User
    participant Analyzer as QueryAnalyzer
    participant Retrievers as CandidateRetrievers
    participant Fusion as RetrievalFusion
    participant Reranker as PlayerReranker
    participant Governor as RecommendationGovernor
    participant Pack as EvidencePack
    participant Generator as AnswerGenerator

    User->>Analyzer: natural-language query
    Analyzer-->>Retrievers: QueryProfile
    par Independent broad recall
        Retrievers->>Retrievers: exact retrieval
        Retrievers->>Retrievers: structured retrieval
        Retrievers->>Retrievers: sparse retrieval
        Retrievers->>Retrievers: dense retrieval
    end
    Retrievers-->>Fusion: candidates + per-strategy traces
    Fusion-->>Reranker: normalized, deduplicated broad pool
    Reranker-->>Governor: relevance-ranked candidates
    Governor-->>Pack: verdict, reasons, warnings, missing evidence
    Pack-->>User: auditable retrieval result
    opt Natural-language answer requested
        Pack->>Generator: immutable grounded input
        Generator-->>User: verdict-aware answer
    end
```

## Phase 1 ports

The `PipelineComponents` dependency graph exposes six independent roles:

1. `QueryAnalyzer` creates a deterministic `QueryProfile`.
2. One or more `PlayerRetriever` instances perform broad recall.
3. `RetrievalFusion` normalizes and combines strategy-specific results.
4. `PlayerReranker` produces the relevance order.
5. `RecommendationGovernor` assesses evidence sufficiency.
6. `AnswerGenerator`, when configured, renders only the completed evidence pack.

`CandidateRetriever` is a semantic alias of `PlayerRetriever` in the recall stage. A later
composition root can inject exact, structured, sparse, and dense adapters without changing domain
or API contracts.

## Invariants

### Season integrity

A `PlayerSeasonProfile` represents one player, one competition, and one season. Cross-season
aggregation must be explicit in a future service and must never mutate or silently merge the
source profiles.

### Provenance

Every `PlayerMetricEvidence` carries a `season_id`, comparison group, and `source_reference`.
Evidence-pack validation rejects evidence grouped under a player that is not among the returned
candidates.

### Score semantics

| Value | Meaning | Must not be called |
| --- | --- | --- |
| dense/sparse/structured/exact score | strategy-specific retrieval signal | recommendation confidence |
| fused score | normalized recall-stage combination | evidence quality |
| reranker score | pairwise query/profile relevance | calibrated probability |
| evidence quality score | rule-based sufficiency summary | probability of correctness |

### Governed generation

The future `AnswerGenerator` receives only a `RecommendationEvidencePack`. Expected behavior:

- `SUFFICIENT`: a complete evidence-backed recommendation is allowed.
- `LIMITED`: results are allowed, with prominent limitations.
- `INSUFFICIENT`: abstain from a confident recommendation and explain missing evidence.
- `CONFLICTING`: surface the conflicting signals.
- `OUT_OF_SCOPE`: explain supported ScoutRAG capabilities.

Generation cannot calculate new statistics, add candidates, fill missing values, invent seasons,
or create unsupported tactical claims.

## Deployment boundary in Phase 1

Only `GET /health` is exposed. Retrieval endpoints are intentionally deferred until their
underlying pipeline stages exist:

| Future endpoint | Contract |
| --- | --- |
| `POST /api/v1/retrieve` | returns `RecommendationEvidencePack` |
| `POST /api/v1/answer` | renders a supplied or newly produced evidence pack |
| `POST /api/v1/search` | compact facade over the same retrieval pipeline |

This prevents an HTTP facade from becoming a hidden monolithic implementation.

