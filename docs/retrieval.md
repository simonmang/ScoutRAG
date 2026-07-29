# Phase 4 retrieval design

ScoutRAG uses hybrid retrieval because names, hard statistical constraints, rare football terms,
and semantic role descriptions require different recall mechanisms. No retrieval score is reused
as evidence quality.

## Query analysis

`RuleBasedQueryAnalyzer` supports inspectable German and English patterns. It extracts:

- `QueryIntent`
- named players
- team, competition, and season filters
- position groups
- requested traits and their metric names
- minimum minutes
- requested result count

Explicit team, competition, season, position, and minute constraints are hard filters shared by
all retrievers. For example, `von Bayern München` cannot be overridden by a high dense similarity
from another club.

## Independent recall strategies

| Strategy | Primary signal | Typical strength |
| --- | --- | --- |
| Exact | names, teams, competitions, seasons, positions | deterministic lookup |
| Structured | typed filters, per-90 features, percentiles | statistical constraints |
| BM25 | token frequency and inverse document frequency | rare terms and names |
| Dense | query/profile embeddings and cosine similarity | paraphrases and semantic similarity |

The structured retriever uses position percentiles where the Phase 3 evidence rules permit them.
If a profile has no valid percentile, it uses a min-max-normalized structured feature only for
recall. That fallback does not manufacture a percentile or improve the profile's Data Quality
Score.

The BM25 implementation is local and dependency-free. It uses Unicode-aware tokenization and
standard configurable `k1=1.5`, `b=0.75` defaults.

## Dense baseline

The baseline is
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
It maps German and English text to 384-dimensional vectors. Queries and player documents are
encoded through separate bi-encoder calls and compared with cosine similarity.

The index contains:

- schema version
- exact model identifier
- `(player_id, competition_name, season_name)` keys
- embedding dimension
- document embedding matrix

Loading rejects an index created for another model or profile dataset. The model and generated
index are not committed to Git.

## Fusion

Each strategy's returned scores are independently min-max normalized to `0..1`. When every score
from one strategy is equal, each receives `1.0`; the strategy has identified candidates but
provides no internal separation.

Default weighted fusion:

```text
fused_score =
0.30 × dense
+ 0.25 × sparse
+ 0.30 × structured
+ 0.15 × exact
```

Weights are configurable and must be non-negative and sum to one. A candidate missing from one
strategy receives no contribution from that strategy. This rewards cross-strategy agreement while
preserving broad recall.

## Broad recall and trace

The default candidate pool is 40 profiles before reranking, even when the requested result count
is 10. The current `NoOpPlayerReranker` preserves fused order and makes the Phase 6 replacement
boundary explicit.

Every run records:

- query ID and intent
- strategies used
- candidate count per strategy
- count before and after reranking
- hard filters
- per-stage timing
- normalized per-candidate strategy scores
- every strategy that retrieved each candidate

## Reproduce

```powershell
python -m pip install -e ".[dev,retrieval]"
scoutrag-data build --competition-id 9 --season-id 281
scoutrag-retrieve "pressingstarker Sechser von Bayern München mit mindestens 100 Minuten"
```

The first command using dense retrieval downloads the model. The first search builds the local
profile index; subsequent runs reuse it. Use `--disable-dense` for an exact/structured/BM25
baseline or `--rebuild-dense-index` after changing profiles or the model.
