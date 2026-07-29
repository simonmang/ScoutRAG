# Retrieval evaluation

`golden_queries.json` is the versioned Phase 5 relevance seed. It mixes Bayern identity/team
queries with statistically labeled trait queries from source-covered Leverkusen profiles. This is
intentional: partial Bayern data is suitable for exact metadata retrieval but not full-season
statistical superiority claims.

Run all Phase 5 ablations:

```powershell
scoutrag-evaluate --local-files-only
```

The command evaluates:

- A: BM25
- B: pretrained multilingual bi-encoder
- C: BM25 plus bi-encoder
- D: BM25, bi-encoder, and structured features
- H: complete Phase 4 hybrid including exact retrieval

Generated reports are written below `evaluation/results/` and are ignored by Git. The committed
documentation records the reviewed baseline numbers; reports can be regenerated after every
retrieval change.

## Metrics

| Metric | Scope | Meaning |
| --- | --- | --- |
| Candidate Recall | broad pool | share of judged relevant players available before reranking |
| Precision@K | final ranking | relevant results divided by the fixed cutoff K |
| Recall@K | final ranking | share of judged relevant players returned by K |
| MRR | final ranking | reciprocal rank of the first relevant result |
| nDCG@K | final ranking | graded ranking quality relative to the ideal order |
| Hit Rate@K | final ranking | share of queries with at least one relevant result by K |

Queries are macro-averaged, so a query with three judgments cannot dominate an exact lookup with
one judgment.

## Phase 5 baseline

| Variant | Candidate Recall | MRR | P@1 | R@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: BM25 | 1.000 | 0.950 | 0.900 | 1.000 | 0.847916 |
| B: pretrained bi-encoder | 1.000 | 1.000 | 1.000 | 1.000 | 0.877445 |
| C: BM25 + bi-encoder | 1.000 | 1.000 | 1.000 | 1.000 | 0.876800 |
| D: C + structured | 1.000 | 1.000 | 1.000 | 1.000 | 0.892273 |
| H: full Phase 4 hybrid | 1.000 | 1.000 | 1.000 | 1.000 | **0.900637** |

The complete hybrid improves nDCG@5 by `0.052721` over BM25 on this seed. The result is useful as
a regression baseline, not as a claim of production accuracy:

- only 10 queries and 21 positive judgments exist
- labels are rule-assisted and not independently annotated
- hard filters make Candidate Recall comparatively easy
- opponent profiles have partial source coverage
- some relevant groups contain only Leverkusen players because they are the only source-covered
  season profiles

## Phase 6 reranking evaluation

`scoutrag-rerank-evaluate` performs broad hybrid retrieval once per query and gives the exact same
fused candidate list to the baseline and cross-encoder orderings. Candidate Recall therefore
cannot change during this comparison.

```powershell
scoutrag-rerank-evaluate --local-files-only
scoutrag-rerank-evaluate --backend onnx --onnx-file-name onnx/model.onnx
```

The command writes full per-query reports to the ignored `evaluation/results/` directory. The
reviewed aggregate is committed as `reranking_summary.json`.

| Ranking | MRR | HR@1 | HR@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Fused baseline | 1.000 | 1.000 | 1.000 | 0.900637 |
| Multilingual cross-encoder | 0.900 | 0.800 | 1.000 | 0.846097 |
| Delta | -0.100 | -0.200 | 0.000 | -0.054540 |

The cross-encoder is intentionally opt-in because this pretrained, non-football model makes the
seed ranking worse. This is evidence of domain shift, not a reason to hide the result. A later
football-specific reranker can reuse the same report contract and must beat the fused baseline
before becoming the default.

Warm local CPU mean latency was 824.084 ms with Torch and 310.008 ms with the model's FP32 ONNX
artifact. These values exclude loading and warm-up and should not be generalized to other
hardware.
