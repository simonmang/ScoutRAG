# Changelog

## 1.0.0

ScoutRAG's first portfolio-ready release.

### Added

- typed season profiles, metric evidence, query profiles, traces, and Evidence Packs
- exact, structured, BM25, and multilingual dense retrieval with normalized fusion
- broad recall, cross-encoder reranking boundary, and before/after evaluation
- rule-based Evidence Governance with transparent abstention
- governed retrieve, search, and answer APIs plus an explainability dashboard
- bilingual football bi-encoder training and Hard-Negative evaluation
- fact-bound structured answer generation and local groundedness validation
- hallucination benchmark with safe template fallback
- compact source-attributed demo snapshot, Docker image, and Render Blueprint

### Safety and limitations

- cosine, fusion, and reranker scores are never presented as recommendation confidence
- model-backed answer generation is optional and disabled in the public demo
- Bayern Munich data covers only two matches in the available StatsBomb partition
- Leverkusen is the full-season statistical reference because of upstream data availability
- benchmark results are small regression seeds, not calibrated production guarantees
