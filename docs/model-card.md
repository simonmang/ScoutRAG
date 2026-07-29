---
language:
  - de
  - en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - football
  - information-retrieval
  - hard-negatives
  - portfolio
---

# ScoutRAG Football Bi-Encoder

## Model summary

ScoutRAG Football Bi-Encoder is an experimental multilingual retrieval model fine-tuned from
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. It encodes a scouting query and a
deterministic `PlayerSeasonProfile` projection independently for fast candidate recall.

The checkpoint is a portfolio research artifact, not a calibrated scouting decision system. Its
similarity score expresses retrieval relevance only. It must not be presented as confidence,
evidence quality, player quality, or probability of a correct recommendation.

## Intended use

- broad candidate retrieval for German and English football scouting queries
- offline comparison with the pretrained multilingual baseline
- hard-negative and language-variant retrieval experiments
- input to ScoutRAG fusion, reranking, and Evidence Governance

It is not intended for transfer valuation, match prediction, automated recruitment decisions, or
use without the downstream governance layer.

## Training data

The committed `evaluation/bi_encoder_training_queries.json` contains 32 rule-authored query
specifications tied to Bundesliga 2023/2024 typed profiles:

| Split | Examples | Languages |
| --- | ---: | --- |
| Training | 20 | German and English |
| Validation | 12 | German and English |

Each resolved tuple contains:

1. a normalized query,
2. one positive player profile,
3. one embedding-similar, same-position Hard Negative that fails a target metric, team, or
   identity constraint,
4. one different-position Easy Negative.

Metric labels use source-comparable position percentiles. Bayern Munich examples use identity,
team, and position only because the open-data source provides partial Bayern coverage.

## Training procedure

| Parameter | Value |
| --- | --- |
| Loss | `MultipleNegativesRankingLoss` |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | `2e-5` |
| Warmup ratio | 0.1 |
| Seed | 42 |
| Hardware | local CPU |
| Optimization steps | 9 |
| Training runtime | 93.8 seconds |
| Dataset fingerprint | `c95041674b7bfc34c27695ffe36699839b7bedd9720dfccc4338f04283842fd1` |

The loss receives the positive, constrained Hard Negative, and Easy Negative explicitly while
also using non-duplicate in-batch negatives.

## Evaluation

### Full Golden retrieval

| Model | Candidate Recall | MRR | nDCG@5 |
| --- | ---: | ---: | ---: |
| Pretrained baseline | 1.000 | **1.000** | **0.877445** |
| Football fine-tuned | 1.000 | 0.950 | 0.859452 |
| Delta | 0.000 | -0.050 | -0.017993 |

### Held-out difficult negatives and language variants

| Model | Hard-negative accuracy | Pairwise MRR | Positive-hard margin | DE | EN | Bilingual stability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained baseline | 0.250000 | 0.625000 | -0.018547 | 0.166667 | 0.333333 | 0.166667 |
| Football fine-tuned | **0.416667** | **0.708333** | **0.010259** | **0.333333** | **0.500000** | **0.333333** |

The checkpoint learned the targeted difficult distinctions and changed the average positive-hard
margin from negative to positive. It simultaneously reduced ranking quality on the ten-query
Golden seed. ScoutRAG therefore keeps it opt-in; the pretrained encoder remains the default until
a larger independently labeled set demonstrates a clear overall improvement.

## Reproduction

```powershell
python -m pip install -e ".[training]"
scoutrag-bi-encoder mine --local-files-only
scoutrag-bi-encoder train --local-files-only
scoutrag-bi-encoder evaluate --local-files-only
```

The generated model is stored below `models/` and is ignored by Git because it is approximately
470 MB. A SHA-256 fingerprint of the resolved training dataset is written next to every local
checkpoint.

## Limitations and risks

- only 20 training and 12 validation queries
- validation phrasings are held out, but player concepts can overlap across splits
- rule-assisted labels rather than independent multi-annotator judgments
- evaluation covers one competition-season source partition
- partial opponent coverage, including Bayern Munich
- no fairness, transfer-market, injury, age, salary, or tactical-system assessment
- no calibrated probability or decision confidence

All recommendations still require ScoutRAG's structured filters, cross-encoder activation policy,
typed evidence, and `RecommendationGovernor`.
