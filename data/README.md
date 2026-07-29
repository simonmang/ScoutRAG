# ScoutRAG data workspace

Downloaded StatsBomb Open Data and generated Parquet datasets are intentionally excluded from
Git. This keeps the repository small and makes every artifact reproducible from source.

```text
data/
├── raw/        # StatsBomb JSON hierarchy
└── processed/  # normalized and aggregated Parquet artifacts
```

The default Phase 3 reference dataset is 1. Bundesliga 2023/2024:

```powershell
scoutrag-data download --competition-id 9 --season-id 281
scoutrag-data build --competition-id 9 --season-id 281
```

The available competition entry contains Bayer Leverkusen's 34 matches rather than every
Bundesliga match. The pipeline is not club-specific: Bayern Munich is preferred in examples, but
its profiles in this partition cover only two matches. ScoutRAG therefore retains their measured
features while correctly withholding position percentiles. Leverkusen is the full-season
reference only because of the available upstream test data.

StatsBomb must be credited when publishing analysis based on their open data. See the
[StatsBomb Open Data repository](https://github.com/statsbomb/open-data) for its current terms.

Phase 4 creates an additional ignored `processed/bundesliga-2023-2024/dense_index.json` artifact.
It contains the selected model name, season-safe profile keys, and document embeddings. Rebuild it
after changing the player profiles or embedding model:

```powershell
scoutrag-retrieve "Zeige das Profil von Joshua Kimmich" --rebuild-dense-index
```
