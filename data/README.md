# ScoutRAG data workspace

Downloaded StatsBomb Open Data and generated Parquet datasets are intentionally excluded from
Git. This keeps the repository small and makes every artifact reproducible from source.

```text
data/
├── raw/        # StatsBomb JSON hierarchy
└── processed/  # normalized and aggregated Parquet artifacts
```

The default Phase 2 reference dataset is 1. Bundesliga 2023/2024:

```powershell
scoutrag-data download --competition-id 9 --season-id 281
scoutrag-data build --competition-id 9 --season-id 281
```

The available competition entry contains Bayer Leverkusen's 34 matches rather than every
Bundesliga match. Treat non-Leverkusen player aggregates as partial source coverage. ScoutRAG
records this limitation in the generated validation report.

StatsBomb must be credited when publishing analysis based on their open data. See the
[StatsBomb Open Data repository](https://github.com/statsbomb/open-data) for its current terms.
