# ScoutRAG data workspace

Raw provider responses and all generated Parquet datasets are intentionally excluded from Git.
API-Football's terms prohibit redistributing their data to third parties, so nothing derived from
it is ever committed — every real artifact must be regenerated locally from your own API key.

```text
data/
├── raw/        # downloaded provider JSON (StatsBomb, API-Football) — all ignored
└── processed/  # normalized and aggregated Parquet artifacts — all ignored except the CI fixture
```

The only committed data is a fully synthetic three-player fixture used solely by Docker and CI to
prove the packaged application starts and serves a governed result:

- `data/processed/synthetic-ci-fixture/player_season_profiles.parquet`
- `data/processed/synthetic-ci-fixture/player_metric_evidence.parquet`

Every name and statistic in it is invented by `scripts/build_synthetic_ci_fixture.py`; it is never
presented as a real dataset. Regenerate it with:

```powershell
python scripts/build_synthetic_ci_fixture.py
```

## API-Football source (primary)

API-Football supplies current, multi-league, multi-season player statistics. Live responses are
cached under `data/raw/` and remain excluded from Git. The dashboard never calls the provider
directly; a deliberate local sync creates typed `PlayerSeasonProfile` and `PlayerMetricEvidence`
artifacts used by retrieval. A free API-Football plan is enough to generate one league/season for
a first look; the multi-league dataset used in development needs a paid plan for its request
volume.

As a smaller single-league example, one competition-season can be reconstructed directly from its
fixture packages:

```powershell
scoutrag-data api-football-fixture-sync --league-id 78 --season 2024
scoutrag-data api-football-fixture-build --league-id 78 --season 2024
```

For Bundesliga 2024/2025 this produces 481 player-season profiles, 18,102 typed metric-evidence
records, and 39 metric definitions across all 18 clubs. The two relegation fixtures remain in the
ignored raw artifact but are excluded from league percentiles. `/players` contributes identity
metadata only; fixture player statistics are authoritative for minutes, historical teams, and
performance. The `sync_scouting_universe.ps1` workflow below runs this same fixture-sync and
fixture-build pair once per configured league, so the actual dataset the dashboard and `.env`
point to by default is the combined multi-league output, not this single-league example.

The expanded scouting universe is configured in `config/scouting_leagues.json` and can be
reproduced for an explicit season with:

```powershell
.\sync_scouting_universe.ps1 -SeasonStartYear 2025
.\sync_scouting_universe.ps1 -SeasonStartYear 2024
```

The current quality-gated artifact contains 12,713 competition-season profiles for 11,924 unique
players, 477,131 metric-evidence records, and 229,661 per-match player performances from 24
accepted competitions. The 2024/2025 artifact contains 12,116 profiles and 226,880 player-match
performances from 23 accepted competitions. Individual league builds keep their own comparison
groups and percentiles; excluded datasets and reasons remain in the combined manifest.

`position_group` is refined beyond API-Football's coarse goalkeeper/defender/midfielder/forward
tag where the cached fixture lineups support it: a player's starting-formation grid slots across
the season (`scoutrag.data.position_inference`) can separate a fullback from a centre-back, or a
holding midfielder from an attacking one. Refinement only narrows a coarse group and requires a
clear majority across enough starts in a back-four formation; every other profile keeps the
coarse provider value instead of a guessed one.

The optional local history build keeps seasons separate:

```powershell
scoutrag-data api-football-history-build `
  --input data/processed/scouting-2025-2026/combined `
  --input data/processed/scouting-2024-2025/combined `
  --output data/processed/scouting-history-2024-2026
```

It persists stable identities, club-season stints, bounded recent-form snapshots, individual
match performances, and descriptive season trends. These generated artifacts remain ignored by
Git; only schemas, code, tests, catalog, and reproduction commands belong in the repository.

### Optional Wikidata biography enrichment

Wikidata's structured data is [CC0 (public domain)](https://www.wikidata.org/wiki/Wikidata:Licensing),
unlike API-Football or Transfermarkt, so it can be used freely. It adds only what API-Football does
not have — national-team caps, footedness, career honours, and club history predating the tracked
seasons — and never duplicates a field ScoutRAG already tracks (name, birth date/place,
nationality, height, weight, current club, season stats):

```powershell
scoutrag-data wikidata-enrich --input data/processed/scouting-2025-2026/combined
```

A player is matched by exact name, then the match is only kept when the Wikidata entity's date of
birth agrees exactly with the value already trusted from API-Football. An unresolved or ambiguous
name is left unenriched rather than guessed. On a random sample of 20 players across all 24
leagues (not just star names), about 75-80% matched with a confirmed birth date; a global check
found only ~1,800 footballers on all of Wikidata have footedness recorded at all, so that specific
field stays sparse regardless of match rate. Responses are cached under `data/raw/wikidata/`, and
requests are throttled by default as a courtesy to the shared public endpoint.

### Optional API-Football career events

Transfers, official trophies, and injury history come from the same already-licensed API-Football
subscription used for statistics — no second provider, no name/date-of-birth matching, since every
record already shares the pipeline's numeric player ID:

```powershell
scoutrag-data api-football-career-events --input data/processed/scouting-2025-2026/combined
```

Each player needs three requests (`/transfers`, `/trophies`, `/sidelined`), so the full 12,713-player
catalog exceeds a single day's quota on most plans. The command stops cleanly once the request
budget or the provider's own daily quota is reached and reports how many players remain; rerunning
the exact same command on a later day resumes for free for every already-completed player, since
`ApiFootballClient`'s existing content-addressed cache under `data/raw/api_football/` answers those
without a new network call. `type` on a transfer is the provider's raw fee text (for example
`"€ 8.5M"`, `"Free"`, `"Loan"`, `"N/A"`) and is stored as-is rather than parsed into a number.

Phase 4 creates an additional ignored `dense_index.json` artifact per profile set. It contains the
selected model name, season-safe profile keys, and document embeddings. Rebuild it after changing
the player profiles or embedding model:

```powershell
scoutrag-retrieve "Show the profile of Joshua Kimmich" \
  --profiles data/processed/scouting-2025-2026/combined/player_season_profiles.parquet \
  --dense-index data/processed/scouting-2025-2026/combined/dense_index.json \
  --rebuild-dense-index
```

## Optional StatsBomb data pipeline (secondary)

StatsBomb Open Data is an independently working alternative ingestion path into the same typed
schema — not part of the demo or the default configuration, kept to show ScoutRAG is not tied to
one provider:

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
[StatsBomb Open Data repository](https://github.com/statsbomb/open-data) for its current terms —
unlike API-Football, StatsBomb's Open Data license does permit this kind of use.

The sources remain distinct. API-Football supplies aggregated fields such as appearances,
minutes, shots, goals, passes, tackles, interceptions and duels. StatsBomb supplies event-level
evidence and supports derived metrics such as pressures and progressive actions. ScoutRAG does
not fill an unavailable provider metric with an invented value.
