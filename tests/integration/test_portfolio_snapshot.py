"""The committed synthetic CI/Docker fixture must serve a real governed lookup.

This fixture is entirely invented (see scripts/build_synthetic_ci_fixture.py) so it can be
committed and baked into a public Docker image without redistributing any licensed provider
data. It exists only to prove the packaged application works end to end, not as a demo dataset.
"""

from pathlib import Path

from scoutrag.domain.evidence import EvidenceVerdict
from scoutrag.governance.evidence import load_metric_evidence
from scoutrag.governance.factory import build_governed_pipeline
from scoutrag.retrieval.common import load_profiles

FIXTURE_ROOT = Path("data/processed/synthetic-ci-fixture")


def test_synthetic_ci_fixture_serves_governed_lookup() -> None:
    profiles = load_profiles(FIXTURE_ROOT / "player_season_profiles.parquet")
    evidence = load_metric_evidence(FIXTURE_ROOT / "player_metric_evidence.parquet")

    assert len(profiles) == 3
    musterfeld = next(profile for profile in profiles if profile.player_id == "synthetic:1")
    assert musterfeld.player_name == "Alex Musterfeld"
    assert musterfeld.team_name == "FC Beispielhausen"

    pack = build_governed_pipeline(profiles, evidence).search(
        "Show the profile of Alex Musterfeld",
        result_count=3,
    )

    assert pack.candidates[0].profile.player_id == "synthetic:1"
    assert pack.governance.verdict is EvidenceVerdict.LIMITED
    assert pack.retrieval_trace.strategies_used == ["exact", "sparse"]
