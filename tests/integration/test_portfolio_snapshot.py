"""The committed model-free snapshot must serve a real Bayern demo query."""

import hashlib
import json
from pathlib import Path

from scoutrag.domain.evidence import EvidenceVerdict
from scoutrag.governance.evidence import load_metric_evidence
from scoutrag.governance.factory import build_governed_pipeline
from scoutrag.retrieval.common import load_profiles

SNAPSHOT_ROOT = Path("data/processed/bundesliga-2023-2024")


def test_portfolio_snapshot_serves_governed_kimmich_lookup() -> None:
    manifest = json.loads((SNAPSHOT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for artifact_name in (
        "player_season_profiles.parquet",
        "player_metric_evidence.parquet",
    ):
        digest = hashlib.sha256((SNAPSHOT_ROOT / artifact_name).read_bytes()).hexdigest()
        assert digest == manifest["artifacts"][artifact_name]["sha256"]

    profiles = load_profiles(SNAPSHOT_ROOT / "player_season_profiles.parquet")
    evidence = load_metric_evidence(SNAPSHOT_ROOT / "player_metric_evidence.parquet")

    assert len(profiles) == 373
    assert len(evidence) == 11_563
    kimmich = next(profile for profile in profiles if profile.player_id == "5579")
    assert kimmich.player_name == "Joshua Kimmich"
    assert kimmich.team_name == "Bayern Munich"
    assert kimmich.structured_features["source_coverage_ratio"] == 0.0588

    pack = build_governed_pipeline(profiles, evidence).search(
        "Zeige das Profil von Joshua Kimmich",
        result_count=3,
    )

    assert pack.candidates[0].profile.player_id == "5579"
    assert pack.governance.verdict is EvidenceVerdict.LIMITED
    assert pack.retrieval_trace.strategies_used == ["exact", "sparse"]
