"""Phase 10 answer safety, grounding, and optional backend tests."""

from pathlib import Path
from types import SimpleNamespace

from scoutrag.answering.facts import build_fact_catalog
from scoutrag.answering.generator import GroundedAnswerGenerator
from scoutrag.answering.models import GroundedAnswerDraft, GroundedClaim
from scoutrag.answering.openai_backend import OpenAIResponsesBackend
from scoutrag.domain.evidence import (
    EvidenceVerdict,
    GenerationMode,
    RecommendationGovernance,
)
from scoutrag.evaluation.answer_grounding import load_answer_grounding_dataset

CASES_PATH = Path("evaluation/answer_grounding_cases.json")


class DraftBackend:
    def __init__(self, draft: GroundedAnswerDraft, *, fail: bool = False) -> None:
        self.draft = draft
        self.fail = fail
        self.called = False

    @property
    def model_name(self) -> str:
        return "fake-grounded-model"

    def generate_draft(self, *, instructions: str, input_text: str) -> GroundedAnswerDraft:
        assert "do not calculate" in instructions
        assert '"allowed_player_ids"' in input_text
        self.called = True
        if self.fail:
            raise TimeoutError("synthetic timeout")
        return self.draft


def test_fact_catalog_preserves_values_and_source_references() -> None:
    pack = load_answer_grounding_dataset(CASES_PATH).evidence_pack

    catalog = build_fact_catalog(pack).by_id()

    normalized = catalog["metric:5579:pressures_per_90:0:normalized"]
    percentile = catalog["metric:5579:pressures_per_90:0:percentile"]
    assert normalized.value == "14.3"
    assert percentile.value == "91"
    assert normalized.source_reference == (
        "statsbomb:competition=9:season=281:player=5579"
    )


def test_supported_claim_is_returned_with_fact_audit() -> None:
    dataset = load_answer_grounding_dataset(CASES_PATH)
    draft = dataset.cases[0].draft
    backend = DraftBackend(draft)

    answer = GroundedAnswerGenerator(backend).generate(dataset.evidence_pack)

    assert answer.generation_mode is GenerationMode.GROUNDED_MODEL
    assert answer.grounding.validation_passed
    assert answer.grounding.grounding_score == 1
    assert "metric:5579:pressures_per_90:0:normalized" in (
        answer.grounding.cited_fact_ids
    )
    assert answer.cited_player_ids == ["5579"]
    assert "14.3" in answer.text


def test_fabricated_number_is_blocked_and_template_replaces_it() -> None:
    dataset = load_answer_grounding_dataset(CASES_PATH)
    fabricated = next(
        case for case in dataset.cases if case.case_id == "fabricated-statistic"
    )

    answer = GroundedAnswerGenerator(DraftBackend(fabricated.draft)).generate(
        dataset.evidence_pack
    )

    assert answer.generation_mode is GenerationMode.SAFE_FALLBACK
    assert answer.grounding.fallback_used
    assert not answer.grounding.validation_passed
    assert any(
        "unsupported numbers ['99']" in item
        for item in answer.grounding.violations
    )
    assert "99" not in answer.text
    assert "14.3" in answer.text


def test_unsupported_tactical_inference_is_blocked() -> None:
    dataset = load_answer_grounding_dataset(CASES_PATH)
    tactical = next(
        case
        for case in dataset.cases
        if case.case_id == "unsupported-tactical-inference"
    )

    answer = GroundedAnswerGenerator(DraftBackend(tactical.draft)).generate(
        dataset.evidence_pack
    )

    assert answer.generation_mode is GenerationMode.SAFE_FALLBACK
    assert any("unsupported wording" in item for item in answer.grounding.violations)
    assert "intelligently" not in answer.text


def test_governance_abstention_never_calls_model() -> None:
    dataset = load_answer_grounding_dataset(CASES_PATH)
    backend = DraftBackend(dataset.cases[0].draft)
    governance = RecommendationGovernance(
        verdict=EvidenceVerdict.INSUFFICIENT,
        evidence_quality_score=0.2,
        reasons=["Requested evidence is unavailable."],
        missing_evidence=["pressures_per_90"],
    )
    pack = dataset.evidence_pack.model_copy(update={"governance": governance})

    answer = GroundedAnswerGenerator(backend).generate(pack)

    assert not backend.called
    assert answer.generation_mode is GenerationMode.TEMPLATE
    assert answer.cited_player_ids == []
    assert "Keine belastbare Spielerempfehlung" in answer.text


def test_backend_failure_is_audited_and_falls_back() -> None:
    dataset = load_answer_grounding_dataset(CASES_PATH)
    backend = DraftBackend(dataset.cases[0].draft, fail=True)

    answer = GroundedAnswerGenerator(backend).generate(dataset.evidence_pack)

    assert answer.generation_mode is GenerationMode.SAFE_FALLBACK
    assert answer.grounding.fallback_used
    assert answer.grounding.violations == [
        "generation backend failed: TimeoutError"
    ]


def test_limited_model_answer_retains_limitations() -> None:
    dataset = load_answer_grounding_dataset(CASES_PATH)
    limited = next(
        case
        for case in dataset.cases
        if case.case_id == "supported-team-claim-limited"
    )
    governance = dataset.evidence_pack.governance.model_copy(
        update={"verdict": EvidenceVerdict.LIMITED}
    )
    pack = dataset.evidence_pack.model_copy(update={"governance": governance})

    answer = GroundedAnswerGenerator(DraftBackend(limited.draft)).generate(pack)

    assert answer.generation_mode is GenerationMode.GROUNDED_MODEL
    assert "Ergebnisse mit Einschränkungen" in answer.text
    assert "Single-season evidence only." in answer.text


def test_openai_adapter_uses_responses_structured_output() -> None:
    draft = GroundedAnswerDraft(
        claims=[
            GroundedClaim(
                player_id="5579",
                text="Joshua Kimmich spielt für Bayern Munich.",
                fact_ids=["player:5579:name", "player:5579:team"],
            )
        ]
    )
    calls: list[dict[str, object]] = []

    class FakeResponses:
        def parse(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(output_parsed=draft)

    client = SimpleNamespace(responses=FakeResponses())
    backend = OpenAIResponsesBackend(
        model="test-model",
        max_output_tokens=321,
        client=client,
    )

    result = backend.generate_draft(instructions="rules", input_text="facts")

    assert result == draft
    assert calls == [
        {
            "model": "test-model",
            "instructions": "rules",
            "input": "facts",
            "text_format": GroundedAnswerDraft,
            "max_output_tokens": 321,
            "store": False,
        }
    ]
