"""Deterministic validation for generated claims and cited facts."""

import re

from scoutrag.answering.models import (
    AllowedFact,
    EvidenceFactCatalog,
    GroundedAnswerDraft,
    GroundedClaim,
)
from scoutrag.domain.evidence import GroundingReport, RecommendationEvidencePack
from scoutrag.domain.player import PlayerSeasonProfile

_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?:\s?%)?")
_SAFE_GLUE = {
    "a",
    "an",
    "and",
    "at",
    "auf",
    "aus",
    "bei",
    "beträgt",
    "das",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "dieser",
    "diesem",
    "dieses",
    "einen",
    "einer",
    "einem",
    "eine",
    "er",
    "for",
    "für",
    "hat",
    "in",
    "im",
    "is",
    "its",
    "liegt",
    "mit",
    "of",
    "on",
    "per",
    "recorded",
    "season",
    "shows",
    "spielt",
    "the",
    "this",
    "verzeichnet",
    "von",
    "weist",
    "with",
    "year",
    "zu",
    "zur",
    "zum",
}


class GroundednessValidator:
    """Reject claims that escape the Evidence Pack's explicit fact boundary."""

    def validate(
        self,
        pack: RecommendationEvidencePack,
        catalog: EvidenceFactCatalog,
        draft: GroundedAnswerDraft,
        *,
        generator: str,
    ) -> GroundingReport:
        fact_index = catalog.by_id()
        candidates = {
            candidate.profile.player_id: candidate.profile for candidate in pack.candidates
        }
        violations: list[str] = []
        supported = 0
        cited_fact_ids: list[str] = []
        seen_players: set[str] = set()

        for claim_index, claim in enumerate(draft.claims, start=1):
            claim_violations = self._validate_claim(
                claim_index,
                claim,
                candidates,
                fact_index,
                seen_players,
            )
            violations.extend(claim_violations)
            if not claim_violations:
                supported += 1
            cited_fact_ids.extend(
                fact_id for fact_id in claim.fact_ids if fact_id in fact_index
            )

        count = len(draft.claims)
        unique_violations = list(dict.fromkeys(violations))
        return GroundingReport(
            validation_passed=not unique_violations,
            grounding_score=supported / count if count else 0,
            claim_count=count,
            supported_claim_count=supported,
            cited_fact_ids=list(dict.fromkeys(cited_fact_ids)),
            violations=unique_violations,
            generator=generator,
        )

    @staticmethod
    def _validate_claim(
        claim_index: int,
        claim: GroundedClaim,
        candidates: dict[str, PlayerSeasonProfile],
        fact_index: dict[str, AllowedFact],
        seen_players: set[str],
    ) -> list[str]:
        label = f"claim {claim_index}"
        violations: list[str] = []

        profile = candidates.get(claim.player_id)
        if profile is None:
            return [f"{label}: player_id {claim.player_id!r} is not a returned candidate"]
        if claim.player_id in seen_players:
            violations.append(f"{label}: duplicate claim for player_id {claim.player_id!r}")
        seen_players.add(claim.player_id)

        unknown_fact_ids = [fact_id for fact_id in claim.fact_ids if fact_id not in fact_index]
        if unknown_fact_ids:
            violations.append(f"{label}: unknown fact IDs {unknown_fact_ids}")
        cited_facts = [
            fact_index[fact_id] for fact_id in claim.fact_ids if fact_id in fact_index
        ]
        mismatched = [
            fact.fact_id for fact in cited_facts if fact.player_id != claim.player_id
        ]
        if mismatched:
            violations.append(f"{label}: facts belong to another player {mismatched}")
        cited_facts = [
            fact for fact in cited_facts if fact.player_id == claim.player_id
        ]
        if not cited_facts:
            violations.append(f"{label}: no valid supporting facts")
            return violations

        expected_name_fact = f"player:{_fact_key(claim.player_id)}:name"
        if expected_name_fact not in claim.fact_ids:
            violations.append(f"{label}: player identity fact is not cited")
        if profile.player_name.casefold() not in claim.text.casefold():
            violations.append(f"{label}: player name is missing")

        allowed_numbers = {
            _normalize_number(number)
            for fact in cited_facts
            for number in _NUMBER.findall(f"{fact.display_name} {fact.value}")
        }
        unsupported_numbers = [
            number
            for number in _NUMBER.findall(claim.text)
            if _normalize_number(number) not in allowed_numbers
        ]
        if unsupported_numbers:
            violations.append(f"{label}: unsupported numbers {unsupported_numbers}")

        allowed_tokens = _tokens(
            " ".join(
                [
                    profile.player_name,
                    *(
                        f"{fact.display_name} {fact.value}"
                        for fact in cited_facts
                    ),
                ]
            )
        )
        claim_tokens = _tokens(_NUMBER.sub(" ", claim.text))
        unsupported_tokens = sorted(
            token
            for token in claim_tokens
            if token not in _SAFE_GLUE
            and not _matches_supported_token(token, allowed_tokens)
        )
        if unsupported_tokens:
            violations.append(
                f"{label}: unsupported wording {unsupported_tokens}"
            )
        return violations


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(value) if len(token) > 1}


def _matches_supported_token(token: str, allowed: set[str]) -> bool:
    if token in allowed:
        return True
    if len(token) < 5:
        return False
    return any(
        len(candidate) >= 5
        and (token.startswith(candidate[:5]) or candidate.startswith(token[:5]))
        for candidate in allowed
    )


def _normalize_number(value: str) -> str:
    return value.replace(" ", "").replace(",", ".").removesuffix("%")


def _fact_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", value.casefold()).strip("-") or "unknown"
