"""Assemble the LLM-free RecommendationEvidencePack after retrieval."""

from time import perf_counter

from scoutrag.domain.evidence import RecommendationEvidencePack, RuntimeMetrics
from scoutrag.governance.evidence import PlayerMetricEvidenceIndex
from scoutrag.ports.governance import RecommendationGovernor
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline


class GovernedRetrievalPipeline:
    """Run retrieval, evidence selection, and governance as separate stages."""

    def __init__(
        self,
        retrieval_pipeline: HybridRetrievalPipeline,
        evidence_index: PlayerMetricEvidenceIndex,
        governor: RecommendationGovernor,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.evidence_index = evidence_index
        self.governor = governor

    def search(self, query: str) -> RecommendationEvidencePack:
        started = perf_counter()
        retrieval_result = self.retrieval_pipeline.search(query)

        assembly_started = perf_counter()
        evidence = self.evidence_index.for_candidates(retrieval_result.candidates)
        evidence_assembly_ms = _elapsed_ms(assembly_started)

        governance_started = perf_counter()
        governance = self.governor.evaluate(
            retrieval_result.query_profile,
            retrieval_result.candidates,
            evidence,
        )
        governance_ms = _elapsed_ms(governance_started)
        stage_timings = retrieval_result.retrieval_trace.stage_timings_ms
        runtime = RuntimeMetrics(
            total_ms=_elapsed_ms(started),
            query_analysis_ms=stage_timings.get("query_analysis", 0),
            candidate_retrieval_ms=round(
                sum(value for name, value in stage_timings.items() if name.endswith("_retrieval")),
                3,
            ),
            fusion_ms=stage_timings.get("fusion", 0),
            reranking_ms=stage_timings.get("reranking", 0),
            governance_ms=governance_ms,
            evidence_assembly_ms=evidence_assembly_ms,
        )
        return RecommendationEvidencePack(
            query_profile=retrieval_result.query_profile,
            governance=governance,
            candidates=retrieval_result.candidates,
            retrieval_trace=retrieval_result.retrieval_trace,
            metric_evidence=evidence,
            limitations=list(governance.warnings),
            missing_evidence=list(governance.missing_evidence),
            runtime_metrics=runtime,
        )


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1_000, 3)
