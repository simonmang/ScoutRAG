"""Architectural ports for replaceable ScoutRAG components."""

from scoutrag.ports.answering import AnswerGenerator
from scoutrag.ports.governance import RecommendationGovernor
from scoutrag.ports.query_analysis import QueryAnalyzer
from scoutrag.ports.reranking import PlayerReranker
from scoutrag.ports.retrieval import CandidateRetriever, PlayerRetriever, RetrievalFusion

__all__ = [
    "AnswerGenerator",
    "CandidateRetriever",
    "PlayerReranker",
    "PlayerRetriever",
    "QueryAnalyzer",
    "RecommendationGovernor",
    "RetrievalFusion",
]
