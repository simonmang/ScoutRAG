"""Football-specific bi-encoder training and evaluation."""

from scoutrag.training.models import (
    BiEncoderTrainingDataset,
    MinedTrainingDataset,
    RetrievalTrainingQuery,
)

__all__ = [
    "BiEncoderTrainingDataset",
    "MinedTrainingDataset",
    "RetrievalTrainingQuery",
]
