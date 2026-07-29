"""Load and resolve versioned football retrieval training specifications."""

import json
from pathlib import Path

from scoutrag.training.models import BiEncoderTrainingDataset, MinedTrainingDataset


def load_training_dataset(path: Path) -> BiEncoderTrainingDataset:
    """Load query specifications with strict schema validation."""
    return BiEncoderTrainingDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_mined_dataset(path: Path) -> MinedTrainingDataset:
    """Load fully resolved training tuples with strict schema validation."""
    return MinedTrainingDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
