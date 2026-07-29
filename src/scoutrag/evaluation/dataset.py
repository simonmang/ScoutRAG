"""Golden-dataset loading with strict schema validation."""

import json
from pathlib import Path

from scoutrag.evaluation.models import GoldenDataset


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load a versioned JSON file and reject contract drift."""
    return GoldenDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
