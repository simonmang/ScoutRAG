"""Lazy Sentence Transformers fine-tuning behind a small application boundary."""

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scoutrag.training.models import MinedTrainingDataset


@dataclass(frozen=True, slots=True)
class BiEncoderTrainingConfig:
    """Reproducible CPU/GPU-neutral fine-tuning parameters."""

    base_model_name: str
    epochs: float = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_steps: int = -1
    seed: int = 42
    use_cpu: bool = True
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    """Small serializable record written next to the trained model."""

    model_name: str
    source_dataset_version: str
    dataset_fingerprint: str
    training_examples: int
    output_path: str
    config: dict[str, Any]


class SentenceTransformerBiEncoderTrainer:
    """Fine-tune a multilingual encoder with explicit hard/easy negatives."""

    def train(
        self,
        dataset: MinedTrainingDataset,
        output_path: Path,
        config: BiEncoderTrainingConfig,
    ) -> TrainingRunSummary:
        examples = [example for example in dataset.examples if example.split == "train"]
        if not examples:
            raise ValueError("mined dataset contains no training examples")
        output_path.mkdir(parents=True, exist_ok=True)

        sentence_transformers = _optional_module("sentence_transformers", "training")
        sentence_transformer_losses = _optional_module(
            "sentence_transformers.losses",
            "training",
        )
        datasets = _optional_module("datasets", "training")
        model_path = _resolve_model_path(
            config.base_model_name,
            local_files_only=config.local_files_only,
        )
        model = sentence_transformers.SentenceTransformer(
            model_path,
            local_files_only=config.local_files_only,
        )
        train_dataset = datasets.Dataset.from_dict(
            {
                "query": [example.query_text for example in examples],
                "positive": [example.positive_text for example in examples],
                "hard_negative": [example.hard_negative_text for example in examples],
                "easy_negative": [example.easy_negative_text for example in examples],
            }
        )
        loss = sentence_transformer_losses.MultipleNegativesRankingLoss(model)
        training_args = sentence_transformers.SentenceTransformerTrainingArguments(
            output_dir=str(output_path / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=config.epochs,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            batch_sampler="no_duplicates",
            save_strategy="no",
            eval_strategy="no",
            logging_strategy="steps",
            logging_steps=1,
            report_to="none",
            use_cpu=config.use_cpu,
            dataloader_pin_memory=not config.use_cpu,
            optim="adamw_torch",
            seed=config.seed,
            data_seed=config.seed,
        )
        trainer = sentence_transformers.SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=loss,
        )
        trainer.train()
        model.save_pretrained(
            str(output_path),
            model_name="ScoutRAG Football Bi-Encoder",
            create_model_card=False,
            safe_serialization=True,
        )
        _set_tokenizer_compatibility(output_path)

        summary = TrainingRunSummary(
            model_name="ScoutRAG Football Bi-Encoder",
            source_dataset_version=dataset.source_dataset_version,
            dataset_fingerprint=_dataset_fingerprint(dataset),
            training_examples=len(examples),
            output_path=str(output_path),
            config=asdict(config),
        )
        (output_path / "scoutrag_training.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "README.md").write_text(
            _generated_model_card(summary),
            encoding="utf-8",
        )
        return summary


def _resolve_model_path(model_name: str, *, local_files_only: bool) -> str:
    if not local_files_only or Path(model_name).exists():
        return model_name
    huggingface_hub = _optional_module("huggingface_hub", "retrieval")
    return str(
        huggingface_hub.snapshot_download(
            repo_id=model_name,
            local_files_only=True,
        )
    )


def _optional_module(name: str, extra: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"{name} is required for this operation; install with 'pip install -e \".[{extra}]\"'"
        ) from error


def _dataset_fingerprint(dataset: MinedTrainingDataset) -> str:
    payload = dataset.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_tokenizer_compatibility(output_path: Path) -> None:
    """Persist the Transformers compatibility flag requested for saved fast tokenizers."""
    config_path = output_path / "tokenizer_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["fix_mistral_regex"] = True
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _generated_model_card(summary: TrainingRunSummary) -> str:
    return f"""---
language:
- de
- en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
---

# ScoutRAG Football Bi-Encoder

Fine-tuned from `{summary.config["base_model_name"]}` for typed football player retrieval.

- Dataset: `{summary.source_dataset_version}`
- Dataset fingerprint: `{summary.dataset_fingerprint}`
- Training examples: {summary.training_examples}
- Loss: MultipleNegativesRankingLoss
- Negatives: one constrained hard negative and one different-position easy negative per query

This local artifact is an experimental portfolio model. See the repository model card for
evaluation results, source-coverage limitations, intended use, and non-goals.
"""
