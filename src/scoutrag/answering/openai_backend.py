"""Optional OpenAI Responses API adapter for schema-constrained drafts."""

from importlib import import_module
from typing import Any

from scoutrag.answering.models import GroundedAnswerDraft


class OpenAIResponsesBackend:
    """Generate structured claims; the local validator remains authoritative."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-terra",
        max_output_tokens: int = 800,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model

    def generate_draft(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> GroundedAnswerDraft:
        client = self._client or self._load_client()
        response = client.responses.parse(
            model=self._model,
            instructions=instructions,
            input=input_text,
            text_format=GroundedAnswerDraft,
            max_output_tokens=self._max_output_tokens,
            store=False,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI response did not contain parsed output")
        return GroundedAnswerDraft.model_validate(parsed)

    @staticmethod
    def _load_client() -> Any:
        try:
            openai = import_module("openai")
        except ImportError as exc:
            raise RuntimeError("OpenAI answer mode requires `pip install -e '.[llm]'`") from exc
        return openai.OpenAI()
