"""FastAPI smoke test for the Phase 1 boundary."""

from fastapi.testclient import TestClient

from scoutrag.answering.generator import GroundedAnswerGenerator
from scoutrag.config import Settings
from scoutrag.main import create_app


def test_health_endpoint() -> None:
    app = create_app(Settings(environment="test"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ScoutRAG",
        "version": "0.10.0",
        "environment": "test",
    }


def test_openai_answer_mode_is_composed_without_eager_api_call() -> None:
    app = create_app(Settings(environment="test", answer_mode="openai"))

    assert isinstance(app.state.answer_generator, GroundedAnswerGenerator)
