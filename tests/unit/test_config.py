"""Security-sensitive configuration behavior."""

import pytest

from scoutrag.config import Settings


def test_api_football_key_uses_unprefixed_environment_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_FOOTBALL_KEY", "local-test-secret")

    settings = Settings(_env_file=None)

    assert settings.api_football_key is not None
    assert settings.api_football_key.get_secret_value() == "local-test-secret"
    assert "local-test-secret" not in repr(settings)
