"""Raw reader and injectable downloader tests."""

from pathlib import Path

import pytest

from scoutrag.data.statsbomb import (
    StatsBombDataError,
    StatsBombOpenDataDownloader,
    StatsBombOpenDataReader,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "statsbomb"


def test_reader_selects_exact_competition_season() -> None:
    reader = StatsBombOpenDataReader(FIXTURE_ROOT)

    competition = reader.competition(9, 281)

    assert competition.competition_name == "1. Bundesliga"
    assert competition.season_name == "2023/2024"
    assert len(reader.matches(9, 281)) == 1


def test_reader_rejects_unknown_season() -> None:
    with pytest.raises(StatsBombDataError, match="not available"):
        StatsBombOpenDataReader(FIXTURE_ROOT).competition(9, 999)


def test_downloader_uses_official_hierarchy_without_network(tmp_path: Path) -> None:
    payloads = {
        "competitions.json": (FIXTURE_ROOT / "competitions.json").read_bytes(),
        "matches/9/281.json": (FIXTURE_ROOT / "matches" / "9" / "281.json").read_bytes(),
        "events/1001.json": (FIXTURE_ROOT / "events" / "1001.json").read_bytes(),
        "lineups/1001.json": (FIXTURE_ROOT / "lineups" / "1001.json").read_bytes(),
    }

    def fake_fetch(url: str) -> bytes:
        return payloads[url.removeprefix("https://example.test/")]

    result = StatsBombOpenDataDownloader(
        fetch=fake_fetch,
        base_url="https://example.test",
    ).download(9, 281, tmp_path)

    assert result.match_ids == [1001]
    assert result.files_downloaded == 4
    assert (tmp_path / "events" / "1001.json").is_file()
    assert (tmp_path / "lineups" / "1001.json").is_file()
