"""Raw StatsBomb Open Data downloading and filesystem access."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

from scoutrag.data.models import CompetitionSeason, DownloadSummary

STATSBOMB_OPEN_DATA_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


class StatsBombDataError(ValueError):
    """Raised when source data is absent or structurally invalid."""


def _as_object_list(value: object, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise StatsBombDataError(f"{source} must contain a JSON array of objects")
    return cast(list[dict[str, Any]], value)


def _read_object_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise StatsBombDataError(f"required StatsBomb file does not exist: {path}")
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise StatsBombDataError(f"invalid JSON in {path}: {exc}") from exc
    return _as_object_list(value, source=str(path))


class StatsBombOpenDataReader:
    """Read the official StatsBomb Open Data directory hierarchy."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def competition(self, competition_id: int, season_id: int) -> CompetitionSeason:
        records = _read_object_list(self.root / "competitions.json")
        for record in records:
            if (
                record.get("competition_id") == competition_id
                and record.get("season_id") == season_id
            ):
                return CompetitionSeason(
                    competition_id=competition_id,
                    season_id=season_id,
                    country_name=str(record["country_name"]),
                    competition_name=str(record["competition_name"]),
                    season_name=str(record["season_name"]),
                    competition_gender=(
                        str(record["competition_gender"])
                        if record.get("competition_gender") is not None
                        else None
                    ),
                    source_reference=(
                        f"statsbomb:competitions/{competition_id}/seasons/{season_id}"
                    ),
                )
        raise StatsBombDataError(
            f"competition_id={competition_id}, season_id={season_id} is not available"
        )

    def matches(self, competition_id: int, season_id: int) -> list[dict[str, Any]]:
        return _read_object_list(self.root / "matches" / str(competition_id) / f"{season_id}.json")

    def events(self, match_id: int) -> list[dict[str, Any]]:
        return _read_object_list(self.root / "events" / f"{match_id}.json")

    def lineups(self, match_id: int) -> list[dict[str, Any]]:
        return _read_object_list(self.root / "lineups" / f"{match_id}.json")


class UrlLibJsonFetcher:
    """Small HTTP adapter kept injectable so tests never need the network."""

    def __call__(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "ScoutRAG/0.2"})
        with urlopen(request, timeout=60) as response:
            return cast(bytes, response.read())


class StatsBombOpenDataDownloader:
    """Download one competition-season without cloning the complete repository."""

    def __init__(
        self,
        fetch: Callable[[str], bytes] | None = None,
        base_url: str = STATSBOMB_OPEN_DATA_BASE_URL,
    ) -> None:
        self.fetch = fetch or UrlLibJsonFetcher()
        self.base_url = base_url.rstrip("/")

    def download(
        self,
        competition_id: int,
        season_id: int,
        output_root: Path,
        *,
        match_limit: int | None = None,
    ) -> DownloadSummary:
        files_downloaded = 0
        competitions_bytes = self._fetch_json_bytes("competitions.json")
        files_downloaded += self._write(output_root / "competitions.json", competitions_bytes)

        matches_path = f"matches/{competition_id}/{season_id}.json"
        matches_bytes = self._fetch_json_bytes(matches_path)
        matches = _as_object_list(cast(object, json.loads(matches_bytes)), source=matches_path)
        if match_limit is not None:
            if match_limit < 1:
                raise ValueError("match_limit must be at least one")
            matches = matches[:match_limit]
            matches_bytes = json.dumps(
                matches,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
        files_downloaded += self._write(output_root / matches_path, matches_bytes)
        match_ids = [int(match["match_id"]) for match in matches]

        for match_id in match_ids:
            for category in ("events", "lineups"):
                relative_path = f"{category}/{match_id}.json"
                payload = self._fetch_json_bytes(relative_path)
                files_downloaded += self._write(output_root / relative_path, payload)

        return DownloadSummary(
            competition_id=competition_id,
            season_id=season_id,
            match_ids=match_ids,
            files_downloaded=files_downloaded,
            output_directory=str(output_root.resolve()),
        )

    def _fetch_json_bytes(self, relative_path: str) -> bytes:
        payload = self.fetch(f"{self.base_url}/{relative_path}")
        try:
            json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StatsBombDataError(
                f"downloaded content is not valid JSON: {relative_path}"
            ) from exc
        return payload

    @staticmethod
    def _write(path: Path, payload: bytes) -> int:
        if path.is_file() and path.read_bytes() == payload:
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_bytes(payload)
        temporary_path.replace(path)
        return 1
