"""Optional Wikidata biography enrichment, cached and confidence-gated.

Wikidata's structured data is CC0 (public domain), unlike API-Football or
Transfermarkt, so it can be committed and reused freely. It is never used to
override or duplicate fields ScoutRAG already tracks from API-Football (name,
birth date/place, nationality, height, weight, current club, season stats).
It only ever adds what those sources do not have: national-team caps,
footedness, career honours, and club history predating the tracked seasons.

A player is matched to a Wikidata entity by exact name, then confirmed only
when the entity's date of birth agrees exactly with the already-trusted
API-Football value. An unresolved or ambiguous name (no candidate, or several
candidates with the same name but no date-of-birth agreement) is left
unenriched rather than guessed - consistent with every other optional
evidence source in this project.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scoutrag.domain.player import PlayerExternalContext

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
_USER_AGENT = "ScoutRAG/1.0 (portfolio project; contact via GitHub issues)"
_HUMAN = "wd:Q5"
_FOOTBALLER_OCCUPATION = "wd:Q937857"
_NATIONAL_TEAM_CLASS = "wd:Q6979593"
_DATE_OF_BIRTH = "wdt:P569"
_FOOTEDNESS = "wdt:P8006"
_AWARD_RECEIVED = "wdt:P166"
_NUMBER_OF_MATCHES = "pq:P1350"

DEFAULT_BATCH_SIZE = 60
DEFAULT_THROTTLE_SECONDS = 2.0


class WikidataEnrichmentError(RuntimeError):
    """Base error for Wikidata enrichment failures."""


_RETRYABLE_HTTP_STATUSES = {429, 502, 503, 504}


class WikidataTransientError(WikidataEnrichmentError):
    """A rate limit or transient server error from the shared public endpoint.

    Covers 429 (Too Many Requests) and the common transient gateway errors
    (502/503/504) a busy public endpoint can return; all are safe to retry.
    """

    def __init__(self, status: int, retry_after: float | None) -> None:
        super().__init__(f"Wikidata SPARQL endpoint returned transient HTTP {status}")
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class WikidataCandidate:
    """One player's known identity, as already trusted from API-Football."""

    player_id: str
    player_name: str
    date_of_birth: date | None
    current_team_names: frozenset[str] = frozenset()


SparqlTransport = Callable[[str], Any]


def urllib_sparql_transport(query: str, *, timeout: float = 30.0) -> Any:
    """Default transport: a single SPARQL query over HTTPS, JSON results."""
    from urllib.parse import urlencode

    url = f"{WIKIDATA_SPARQL_URL}?{urlencode({'query': query, 'format': 'json'})}"
    request = Request(
        url, headers={"Accept": "application/sparql-results+json", "User-Agent": _USER_AGENT}
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code in _RETRYABLE_HTTP_STATUSES:
            retry_after = error.headers.get("Retry-After") if error.headers else None
            raise WikidataTransientError(
                error.code, float(retry_after) if retry_after else None
            ) from error
        raise WikidataEnrichmentError(f"Wikidata SPARQL request failed: {error}") from error
    except TimeoutError as error:
        # A slow response from a shared public endpoint is expected sometimes, not fatal.
        raise WikidataTransientError(status=0, retry_after=None) from error
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise WikidataTransientError(status=0, retry_after=None) from error
        raise WikidataEnrichmentError(f"Wikidata SPARQL request failed: {error}") from error


DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0


class CachedSparqlClient:
    """Content-addressed local cache around one SPARQL transport, like API-Football's.

    A cache hit never counts against the retry/throttle budget, so an interrupted or
    rate-limited run can simply be restarted: every already-fetched batch resolves
    instantly from disk and only the remaining ones touch the network again.
    """

    def __init__(
        self,
        *,
        transport: SparqlTransport = urllib_sparql_transport,
        cache_dir: Path | None = None,
        throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._transport = transport
        self._cache_dir = cache_dir
        self._throttle_seconds = throttle_seconds
        self._max_retries = max_retries
        self._last_request_at: float | None = None

    def query(self, sparql: str) -> Any:
        cache_path = self._cache_path(sparql)
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        result = self._query_with_retries(sparql)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result), encoding="utf-8")
        return result

    def _query_with_retries(self, sparql: str) -> Any:
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                return self._transport(sparql)
            except WikidataTransientError as error:
                if attempt == self._max_retries:
                    raise
                wait_seconds = error.retry_after or DEFAULT_RETRY_BACKOFF_SECONDS * (2**attempt)
                time.sleep(wait_seconds)
        raise AssertionError("unreachable")  # pragma: no cover

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self._throttle_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _cache_path(self, sparql: str) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(sparql.encode("utf-8")).hexdigest()
        return self._cache_dir / "wikidata" / f"{digest}.json"


def enrich_players(
    candidates: Sequence[WikidataCandidate],
    *,
    client: CachedSparqlClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[PlayerExternalContext]:
    """Match, confirm, and enrich a batch of players. Unmatched players are skipped."""

    confirmed = _match_confirmed_entities(candidates, client=client, batch_size=batch_size)
    if not confirmed:
        return []

    footedness = _fetch_footedness(confirmed.values(), client=client, batch_size=batch_size)
    national_teams = _fetch_national_team_caps(
        confirmed.values(), client=client, batch_size=batch_size
    )
    honours = _fetch_honours(confirmed.values(), client=client, batch_size=batch_size)
    clubs = _fetch_earlier_clubs(confirmed.values(), client=client, batch_size=batch_size)
    current_team_names = {
        candidate.player_id: candidate.current_team_names for candidate in candidates
    }

    results: list[PlayerExternalContext] = []
    for player_id, wikidata_id in confirmed.items():
        team_name, caps = national_teams.get(wikidata_id, (None, None))
        # Wikidata's P54 club history includes the current club too; excluding whatever
        # ScoutRAG already tracks as "current" keeps this field genuinely additive.
        already_tracked = current_team_names.get(player_id, frozenset())
        earlier_clubs = clubs.get(wikidata_id, set()) - already_tracked
        results.append(
            PlayerExternalContext(
                player_id=player_id,
                wikidata_id=wikidata_id,
                footedness=footedness.get(wikidata_id),
                national_team_name=team_name,
                national_team_caps=caps,
                honours=sorted(honours.get(wikidata_id, set())),
                earlier_clubs=sorted(earlier_clubs),
                source_reference=f"wikidata:{wikidata_id}",
            )
        )
    return sorted(results, key=lambda item: item.player_id)


def _match_confirmed_entities(
    candidates: Sequence[WikidataCandidate],
    *,
    client: CachedSparqlClient,
    batch_size: int,
) -> dict[str, str]:
    by_name: defaultdict[str, list[WikidataCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.date_of_birth is not None:
            by_name[candidate.player_name].append(candidate)
    names = sorted(by_name)
    confirmed: dict[str, str] = {}
    for batch in _chunks(names, batch_size):
        query = _identity_query(batch)
        raw = client.query(query)
        rows_by_name: defaultdict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for binding in raw["results"]["bindings"]:
            name = binding["playerLabel"]["value"]
            entity_uri = binding["player"]["value"]
            wikidata_id = entity_uri.rsplit("/", 1)[-1]
            dob = binding.get("dob", {}).get("value")
            rows_by_name[name].append((wikidata_id, dob[:10] if dob else None))
        for name in batch:
            for candidate in by_name[name]:
                if candidate.date_of_birth is None:
                    continue  # Filtered when building by_name; re-checked for mypy narrowing.
                exact_dob = candidate.date_of_birth.isoformat()
                agreeing = {
                    wikidata_id
                    for wikidata_id, dob in rows_by_name.get(name, [])
                    if dob == exact_dob
                }
                if len(agreeing) == 1:
                    confirmed[candidate.player_id] = next(iter(agreeing))
    return confirmed


def _identity_query(names: Iterable[str]) -> str:
    values = " ".join(f"{json.dumps(name)}@en" for name in names)
    return f"""
SELECT ?player ?playerLabel ?dob WHERE {{
  VALUES ?playerLabel {{ {values} }}
  ?player rdfs:label ?playerLabel;
          wdt:P31 {_HUMAN};
          wdt:P106 {_FOOTBALLER_OCCUPATION}.
  OPTIONAL {{ ?player {_DATE_OF_BIRTH} ?dob. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""


def _fetch_footedness(
    wikidata_ids: Iterable[str],
    *,
    client: CachedSparqlClient,
    batch_size: int,
) -> dict[str, str]:
    by_entity = _run_entity_query(
        wikidata_ids,
        client=client,
        batch_size=batch_size,
        select="?player ?footednessLabel",
        body=f"?player {_FOOTEDNESS} ?footedness.",
    )
    return {
        wikidata_id: binding["footednessLabel"]["value"]
        for wikidata_id, bindings in by_entity.items()
        for binding in bindings[:1]
        if "footednessLabel" in binding
    }


def _fetch_national_team_caps(
    wikidata_ids: Iterable[str],
    *,
    client: CachedSparqlClient,
    batch_size: int,
) -> dict[str, tuple[str, int]]:
    by_entity = _run_entity_query(
        wikidata_ids,
        client=client,
        batch_size=batch_size,
        select="?player ?teamLabel ?caps",
        body=f"""
        ?player p:P54 ?stmt.
        ?stmt ps:P54 ?team.
        ?team wdt:P31/wdt:P279* {_NATIONAL_TEAM_CLASS}.
        OPTIONAL {{ ?stmt {_NUMBER_OF_MATCHES} ?caps. }}
        """,
    )
    result: dict[str, tuple[str, int]] = {}
    for wikidata_id, bindings in by_entity.items():
        best_team, best_caps = None, -1
        for binding in bindings:
            if "teamLabel" not in binding:
                continue  # The label service failed to resolve a label for this entity.
            caps = int(float(binding["caps"]["value"])) if "caps" in binding else 0
            if caps > best_caps:
                best_team, best_caps = binding["teamLabel"]["value"], caps
        if best_team is not None:
            result[wikidata_id] = (best_team, max(best_caps, 0))
    return result


def _fetch_honours(
    wikidata_ids: Iterable[str],
    *,
    client: CachedSparqlClient,
    batch_size: int,
) -> dict[str, set[str]]:
    by_entity = _run_entity_query(
        wikidata_ids,
        client=client,
        batch_size=batch_size,
        select="?player ?awardLabel",
        body=f"?player {_AWARD_RECEIVED} ?award.",
    )
    return {
        wikidata_id: {
            binding["awardLabel"]["value"] for binding in bindings if "awardLabel" in binding
        }
        for wikidata_id, bindings in by_entity.items()
    }


def _fetch_earlier_clubs(
    wikidata_ids: Iterable[str],
    *,
    client: CachedSparqlClient,
    batch_size: int,
) -> dict[str, set[str]]:
    by_entity = _run_entity_query(
        wikidata_ids,
        client=client,
        batch_size=batch_size,
        select="?player ?teamLabel",
        body=f"""
        ?player p:P54 ?stmt.
        ?stmt ps:P54 ?team.
        FILTER NOT EXISTS {{ ?team wdt:P31/wdt:P279* {_NATIONAL_TEAM_CLASS}. }}
        """,
    )
    return {
        wikidata_id: {
            binding["teamLabel"]["value"] for binding in bindings if "teamLabel" in binding
        }
        for wikidata_id, bindings in by_entity.items()
    }


def _run_entity_query(
    wikidata_ids: Iterable[str],
    *,
    client: CachedSparqlClient,
    batch_size: int,
    select: str,
    body: str,
) -> dict[str, list[dict[str, Any]]]:
    unique_ids = sorted(set(wikidata_ids))
    by_entity: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in _chunks(unique_ids, batch_size):
        values = " ".join(f"wd:{wikidata_id}" for wikidata_id in batch)
        query = f"""
SELECT {select} WHERE {{
  VALUES ?player {{ {values} }}
  {body}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""
        raw = client.query(query)
        for binding in raw["results"]["bindings"]:
            entity_uri = binding["player"]["value"]
            wikidata_id = entity_uri.rsplit("/", 1)[-1]
            by_entity[wikidata_id].append(binding)
    return dict(by_entity)


def _chunks(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
