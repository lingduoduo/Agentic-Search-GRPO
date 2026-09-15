"""Measure whether a domain hint moves web results toward topic sources.

This module stays torch-free on purpose: importing ``src.model.post_training``
pulls torch in through package ``__init__`` side effects, which would drop this
module out of the torch-free CI job. The paired statistics therefore live in
the CLI, not here.

What it measures is source alignment, not answer quality. A result from
arxiv.org is not automatically a better answer than a good blog post.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from src.internal.tools.search import prepare_domain_query

TOP_K = 10


def host_matches(host: str, pattern: str) -> bool:
    """True when *host* is *pattern* or a subdomain of it.

    The boundary is a dot, so "notstanford.edu" does not match "stanford.edu"
    and neither does "stanford.edu.evil.com".
    """
    host = (host or "").strip().lower()
    pattern = (pattern or "").strip().lower()
    if not host or not pattern:
        return False
    return host == pattern or host.endswith("." + pattern)


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def authority_precision(urls: list[str], hosts: set[str]) -> float:
    """Share of *urls* served by one of the authority *hosts*.

    An empty list scores 0.0 rather than raising, so an arm that returned
    nothing is comparable with one that returned only off-topic results.
    """
    if not urls:
        return 0.0
    hits = sum(1 for url in urls if any(host_matches(_host_of(url), h) for h in hosts))
    return hits / len(urls)


def jaccard(a: list[str], b: list[str]) -> float:
    """Overlap of two result URL lists. Two empty arms are identical."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


DEFAULT_LABELS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "eval"
    / "domain_relevance_queries.json"
)


@dataclass(frozen=True)
class LabelledQuery:
    domain: str
    query: str
    authority_hosts: set[str] = field(default_factory=set)


def load_labels(path: str | Path = DEFAULT_LABELS_PATH) -> list[LabelledQuery]:
    """Read the label set, failing loudly before any provider call."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"label file is not valid JSON: {path}") from exc
    out: list[LabelledQuery] = []
    for domain, entries in raw.get("domains", {}).items():
        for entry in entries:
            out.append(
                LabelledQuery(
                    domain=domain,
                    query=entry["query"],
                    authority_hosts=set(entry["authority_hosts"]),
                )
            )
    return out


class DiskCache:
    """Cache provider responses so a re-run spends no quota."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> list[str] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: list[str]) -> None:
        self._path(key).write_text(json.dumps(value))


@dataclass(frozen=True)
class ArmResult:
    urls: list[str]
    empty: bool
    error: str | None = None


@dataclass(frozen=True)
class QueryOutcome:
    domain: str
    query: str
    general: ArmResult
    domain_arm: ArmResult
    delta: float
    jaccard: float
    excluded: bool


def cache_key(query: str, *, provider: str) -> str:
    """One place defines the key, so the CLI's dry run cannot drift from it."""
    return f"{provider}|{TOP_K}|{query}"


async def _run_arm(
    query: str, *, search_fn, cache: DiskCache, provider: str
) -> ArmResult:
    cached = cache.get(cache_key(query, provider=provider))
    if cached is not None:
        return ArmResult(urls=cached, empty=not cached)
    try:
        pages = await search_fn(query, provider=provider, page_size=TOP_K)
    except Exception as exc:  # provider error -> this arm is unusable
        return ArmResult(urls=[], empty=True, error=str(exc))
    urls = [
        p.url
        for p in pages
        if getattr(p, "url", None) and not getattr(p, "error", None)
    ]
    cache.set(cache_key(query, provider=provider), urls)
    return ArmResult(urls=urls, empty=not urls)


async def run_query(
    label: LabelledQuery, *, search_fn, cache: DiskCache, provider: str = "serpapi"
) -> QueryOutcome:
    """Run one labelled query through both arms and score them."""
    hinted = prepare_domain_query(label.query, label.domain)
    general = await _run_arm(
        label.query, search_fn=search_fn, cache=cache, provider=provider
    )
    domain_arm = await _run_arm(
        hinted, search_fn=search_fn, cache=cache, provider=provider
    )
    # A delta against a failed or empty arm measures that failure, not the
    # domain, so the query leaves the paired comparison.
    excluded = bool(
        general.error or domain_arm.error or general.empty or domain_arm.empty
    )
    delta = authority_precision(
        domain_arm.urls, label.authority_hosts
    ) - authority_precision(general.urls, label.authority_hosts)
    return QueryOutcome(
        domain=label.domain,
        query=label.query,
        general=general,
        domain_arm=domain_arm,
        delta=delta,
        jaccard=jaccard(general.urls, domain_arm.urls),
        excluded=excluded,
    )
