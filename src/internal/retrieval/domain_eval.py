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
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

from src.internal.tools.search import (
    DOMAIN_REGISTRY,
    SearchPage,
    prepare_domain_query,
    search_category,
)

TOP_K = 10

# The label file predates the search-source taxonomy: its keys are the topic
# domains measured in #588, not source categories. Their hints are frozen here
# so a rerun reproduces that measurement (and its cache keys) instead of
# silently reading health/gaming/... as web aliases with an empty hint.
HISTORICAL_TOPIC_HINTS: dict[str, str] = {
    "general": "",
    "resource": "resources",
    "social_media": "social media",
    "finance": "finance",
    "academic": "academic research",
    "legal": "law",
    "health": "health",
    "business": "business",
    "security": "cybersecurity",
    "ip": "intellectual property",
    "code": "programming",
    "energy": "energy",
    "environment": "environment",
    "agriculture": "agriculture",
    "travel": "travel",
    "film": "film",
    "gaming": "video games",
}


def historical_topic_query(query: str, topic: str) -> str:
    """Append an archived topic hint; ``general`` leaves the query unchanged."""
    if topic not in HISTORICAL_TOPIC_HINTS:
        raise ValueError(
            f"{topic!r} is not a historical topic label; this evaluation "
            "measures the archived topic hints, not search-source categories"
        )
    hint = HISTORICAL_TOPIC_HINTS[topic]
    return f"{query} {hint}" if hint and query.strip() else query


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
    if raw.get("taxonomy") is not None:
        raise ValueError(
            f"label file {path} is a {raw['taxonomy']!r} file; "
            "load it with load_source_labels"
        )
    out: list[LabelledQuery] = []
    for domain, entries in raw.get("domains", {}).items():
        if domain not in HISTORICAL_TOPIC_HINTS:
            raise ValueError(
                f"label file {path} uses {domain!r}, which is not a historical "
                "topic label"
            )
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
    seconds: float | None = None  # live call time; None when served from cache


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
    started = time.perf_counter()
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
    return ArmResult(urls=urls, empty=not urls, seconds=time.perf_counter() - started)


async def run_query(
    label: LabelledQuery, *, search_fn, cache: DiskCache, provider: str = "serpapi"
) -> QueryOutcome:
    """Run one labelled query through both arms and score them."""
    hinted = historical_topic_query(label.query, label.domain)
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


SOURCE_TAXONOMY_MARKER = "source_categories"
DEFAULT_SOURCE_LABELS_PATH = DEFAULT_LABELS_PATH.with_name(
    "source_category_queries.json"
)


@dataclass(frozen=True)
class SourceLabelledQuery:
    domain: str
    query: str
    authority_hosts: set[str] = field(default_factory=set)


def load_source_labels(
    path: str | Path = DEFAULT_SOURCE_LABELS_PATH,
) -> list[SourceLabelledQuery]:
    """Read source-category labels; only categories with source_sites."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"label file is not valid JSON: {path}") from exc
    if raw.get("taxonomy") != SOURCE_TAXONOMY_MARKER:
        raise ValueError(f"label file {path} lacks taxonomy={SOURCE_TAXONOMY_MARKER!r}")
    out: list[SourceLabelledQuery] = []
    for domain, entries in raw.get("domains", {}).items():
        if domain not in DOMAIN_REGISTRY or not DOMAIN_REGISTRY[domain].source_sites:
            raise ValueError(f"{domain!r} has no source_sites; it cannot be labelled")
        for entry in entries:
            out.append(
                SourceLabelledQuery(
                    domain, entry["query"], set(entry["authority_hosts"])
                )
            )
    return out


def on_list_share(domain: str, hosts: set[str]) -> float:
    """Share of authority hosts inside the category's own site list.

    A high share means the category arm's precision is inflated by
    construction; the report states it so a reader can discount it.
    """
    sites = {site.split("/", 1)[0] for site in DOMAIN_REGISTRY[domain].source_sites}
    if not hosts:
        return 0.0
    return sum(1 for h in hosts if any(host_matches(h, s) for s in sites)) / len(hosts)


@dataclass(frozen=True)
class SourceQueryOutcome:
    domain: str
    query: str
    plain: ArmResult
    suffix: ArmResult
    category: ArmResult
    fallback_fired: bool
    delta_vs_suffix: float
    delta_vs_plain: float
    jaccard_vs_suffix: float
    excluded: bool
    on_list_share: float


async def _cached_pages(query, page_size, *, search_fn, cache, provider):
    """search_category's search function: the same cache keys as _run_arm, so
    the fallback leg reuses the suffix arm's response."""
    del page_size  # every arm is scored at TOP_K
    cached = cache.get(cache_key(query, provider=provider))
    if cached is None:
        pages = await search_fn(query, provider=provider, page_size=TOP_K)
        cached = [
            p.url
            for p in pages
            if getattr(p, "url", None) and not getattr(p, "error", None)
        ]
        cache.set(cache_key(query, provider=provider), cached)
    return [SearchPage(url=url) for url in cached]


async def run_source_query(
    label: SourceLabelledQuery,
    *,
    search_fn,
    cache: DiskCache,
    provider: str = "serpapi",
) -> SourceQueryOutcome:
    plain = await _run_arm(
        label.query, search_fn=search_fn, cache=cache, provider=provider
    )
    suffix = await _run_arm(
        prepare_domain_query(label.query, label.domain),
        search_fn=search_fn,
        cache=cache,
        provider=provider,
    )
    fetch = partial(_cached_pages, search_fn=search_fn, cache=cache, provider=provider)
    try:
        # restricted_fn is passed unconditionally: the eval measures the
        # category arm whatever the operator's AGENTIC_SEARCH_CATEGORY_SITES.
        result = await search_category(
            label.query,
            domain=label.domain,
            page_size=TOP_K,
            search_fn=fetch,
            restricted_fn=fetch,
        )
        urls = [page.url for page in result.pages]
        category = ArmResult(urls=urls, empty=not urls)
        fallback = len(result.executed_queries) > 1
    except Exception as exc:
        category = ArmResult(urls=[], empty=True, error=str(exc))
        fallback = False
    hosts = label.authority_hosts
    excluded = any(arm.error or arm.empty for arm in (plain, suffix, category))
    return SourceQueryOutcome(
        domain=label.domain,
        query=label.query,
        plain=plain,
        suffix=suffix,
        category=category,
        fallback_fired=fallback,
        delta_vs_suffix=authority_precision(category.urls, hosts)
        - authority_precision(suffix.urls, hosts),
        delta_vs_plain=authority_precision(category.urls, hosts)
        - authority_precision(plain.urls, hosts),
        jaccard_vs_suffix=jaccard(suffix.urls, category.urls),
        excluded=excluded,
        on_list_share=on_list_share(label.domain, hosts)
        if label.domain != "web"
        else 0.0,
    )
