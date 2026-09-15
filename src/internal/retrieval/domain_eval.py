"""Measure whether a domain hint moves web results toward topic sources.

This module stays torch-free on purpose: importing ``src.model.post_training``
pulls torch in through package ``__init__`` side effects, which would drop this
module out of the torch-free CI job. The paired statistics therefore live in
the CLI, not here.

What it measures is source alignment, not answer quality. A result from
arxiv.org is not automatically a better answer than a good blog post.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

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
