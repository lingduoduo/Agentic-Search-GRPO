"""Shared topic taxonomy for search tools, independent of search providers.

Domains provide query hints, not result filters or authorization constraints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchDomain:
    description: str
    query_hint: str


DOMAIN_REGISTRY: dict[str, SearchDomain] = {
    "general": SearchDomain("Broad or mixed-topic search; default", ""),
    "resource": SearchDomain(
        "Datasets, reference materials, directories, and reusable tools", "resources"
    ),
    "social_media": SearchDomain(
        "Public social posts, communities, and discussions", "social media"
    ),
    "finance": SearchDomain(
        "Markets, investments, banking, and financial analysis", "finance"
    ),
    "academic": SearchDomain(
        "Scholarly literature, research methods, and publications", "academic research"
    ),
    "legal": SearchDomain("Law, regulation, case law, and legal procedure", "law"),
    "health": SearchDomain(
        "Medicine, public health, and clinical information", "health"
    ),
    "business": SearchDomain(
        "Companies, operations, strategy, and commerce", "business"
    ),
    "security": SearchDomain(
        "Cybersecurity, vulnerabilities, and defensive practices", "cybersecurity"
    ),
    "ip": SearchDomain(
        "Intellectual property: patents, trademarks, copyright, and licensing",
        "intellectual property",
    ),
    "code": SearchDomain(
        "Source code, programming, APIs, and developer documentation", "programming"
    ),
    "energy": SearchDomain("Generation, fuels, storage, and energy systems", "energy"),
    "environment": SearchDomain(
        "Climate, ecosystems, conservation, and pollution", "environment"
    ),
    "agriculture": SearchDomain(
        "Farming, crops, livestock, and agricultural systems", "agriculture"
    ),
    "travel": SearchDomain(
        "Destinations, transport, lodging, and trip planning", "travel"
    ),
    "film": SearchDomain("Cinema, films, filmmaking, and the film industry", "film"),
    "gaming": SearchDomain(
        "Video games, game development, and gaming communities", "video games"
    ),
}
AVAILABLE_DOMAINS: list[str] = list(DOMAIN_REGISTRY)


def normalize_search_domain(value: str = "general") -> str:
    """Normalize an explicit topic selector or reject it before dispatch."""
    if isinstance(value, str):
        canonical = value.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical in DOMAIN_REGISTRY:
            return canonical
    raise ValueError("domain must be one of: " + ", ".join(DOMAIN_REGISTRY))


def prepare_domain_query(query: str, domain: str = "general") -> str:
    """Append a topic hint once at a tool entry point; general is unchanged."""
    canonical = normalize_search_domain(domain)
    hint = DOMAIN_REGISTRY[canonical].query_hint
    return f"{query} {hint}" if hint and query.strip() else query


def search_domain_parameter() -> dict[str, object]:
    """Build fresh schema data so callers cannot mutate the registry."""
    descriptions = "; ".join(
        f"{name}: {entry.description}" for name, entry in DOMAIN_REGISTRY.items()
    )
    return {
        "type": "string",
        "enum": list(DOMAIN_REGISTRY),
        "default": "general",
        "description": (
            "Optional topic query hint, not a guaranteed result filter. "
            "Use general for broad or mixed topics. " + descriptions
        ),
    }
