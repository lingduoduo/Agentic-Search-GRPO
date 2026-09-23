"""Every seeded tool declares its constraints as schema keywords, not prose.

A validator can only enforce what the schema declares. These schemas used to
say "(1-10)" or "One of relevance, ..." in a description while the code clamped
or substituted, so the model never learned it was wrong. The guard holds every
seeded tool to one rule set; the table pins each constraint to the value the
tool's own code applies. Validation goes through jsonschema directly so the
tests do not depend on how the registry's validator reports errors.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from src.internal.memory.tools import build_memory_registry
from src.internal.tools.knowledge_base import tool_knowledge_base
from src.internal.tools.routing_tools import build_rag_routing_tool

# Object schemas whose keys legitimately vary, with the reason they stay open.
OPEN_MAPS = {
    # Keys depend on the chosen `tag`; the executor checks them per capability.
    ("search_domain", "$.params"),
    ("batch_search", "$.params"),
    ("batch_search", "$.queries[].params"),
}


class _NoLLM:
    def complete(self, *args, **kwargs):
        return ""


def _seeded_tools():
    tools = tool_knowledge_base()
    tools.append(build_rag_routing_tool(llm=_NoLLM(), search_url="http://x", top_k=5))
    registry, _, _ = build_memory_registry(store=None, user_id="u")
    tools.extend(registry.list_tools())
    return {tool.schema.name: tool.schema.parameters for tool in tools}


SCHEMAS = _seeded_tools()


def _violations(tool: str, path: str, schema: dict, required: bool) -> list[str]:
    found: list[str] = []
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        if (tool, path) not in OPEN_MAPS and schema.get(
            "additionalProperties"
        ) is not False:
            found.append(f"{path}: object is not closed")
        needed = set(schema.get("required", []))
        for key, sub in schema.get("properties", {}).items():
            found += _violations(tool, f"{path}.{key}", sub, key in needed)
    if kind == "integer" and not ("minimum" in schema and "maximum" in schema):
        found.append(f"{path}: integer needs minimum and maximum")
    if kind == "number" and not ("minimum" in schema or "exclusiveMinimum" in schema):
        found.append(f"{path}: number needs a lower bound")
    if kind == "array":
        if "items" not in schema or "maxItems" not in schema:
            found.append(f"{path}: array needs items and maxItems")
        if "items" in schema:
            found += _violations(tool, f"{path}[]", schema["items"], False)
    if kind == "string" and required and schema.get("minLength", 0) < 1:
        found.append(f"{path}: required string needs minLength >= 1")
    return found


def test_every_seeded_tool_is_covered():
    # A tool added to the seed set is checked automatically; this pins that the
    # set is the one the guard claims to cover.
    assert {"web_search", "search_wikipedia", "extract_page", "add_memory"} <= set(
        SCHEMAS
    )


@pytest.mark.parametrize("tool", sorted(SCHEMAS))
def test_schema_is_valid(tool):
    Draft202012Validator.check_schema(SCHEMAS[tool])


@pytest.mark.parametrize("tool", sorted(SCHEMAS))
def test_schema_declares_its_constraints(tool):
    assert _violations(tool, "$", SCHEMAS[tool], False) == []


def _defaults(schema: dict):
    for key, sub in schema.get("properties", {}).items():
        if "default" in sub:
            yield key, sub
        yield from _defaults(sub)


@pytest.mark.parametrize("tool", sorted(SCHEMAS))
def test_defaults_satisfy_their_schema(tool):
    for key, sub in _defaults(SCHEMAS[tool]):
        assert Draft202012Validator(sub).is_valid(sub["default"]), (tool, key)


def _ok(tool: str, args: dict) -> bool:
    return Draft202012Validator(SCHEMAS[tool]).is_valid(args)


TABLE = [
    ("search_wikipedia", {"query": "x", "limit": 10}, True),
    ("search_wikipedia", {"query": "x", "limit": 11}, False),
    ("search_wikipedia", {"query": "x", "language": "zh-yue"}, True),
    ("search_wikipedia", {"query": "x", "language": "en; drop"}, False),
    ("search_wikipedia", {"query": ""}, False),
    ("search_wikipedia", {"query": "x", "limti": 3}, False),
    ("search_arxiv", {"query": "x", "max_results": 25}, True),
    ("search_arxiv", {"query": "x", "max_results": 26}, False),
    ("search_arxiv", {"query": "x", "sort_by": "submittedDate"}, True),
    ("search_arxiv", {"query": "x", "sort_by": "newest"}, False),
    ("search_wayback", {"url": "example.com", "limit": 50, "year": 1996}, True),
    ("search_wayback", {"url": "example.com", "limit": 51}, False),
    ("search_wayback", {"url": "example.com", "year": 1995}, False),
    ("get_weather", {"location": "Berlin", "latitude": -90, "longitude": 180}, True),
    ("get_weather", {"location": "Berlin", "latitude": 91}, False),
    ("get_weather", {"location": "Berlin", "longitude": -181}, False),
    (
        "search_nearby_places",
        {"query": "cafe", "latitude": 1, "longitude": 1, "radius_meters": 10000},
        True,
    ),
    (
        "search_nearby_places",
        {"query": "cafe", "latitude": 1, "longitude": 1, "radius_meters": 10001},
        False,
    ),
    (
        "search_nearby_places",
        {"query": "cafe", "latitude": 1, "longitude": 1, "limit": 51},
        False,
    ),
    ("search_location", {"query": "x", "limit": 20, "country_code": "fr"}, True),
    ("search_location", {"query": "x", "limit": 21}, False),
    ("search_location", {"query": "x", "country_code": "fra"}, False),
    (
        "convert_currency",
        {"amount": 1, "from_currency": "usd", "to_currency": "EUR"},
        True,
    ),
    (
        "convert_currency",
        {"amount": 0, "from_currency": "USD", "to_currency": "EUR"},
        False,
    ),
    (
        "convert_currency",
        {"amount": 1, "from_currency": "US", "to_currency": "EUR"},
        False,
    ),
    ("get_crypto_price", {"symbol": "btc", "vs_currency": "usd"}, True),
    ("get_crypto_price", {"symbol": "btc", "vs_currency": "u$d"}, False),
    ("get_stock_quote", {"symbol": ""}, False),
    ("extract_page", {"url": "HTTP://example.test"}, True),
    ("extract_page", {"url": "ftp://example.test"}, False),
    ("web_search", {"queries": ["a", "b", "c", "d", "e"]}, True),
    ("web_search", {"queries": ["a", "b", "c", "d", "e", "f"]}, False),
    ("web_search", {"queries": []}, False),
    ("add_memory", {"content": ""}, False),
    ("delete_memory", {"memory_id": "m1"}, True),
    ("delete_memory", {"memory_id": "m1", "force": True}, False),
]


@pytest.mark.parametrize("tool,args,ok", TABLE)
def test_constraint(tool, args, ok):
    assert _ok(tool, args) is ok
