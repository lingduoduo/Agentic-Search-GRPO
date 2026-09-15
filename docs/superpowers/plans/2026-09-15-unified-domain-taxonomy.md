# Unified Domain Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `DOMAIN_REGISTRY` plus `CAPABILITY_ROUTES` with one registry in which each domain declares its capabilities, and remove the `sub_domain` parameter aliases.

**Architecture:** A `Capability` dataclass hangs off each `SearchDomain`, so a dotted tag is derived as `f"{domain}.{capability.name}"` instead of being a hand-written dict key. `DomainSearch` builds its route map by walking the registry, `get_sub_domains` emits both entry kinds from one loop, and `search` keeps `tag`/`params` while dropping their aliases.

**Tech Stack:** Python 3.10+, dataclasses, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-15-unified-domain-taxonomy-design.md`

## Global Constraints

- Python >=3.10; no new runtime dependencies.
- `DOMAIN_REGISTRY` keeps its name and its exact 17 keys in their current order; `src/internal/servers/web/app.py` and `/api/search-domains` import it.
- The dotted `domain.capability` tag format is unchanged on the wire.
- Every one of the nine tags that `CAPABILITY_ROUTES` defines today must resolve to the same tool name and the same query parameter afterwards.
- `returns` is exactly `"documents"` or `"records"`.
- Do not import `RetrieverTarget`, and do not modify `src/internal/routing/`.
- A capability whose `tool_name` is not among the supplied tools is dropped, not an error.
- `NOT_AGENT_CALLABLE` continues to withhold exactly `search_domain`, `get_sub_domains`, and `batch_search`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/internal/tools/search.py` | `Capability`, `SearchDomain.capabilities`, the merged `DOMAIN_REGISTRY`, `DomainSearch` route building, `get_sub_domains`, `search`, `batch_search`, `_search_properties` |
| `src/internal/mcp_server/tools/search.py` | Drop the two alias parameters from the `search_domain` and `batch_search` wrappers |
| `tests/unit/test_domain_taxonomy.py` | New: the migration table and the registry invariants |
| `tests/unit/test_domain_search.py` | Existing 11 tests; update only where an alias is exercised |
| `docs/search-engine.md` | Record the single registry and the removed aliases |

---

### Task 1: The merged registry

**Files:**
- Modify: `src/internal/tools/search.py:28-81` (`SearchDomain`, `DOMAIN_REGISTRY`), `:815-825` (`CAPABILITY_ROUTES`)
- Test: `tests/unit/test_domain_taxonomy.py`

**Interfaces:**
- Produces: `Capability(name, query_parameter, returns, tool_name=None)`; `SearchDomain(description, query_hint, capabilities=())`; `DOMAIN_REGISTRY: dict[str, SearchDomain]`; `iter_capabilities() -> Iterator[tuple[str, Capability]]` yielding `(tag, capability)`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_domain_taxonomy.py`:

```python
"""The domain taxonomy is one registry; tags are derived, not written.

The migration table below is the load-bearing test: it pins every tag the
removed CAPABILITY_ROUTES defined, so this change is provably a refactor.
"""

from __future__ import annotations

import pytest

from src.internal.tools.search import (
    AVAILABLE_DOMAINS,
    DOMAIN_REGISTRY,
    Capability,
    iter_capabilities,
)

# Exactly what CAPABILITY_ROUTES mapped before this change.
LEGACY_ROUTES = {
    "general.wikipedia": ("search_wikipedia", "query"),
    "academic.arxiv": ("search_arxiv", "query"),
    "resource.wayback": ("search_wayback", "url"),
    "finance.quote": ("get_stock_quote", "symbol"),
    "finance.crypto": ("get_crypto_price", "symbol"),
    "finance.currency": ("convert_currency", "from_currency"),
    "environment.weather": ("get_weather", "location"),
    "travel.location": ("search_location", "query"),
    "travel.nearby": ("search_nearby_places", "query"),
}


def test_every_legacy_tag_still_resolves_identically():
    by_tag = {tag: cap for tag, cap in iter_capabilities()}
    for tag, (tool_name, query_parameter) in LEGACY_ROUTES.items():
        assert tag in by_tag, f"{tag} disappeared"
        assert by_tag[tag].tool_name == tool_name, tag
        assert by_tag[tag].query_parameter == query_parameter, tag


def test_no_extra_tool_backed_capabilities_were_invented():
    # A refactor adds no routes. `web` is tool-less and excluded.
    tool_backed = {tag for tag, cap in iter_capabilities() if cap.tool_name}
    assert tool_backed == set(LEGACY_ROUTES)


def test_every_domain_declares_a_web_capability():
    for name, domain in DOMAIN_REGISTRY.items():
        names = [c.name for c in domain.capabilities]
        assert "web" in names, name


def test_the_web_capability_is_tool_less():
    for name, domain in DOMAIN_REGISTRY.items():
        web = next(c for c in domain.capabilities if c.name == "web")
        assert web.tool_name is None, name
        assert web.query_parameter == "query"
        assert web.returns == "documents"


def test_registry_keeps_its_seventeen_keys_in_order():
    assert len(DOMAIN_REGISTRY) == 17
    assert list(DOMAIN_REGISTRY) == AVAILABLE_DOMAINS
    assert AVAILABLE_DOMAINS[0] == "general"


def test_returns_is_one_of_two_values():
    for tag, cap in iter_capabilities():
        assert cap.returns in ("documents", "records"), tag


def test_document_and_record_capabilities_are_both_present():
    # A constant field would carry no information; this pins the split.
    kinds = {cap.returns for _tag, cap in iter_capabilities()}
    assert kinds == {"documents", "records"}


def test_capability_names_carry_no_dot():
    # The dot is the tag separator; a dotted name would make tags ambiguous.
    for _tag, cap in iter_capabilities():
        assert "." not in cap.name, cap.name


@pytest.mark.parametrize("tag", sorted(LEGACY_ROUTES))
def test_tags_are_derived_from_their_domain(tag):
    domain, _, capability = tag.partition(".")
    assert domain in DOMAIN_REGISTRY
    assert capability in {c.name for c in DOMAIN_REGISTRY[domain].capabilities}


def test_capability_is_immutable():
    cap = Capability(name="x", query_parameter="query", returns="documents")
    with pytest.raises(Exception):
        cap.name = "y"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_domain_taxonomy.py -q`
Expected: FAIL — `Capability` and `iter_capabilities` do not exist.

- [ ] **Step 3: Add the dataclasses**

In `src/internal/tools/search.py`, replace the `SearchDomain` declaration with:

```python
@dataclass(frozen=True)
class Capability:
    """One way to answer a query inside a domain.

    ``tool_name`` of None means the built-in web cascade rather than a seeded
    public-data tool. ``returns`` is "documents" or "records": it tells a
    caller whether to expect titled text or structured fields, which is the
    one axis that actually varies across these capabilities.
    """

    name: str
    query_parameter: str
    returns: str
    tool_name: str | None = None


@dataclass(frozen=True)
class SearchDomain:
    description: str
    query_hint: str
    capabilities: tuple[Capability, ...] = ()
```

Confirm `dataclass` is already imported at the top of the module; `SearchDomain`
is already a frozen dataclass, so the import exists.

- [ ] **Step 4: Declare the shared web capability and merge the routes**

Above `DOMAIN_REGISTRY`, add:

```python
# Every domain can be searched on the public web; the hint is what differs.
# Declaring it makes the entry `get_sub_domains` used to fabricate ordinary.
WEB_CAPABILITY = Capability(
    name="web", query_parameter="query", returns="documents"
)
```

Then give each of the 17 entries its capabilities. Domains with no routed tool
take `capabilities=(WEB_CAPABILITY,)`. The nine that had routes become:

```python
    "general": SearchDomain(
        "Broad or mixed-topic search; default",
        "",
        (
            WEB_CAPABILITY,
            Capability("wikipedia", "query", "documents", "search_wikipedia"),
        ),
    ),
    "resource": SearchDomain(
        "Datasets, reference materials, directories, and reusable tools",
        "resources",
        (
            WEB_CAPABILITY,
            Capability("wayback", "url", "documents", "search_wayback"),
        ),
    ),
    "finance": SearchDomain(
        "Markets, investments, banking, and financial analysis",
        "finance",
        (
            WEB_CAPABILITY,
            Capability("quote", "symbol", "records", "get_stock_quote"),
            Capability("crypto", "symbol", "records", "get_crypto_price"),
            Capability("currency", "from_currency", "records", "convert_currency"),
        ),
    ),
    "academic": SearchDomain(
        "Scholarly literature, research methods, and publications",
        "academic research",
        (
            WEB_CAPABILITY,
            Capability("arxiv", "query", "documents", "search_arxiv"),
        ),
    ),
    "environment": SearchDomain(
        "Climate, ecosystems, conservation, and pollution",
        "environment",
        (
            WEB_CAPABILITY,
            Capability("weather", "location", "records", "get_weather"),
        ),
    ),
    "travel": SearchDomain(
        "Destinations, transport, lodging, and trip planning",
        "travel",
        (
            WEB_CAPABILITY,
            Capability("location", "query", "records", "search_location"),
            Capability("nearby", "query", "records", "search_nearby_places"),
        ),
    ),
```

The remaining eleven — `social_media`, `legal`, `health`, `business`,
`security`, `ip`, `code`, `energy`, `agriculture`, `film`, `gaming` — keep their
existing description and hint and gain `(WEB_CAPABILITY,)` as the third
positional argument. Preserve the current key order exactly.

- [ ] **Step 5: Add the tag iterator and delete CAPABILITY_ROUTES**

After `AVAILABLE_DOMAINS`, add:

```python
def iter_capabilities() -> Iterator[tuple[str, Capability]]:
    """Yield every (tag, capability) pair. The tag is derived, never stored."""
    for domain, entry in DOMAIN_REGISTRY.items():
        for capability in entry.capabilities:
            yield f"{domain}.{capability.name}", capability
```

Add `Iterator` to the `typing` import if absent. Delete the entire
`CAPABILITY_ROUTES` dict at its former location.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_domain_taxonomy.py -q`
Expected: 10 passed.

- [ ] **Step 7: Confirm the rest of the module still imports**

Run: `python -c "import src.internal.tools.search"`
Expected: an ImportError naming `CAPABILITY_ROUTES` from `DomainSearch.__init__`,
which Task 2 fixes. If any *other* name fails, fix that before continuing.

- [ ] **Step 8: Commit**

```bash
git add src/internal/tools/search.py tests/unit/test_domain_taxonomy.py
git commit -m "refactor(tools): merge capability routes into the domain registry"
```

---

### Task 2: Route building and discovery from one structure

**Files:**
- Modify: `src/internal/tools/search.py` — `DomainSearch.__init__`, `DomainSearch.get_sub_domains`
- Test: `tests/unit/test_domain_taxonomy.py`

**Interfaces:**
- Consumes: `iter_capabilities`, `Capability`, `DOMAIN_REGISTRY` from Task 1
- Produces: `DomainSearch.routes: dict[str, tuple[Tool, str]]` keyed by dotted tag, unchanged in shape; `get_sub_domains` entries gain a `returns` key

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_domain_taxonomy.py`:

```python
from src.internal.tools.public_data import public_data_tools
from src.internal.tools.search import DomainSearch


def test_routes_match_the_legacy_table():
    service = DomainSearch(tools=public_data_tools())
    resolved = {
        tag: (tool.name, parameter)
        for tag, (tool, parameter) in service.routes.items()
    }
    assert resolved == LEGACY_ROUTES


def test_a_capability_naming_an_unseeded_tool_is_dropped():
    # DomainSearch must stay usable on a partial tool list; MCP relies on it.
    service = DomainSearch(tools=[])
    assert service.routes == {}


def test_discovery_lists_web_first_then_capabilities():
    service = DomainSearch(tools=public_data_tools())
    entries = service.get_sub_domains(["finance"])["domains"][0]["sub_domains"]
    assert entries[0]["sub_domain"] == "finance.web"
    assert {e["sub_domain"] for e in entries} == {
        "finance.web",
        "finance.quote",
        "finance.crypto",
        "finance.currency",
    }


def test_discovery_reports_what_each_capability_returns():
    service = DomainSearch(tools=public_data_tools())
    entries = service.get_sub_domains(["finance"])["domains"][0]["sub_domains"]
    by_tag = {e["sub_domain"]: e for e in entries}
    assert by_tag["finance.web"]["returns"] == "documents"
    assert by_tag["finance.quote"]["returns"] == "records"


def test_discovery_omits_capabilities_whose_tool_is_absent():
    service = DomainSearch(tools=[])
    entries = service.get_sub_domains(["finance"])["domains"][0]["sub_domains"]
    assert [e["sub_domain"] for e in entries] == ["finance.web"]


def test_discovery_still_rejects_bad_domain_counts():
    service = DomainSearch(tools=[])
    with pytest.raises(ValueError, match="one to five"):
        service.get_sub_domains([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_domain_taxonomy.py -q -k "routes or discovery"`
Expected: FAIL — `DomainSearch.__init__` still references `CAPABILITY_ROUTES`.

- [ ] **Step 3: Build routes from the registry**

In `DomainSearch.__init__`, replace the `self.routes` assignment with:

```python
        self.routes = {
            tag: (by_name[capability.tool_name], capability.query_parameter)
            for tag, capability in iter_capabilities()
            if capability.tool_name and capability.tool_name in by_name
        }
```

The `capability.tool_name and` clause is what keeps the tool-less `web`
capability out of the route map, where it has never belonged.

- [ ] **Step 4: Emit both entry kinds from one loop**

Replace the body of `get_sub_domains` between the `domain = normalize_search_domain(value)`
line and `directories.append(...)` with a single loop:

```python
            entries = []
            for capability in DOMAIN_REGISTRY[domain].capabilities:
                tag = f"{domain}.{capability.name}"
                if capability.tool_name is None:
                    entries.append(
                        {
                            "sub_domain": tag,
                            "tool_name": "web_search",
                            "description": (
                                f"Search the public web with the {domain} topic hint."
                            ),
                            "query_parameter": capability.query_parameter,
                            "query_format": "Natural-language search query",
                            "returns": capability.returns,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "max_results": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 10,
                                    },
                                },
                                "required": ["query"],
                            },
                            "params": {},
                        }
                    )
                    continue
                if tag not in self.routes:
                    continue
                tool, query_parameter = self.routes[tag]
                schema = copy.deepcopy(tool.schema.parameters)
                properties = schema.get("properties", {})
                entries.append(
                    {
                        "sub_domain": tag,
                        "tool_name": tool.name,
                        "description": tool.schema.description,
                        "query_parameter": query_parameter,
                        "query_format": properties.get(query_parameter, {}).get(
                            "description", query_parameter
                        ),
                        "returns": capability.returns,
                        "parameters": schema,
                        "params": {
                            name: {
                                **spec,
                                "required": name in schema.get("required", []),
                            }
                            for name, spec in properties.items()
                            if name != query_parameter
                        },
                    }
                )
```

The `startswith` filter is gone: capabilities arrive already scoped to their
domain, so no string surgery recovers what the structure states.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_domain_taxonomy.py -q`
Expected: 16 passed.

- [ ] **Step 6: Run the existing domain suites**

Run: `pytest tests/unit/test_domain_search.py tests/unit/test_domain_search_tools.py tests/unit/test_domain_search_mcp.py tests/unit/test_knowledge_base.py -q`
Expected: all pass. These tests predate the change and are the regression net;
if one fails, the refactor changed behaviour and must be corrected, not the test.

- [ ] **Step 7: Commit**

```bash
git add src/internal/tools/search.py tests/unit/test_domain_taxonomy.py
git commit -m "refactor(tools): build routes and discovery from the one registry"
```

---

### Task 3: Drop the parameter aliases

**Files:**
- Modify: `src/internal/tools/search.py` — `DomainSearch.search`, `DomainSearch.batch_search`, `_search_properties`
- Modify: `src/internal/mcp_server/tools/search.py:201-260`
- Test: `tests/unit/test_domain_taxonomy.py`

**Interfaces:**
- Produces: `DomainSearch.search(query, *, domain=None, tag=None, params=None, max_results=5)`; `batch_search(queries, **shared_options)` unchanged in signature

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_domain_taxonomy.py`:

```python
import inspect


def test_search_no_longer_accepts_the_aliases():
    parameters = inspect.signature(DomainSearch.search).parameters
    assert "sub_domain" not in parameters
    assert "sub_domain_params" not in parameters
    # The surviving names stay, so callers holding a tag are unaffected.
    for name in ("query", "domain", "tag", "params", "max_results"):
        assert name in parameters


@pytest.mark.asyncio
async def test_passing_a_removed_alias_is_rejected_not_ignored():
    # Silently dropping it would send an unintended query to a provider.
    service = DomainSearch(tools=[])
    with pytest.raises(TypeError):
        await service.search("q", sub_domain="finance.quote")


def test_the_schema_no_longer_advertises_the_aliases():
    from src.internal.tools.search import _search_properties

    properties = _search_properties()
    assert "sub_domain" not in properties
    assert "sub_domain_params" not in properties
    assert set(properties) == {"query", "domain", "tag", "params", "max_results"}


def test_mcp_wrappers_drop_the_aliases():
    from src.internal.mcp_server.tools import search as mcp_search

    for fn in (mcp_search.search_domain, mcp_search.batch_search):
        parameters = inspect.signature(fn).parameters
        assert "sub_domain" not in parameters, fn.__name__
        assert "sub_domain_params" not in parameters, fn.__name__
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_domain_taxonomy.py -q -k "alias or schema or mcp"`
Expected: FAIL — the aliases are still present.

- [ ] **Step 3: Simplify `search`**

In `DomainSearch.search`, delete the `sub_domain` and `sub_domain_params`
parameters, and replace the two reconciliation blocks. The tag block becomes:

```python
        if tag is not None and (not isinstance(tag, str) or "." not in tag):
            raise ValueError("tag must be a capability returned by get_sub_domains")
```

and the options block becomes:

```python
        options = parse_search_params(params)
```

Both `if ... must match` raises are removed with the aliases that required them.

- [ ] **Step 4: Simplify `batch_search`**

Replace the per-item default handling with the three-key version:

```python
                defaults = dict(shared_options)
                if item.get("tag"):
                    # An explicit tag overrides both shared selectors.
                    defaults.pop("tag", None)
                    defaults.pop("domain", None)
                elif "domain" in item:
                    defaults.pop("tag", None)
                if "params" in item:
                    defaults.pop("params", None)
```

- [ ] **Step 5: Simplify the advertised schema**

In `_search_properties`, delete the `sub_domain` and `sub_domain_params`
entries, and reword the `tag` description so it no longer mentions an alias:

```python
        "tag": {
            "type": "string",
            "description": (
                "An implemented capability returned by get_sub_domains, "
                "e.g. finance.quote or academic.arxiv."
            ),
        },
```

- [ ] **Step 6: Simplify the MCP wrappers**

In `src/internal/mcp_server/tools/search.py`, remove the `sub_domain` and
`sub_domain_params` parameters from both `search_domain` and `batch_search`,
and remove the corresponding `sub_domain=`/`sub_domain_params=` arguments they
forward. Leave every other parameter and the docstrings' tag guidance intact.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/unit/test_domain_taxonomy.py -q`
Expected: 20 passed.

- [ ] **Step 8: Run the full suite**

Run: `pytest -q`
Expected: no new failures against the pre-change baseline. Note that
`tests/unit/test_mcp_document_tools.py::test_parser_watchdog_terminates_a_process_over_the_rss_limit`
flakes under full-suite load and is unrelated; re-run it alone to confirm.

- [ ] **Step 9: Commit**

```bash
git add src/internal/tools/search.py src/internal/mcp_server/tools/search.py tests/unit/test_domain_taxonomy.py
git commit -m "refactor(tools): drop the sub_domain parameter aliases"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/search-engine.md`
- Modify: `src/internal/tools/knowledge_base.py:77-85` (the NOT_AGENT_CALLABLE comment)

- [ ] **Step 1: Update the withholding comment**

The comment says "every tag in CAPABILITY_ROUTES routes to a public-data tool
seeded above". That name no longer exists. Reword to name the registry's
capabilities instead, keeping the reasoning identical:

```python
# ``search_domain`` and ``batch_search`` are a facade: every tool-backed
# capability in DOMAIN_REGISTRY routes to a public-data tool seeded above, so
# offering both gives the model two paths to the same nine tools and a third
# way to run a web search it already has in ``web_search``.
```

- [ ] **Step 2: Document the registry**

In `docs/search-engine.md`, under the domain sections, record that domains and
their capabilities are one structure; that a tag is `domain.capability` derived
from it rather than a stored key; that each capability declares whether it
returns documents or records; and that `sub_domain`/`sub_domain_params` were
removed in favour of `tag`/`params`. State that the tag format is unchanged, so
a client that discovers a tag and passes it back is unaffected.

- [ ] **Step 3: Verify the docs name only real symbols**

Run: `grep -rn "CAPABILITY_ROUTES" docs/ src/`
Expected: no output. Every mention must be gone, since the name no longer exists.

- [ ] **Step 4: Commit**

```bash
git add docs/search-engine.md src/internal/tools/knowledge_base.py
git commit -m "docs: describe the unified domain taxonomy"
```

---

## Self-Review

**Spec coverage:** the merged registry and `Capability` (Task 1); tags derived
rather than stored (Task 1, asserted by a parametrized test); `web` declared on
every domain (Task 1); route building from one structure and the dropped
`startswith` surgery (Task 2); `returns` surfaced in discovery (Task 2); the
unseeded-tool drop preserved (Tasks 1-2); alias removal across `search`,
`batch_search`, the advertised schema, and both MCP wrappers (Task 3); the
migration table proving losslessness (Task 1); documentation and the stale
comment (Task 4). The spec's "not merged" decision needs no task — `routing/` is
untouched, and the Global Constraints forbid importing `RetrieverTarget`.

**Placeholder scan:** no TBD or "handle errors appropriately"; every code step
carries real code, and Task 1 Step 4 enumerates the eleven web-only domains by
name rather than saying "and the rest".

**Type consistency:** `Capability(name, query_parameter, returns, tool_name)` is
constructed with that field order in Task 1 and read with those names in Tasks 2
and 3. `iter_capabilities()` yields `(tag, capability)` in Task 1 and is
unpacked that way in Tasks 1 and 2. `DomainSearch.routes` keeps its existing
`dict[str, tuple[Tool, str]]` shape, which Task 2's first test pins against the
legacy table.
