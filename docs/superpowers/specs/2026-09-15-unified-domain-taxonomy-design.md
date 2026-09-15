# Unified domain taxonomy

## Goal

Replace the two structures that jointly describe search domains with one, and
collapse the parameter aliases that accreted on the search API. A domain and its
capabilities become one declaration; a capability's tag becomes derived rather
than hand-maintained.

## The problem

Two structures in `src/internal/tools/search.py` describe overlapping things.

`DOMAIN_REGISTRY` maps 17 topic identifiers to a description and a query hint.
`CAPABILITY_ROUTES` maps 9 dotted strings to a tool name and a query parameter.
The dotted string's prefix is a `DOMAIN_REGISTRY` key, but nothing enforces
that: `"fnance.quote"` is an accepted dict key that belongs to no domain and
would simply never match. The relationship lives in string convention.

That convention is then read back out by string surgery. `get_sub_domains`
filters with `tag.startswith(domain + ".")`; `search` recovers the domain with
`tag.split(".", 1)[0]` and re-validates the prefix it just split.
`get_sub_domains` also hand-builds a synthetic `{domain}.web` entry with a
literal inline schema before looping over the routed tools, so one function
constructs its two result kinds in two unrelated ways.

Separately, `DomainSearch.search` takes five parameters for three concepts:
`tag` with its alias `sub_domain`, and `params` with its alias
`sub_domain_params`, each pair guarded by an equality check that raises when
the two spellings disagree. `batch_search` then re-implements that aliasing a
third time, popping six key names to decide which defaults an item overrides.

## Architecture

One registry. Each domain declares its capabilities; each capability declares
what it calls and what it returns.

```python
@dataclass(frozen=True)
class Capability:
    name: str                    # "arxiv", "quote", "web"
    query_parameter: str         # "query", "symbol", "location"
    returns: str                 # "documents" or "records"
    tool_name: str | None = None # None means the built-in web cascade


@dataclass(frozen=True)
class SearchDomain:
    description: str
    query_hint: str
    capabilities: tuple[Capability, ...] = ()
```

`DOMAIN_REGISTRY` keeps its name and its 17 keys, because the web layer and the
`/api/search-domains` endpoint already import it. `CAPABILITY_ROUTES` is
removed. A tag is no longer a key anyone writes; it is `f"{domain}.{capability
.name}"`, produced from structure.

Every domain gains a `web` capability explicitly, so the synthetic entry
`get_sub_domains` used to fabricate is now declared like any other and both
result kinds are built by one loop.

### Why not RetrieverTarget

An earlier draft gave each capability a `modality: RetrieverTarget` so the
search taxonomy and the routing layer would share a vocabulary. Inspection
killed it: all nine routed tools are remote HTTP APIs, so the field would be
`API` for every capability, a constant carrying no information.
`RetrieverTarget`'s values name retrieval mechanisms — sparse, dense, hybrid,
sql, graph — which describe how a local index is queried, not what a remote
capability hands back.

`returns` is kept instead because it genuinely varies. Wikipedia, arXiv,
Wayback, and web search return documents; stock quotes, crypto prices, currency
conversion, weather, geocoding, and nearby places return records. A caller
choosing between capabilities can use that distinction, and capability
selection is a demonstrated weakness here: three tools are withheld from the
agent menu precisely because a model could not choose among near-duplicates.

### What is deliberately not merged

`src/internal/routing/registry.py` keeps its own `DEFAULT_ROUTES`. Its entries
name modalities — docs, structured, graph, live — bound to retriever targets and
sources. A topic and a modality are orthogonal: a finance question may be an
article, a table, or a live quote. Fusing them yields a namespace whose
combinations are mostly meaningless, and the two structures also belong to
retrieval stacks this repository has deliberately kept separate. They are left
apart.

## API surface

`search` keeps `query`, `domain`, `tag`, `params`, and `max_results`. The
aliases `sub_domain` and `sub_domain_params` are removed, along with the two
equality guards that existed only to reconcile them. One concept has one name.

`batch_search`'s per-item override logic collapses accordingly: with three keys
instead of six, an item that supplies `tag` clears the shared `tag` and
`domain`, and an item that supplies `params` clears the shared `params`.

MCP's `search_domain` and `batch_search` wrappers in
`src/internal/mcp_server/tools/search.py` drop the same two parameters. The
dotted tag format is unchanged, so a client that calls `get_sub_domains` and
passes back a tag continues to work. Only a caller using the alias spellings
breaks; no caller in this repository does.

`get_sub_domains` keeps its response shape — `sub_domain`, `tool_name`,
`description`, `query_parameter`, `query_format`, `parameters`, `params` — and
gains `returns` on each entry.

## Error handling

An unknown domain still raises through `normalize_search_domain`. An unknown
tag still raises naming `get_sub_domains`. A tag whose domain prefix disagrees
with an explicit `domain` argument still raises.

A capability whose `tool_name` is not among the seeded tools is dropped at
construction, exactly as the current `if name in by_name` filter does. This
keeps `DomainSearch` usable when a caller supplies a partial tool list, which
`build_domain_search_tools` and the MCP service both rely on.

## Testing

Test-driven, and the existing 24 tests across `test_domain_search.py`,
`test_domain_search_tools.py`, `test_domain_search_mcp.py`, and
`test_knowledge_base.py` are the regression net; they must pass unchanged
except where they exercise a removed alias.

The load-bearing new test is a migration table: every tag that
`CAPABILITY_ROUTES` defined today, written out literally, still resolves to the
same tool name and the same query parameter. That is what makes this provably a
refactor rather than a rewrite.

Also tested: every domain exposes a `web` capability; a tag round-trips from
`get_sub_domains` into `search`; a capability naming an unseeded tool is
dropped; a malformed tag with no dot is rejected; passing a removed alias is
rejected rather than silently ignored; and `NOT_AGENT_CALLABLE` still withholds
exactly `search_domain`, `get_sub_domains`, and `batch_search`.

## Limits

This is a refactor. No capability is added, no provider changes, and no search
result changes. The relevance question the taxonomy raises was measured
separately and found no effect; nothing here revisits it.

Out of scope: `routing/registry.py`, adding graph or SQL capabilities to
domains, changing the dotted tag format, and the `NOT_AGENT_CALLABLE` decision
itself.
