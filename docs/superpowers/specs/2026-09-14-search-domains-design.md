# Search domains and categories

Status: Implemented on `feat/search-domains`; awaiting PR review.

## Goal

Introduce the user's exact 17 search-domain identifiers as a shared, documented
taxonomy and make them usable as optional topic hints in function-calling search
tools and MCP public-web search.

This is a new capability: repository inspection found no `AVAILABLE_DOMAINS`.
The user requested both a spec and plan; both documents are prepared together.
The user subsequently authorized code implementation and PR creation against
this design.

## Existing behavior and integration points

- `src/internal/tools/search.py`: `search_tool` dispatches to retrieval, Google,
  SerpAPI, or Serper. Its web cache keys include the executed query.
- The same module's `MultiQueryWebSearchTool` accepts multiple queries and an
  injected search callable. `build_search_tool` wraps `search_for_tool_string`.
- `make_web_cascade_search` accepts the existing search-callable signature and
  sends the same query through SerpAPI and browser fallback.
- `src/internal/mcp_server/tools/search.py`: `search_web` calls provider helpers
  directly. `search_indexed_documents` uses authenticated retrieval separately.
- `src/agents/components/search_tool.py`: the search-agent component chooses
  vector or web retrieval and records state. It has no domain parameter.
- `tests/unit/test_tool_categories.py` covers citation and stopping flags. Those
  flags describe tool behavior and are independent of topic domains.
- `docs/search-engine.md` documents the HTTP search experience. Existing source
  types and document sets are distinct from topics and access permissions.

## Alternatives and decision

1. **Shared taxonomy plus optional query hints — proposed.** Works with existing
   providers and injected callables without new services or index migrations.
   Appending topic words can affect recall; callers opt in and can use `general`.
2. **Taxonomy and descriptions only.** Smallest change but selecting a domain
   would not affect retrieval. Useful as a preliminary library, insufficient as
   the complete feature described here.
3. **Strict filtering or dedicated providers per domain.** Stronger category
   guarantees require document classification, backend support, and maintained
   provider mappings. Existing search results do not establish such guarantees.
   This is a separate future project.

The proposed first version applies a deterministic topic suffix once at the
tool entry point. It does not automatically classify queries or change providers.
Descriptions also help the calling model choose domains and compose queries.
No search-quality improvement is claimed without relevance evaluation.

## Global constraints

- Python >=3.10; no new runtime dependencies.
- Preserve the exact 17 identifiers and their supplied order.
- Omitted domain and explicit `general` preserve existing search behavior.
- Topic domains never replace, relax, or infer authorization filters.
- Domain selection never changes provider selection or fallback order.
- Apply topic hints exactly once, before provider dispatch and cache lookup.
- Reject invalid explicit domains before any network call.
- No HTTP API, UI, index schema, or search-agent state changes in this increment.

## Taxonomy

`AVAILABLE_DOMAINS` is a list derived from an insertion-ordered registry. Each
entry has a user-facing description and a fixed query hint. Validation consults
the registry rather than the mutable exported list. Return a fresh list in schemas.

| ID | Meaning and boundaries | Query hint |
| --- | --- | --- |
| general | Broad or mixed-topic search; default | empty string |
| resource | Datasets, reference materials, directories, and reusable tools | resources |
| social_media | Public social posts, communities, and discussions | social media |
| finance | Markets, investments, banking, and financial analysis | finance |
| academic | Scholarly literature, research methods, and publications | academic research |
| legal | Law, regulation, case law, and legal procedure | law |
| health | Medicine, public health, and clinical information | health |
| business | Companies, operations, strategy, and commerce | business |
| security | Cybersecurity, vulnerabilities, and defensive practices | cybersecurity |
| ip | Intellectual property: patents, trademarks, copyright, and licensing | intellectual property |
| code | Source code, programming, APIs, and developer documentation | programming |
| energy | Generation, fuels, storage, and energy systems | energy |
| environment | Climate, ecosystems, conservation, and pollution | environment |
| agriculture | Farming, crops, livestock, and agricultural systems | agriculture |
| travel | Destinations, transport, lodging, and trip planning | travel |
| film | Cinema, films, filmmaking, and the film industry | film |
| gaming | Video games, game development, and gaming communities | video games |

`resource` is a content-purpose category and `social_media` a source category;
most others are subject categories. Retain this mixed taxonomy for compatibility
with the requested identifiers and explain overlaps instead of implying mutually
exclusive labels. `ip` does not mean Internet Protocol; use `security` or `code`
for networking questions as appropriate. Finance/business, legal/ip, and
health/academic overlap. One selected domain applies to an entire tool call;
callers can issue separate calls for different topics. `general` handles ambiguity.

No extra categories are added speculatively. Future additions should have a
distinct retrieval use case, a stable identifier, a definition, and evaluation
examples demonstrating why existing categories are insufficient.

## Shared interfaces

Create `src/internal/tools/search_domains.py`, independent of provider imports:

```python
@dataclass(frozen=True)
class SearchDomain:
    description: str
    query_hint: str

DOMAIN_REGISTRY: dict[str, SearchDomain]
AVAILABLE_DOMAINS: list[str]

def normalize_search_domain(value: str = "general") -> str: ...
def prepare_domain_query(query: str, domain: str = "general") -> str: ...
def search_domain_parameter() -> dict[str, object]: ...
```

Normalize domain strings with `strip().lower().replace("-", "_").replace(" ", "_")`.
Accept canonical names and forms such as `Social Media` and `social-media`.
Reject empty strings, unknown values, and non-strings (including explicit `None`)
with `ValueError` listing valid identifiers. Omission is handled by the default.

`prepare_domain_query` first validates the domain. For `general` or a whitespace-only
query it returns the original query unchanged. Otherwise it returns
`f"{query} {DOMAIN_REGISTRY[canonical].query_hint}"`.
For example, `("battery recycling", "academic")` produces
`"battery recycling academic research"`. Do not heuristically suppress a suffix
already present: enforce one application by call structure, not text matching.

`search_domain_parameter` returns a fresh JSON Schema string property containing
`enum`, `default: "general"`, and descriptions of all domains plus the explicit
statement that domain selection is a query hint, not a guaranteed result filter.

## Entry points and execution

### Function-calling tools

Add an optional `domain` property to the schemas of `MultiQueryWebSearchTool` and
`build_search_tool`. Existing required query fields remain unchanged.

For multi-query execution, normalize/validate the domain before the empty-query
return. Apply existing query sanitation, then transform each query once before
passing it to `_search_fn`. Preserve the injected callable's signature: do not
pass a new `domain` keyword into it. Preserve result formatting and URL deduplication.

Keep existing metadata for general searches. For a non-general search, retain
the sanitized original `queries`, add the canonical `domain`, and add
`executed_queries` so callers can inspect the transformation.

For the single-query tool, change its local callable to
`async def search(query: str, domain: str = "general") -> str` and prepare the
query before calling the existing `search_for_tool_string`. Preserve the
FunctionTool text/raw/metadata contract.

### MCP public-web search

Add `domain: str = "general"` to `search_web`. Normalize and prepare the query
before the provider try/except, allowing invalid input to surface as a tool error
rather than an empty search result. Send the executed query to every provider
branch. Preserve the original `query` in responses; on non-general searches add
`domain` and `executed_query` to both success and provider-exception responses.
Document accepted identifiers in the MCP tool description, generated from the
registry before registration to prevent a second manually maintained list.

Do not change MCP indexed-document search or introduce a domain discovery tool.

### Cache, fallback, and permissions

Lower-level provider functions, `search_tool`, and the cascade receive prepared
strings and do not add hints. Existing cache keys therefore distinguish different
executed queries; identical executed queries may safely share a cached result.
Both cascade legs see exactly the same prepared query. Provider errors retain
their current reporting behavior. Existing retrieval ACL filters remain intact.

## Acceptance criteria and verification

1. Registry includes exactly the requested identifiers, order, meanings, and hints.
2. Default/general calls preserve existing arguments, responses, and metadata.
3. Non-general hints reach single-query, multi-query, and MCP provider calls once.
4. Invalid domains fail before dispatch, including empty multi-query calls.
5. Injected callables with the original explicit signature continue to work.
6. Multi-query metadata distinguishes requested and executed queries.
7. MCP success and error responses preserve the original query and selected domain.
8. Cache and fallback tests demonstrate separation/reuse using executed queries.
9. Search-tool category flags, deduplication, and authorization regressions pass.
10. Documentation names supported entry points and explains all taxonomy overlaps.

Use mocked provider calls for deterministic unit tests; no credentials required.
Run the targeted suites listed in the companion plan. Live relevance evaluation
is optional follow-up, not evidence furnished by mocked tests. Before describing
the feature as improving quality, compare general/domain runs on labeled queries
for all 17 categories, including overlapping topics, measuring relevance and
empty-result rates with the same provider configuration.

## Rollout and limits

This increment makes domains usable through existing callable tools and MCP web
search. The HTTP/UI experience and the vector/web search-agent component do not
gain domain selectors. This boundary avoids silently accepting ignored fields.
Later HTTP/UI integration should be planned against the completed shared API.
Hints do not guarantee source authority, freshness, topic membership, or access
to private social content. The initial hints are transparent defaults that can
be revised after evaluation; adding native vertical providers is a separate change.
