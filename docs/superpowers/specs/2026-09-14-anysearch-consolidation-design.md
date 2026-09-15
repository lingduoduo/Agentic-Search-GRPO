# Consolidate the AnySearch sample into repository tools

## Scope and decisions

The user supplied a standalone AnySearch sample appended to
`src/internal/tools/search_domains.py` and requested consolidation into the
repository. The user explicitly excluded adding an AnySearch CLI. Implement the
same four service operations through Python methods and the existing tool
framework: search, capability discovery, extraction, and batch search.

The sample defines the REST contract for this change. No live request is needed
to verify consolidation, and unit tests must mock network access. The original
sample was backed up outside the repository before editing.

## Existing problems

- The appended sample defines `AVAILABLE_DOMAINS` a second time.
- Importing the taxonomy rewrites stdout/stderr and loads local dotenv values
  into the process environment.
- Synchronous requests, thread/queue batching, printing, and process exits mix
  application behavior with a module imported by search tools and MCP.
- Batch requests can discard `domain` silently while single requests validate it.
- A sample documentation command depends on template files absent from the repo.

## Architecture

| Module | Responsibility |
| --- | --- |
| `search_domains.py` | Existing pure registry, normalization, query hints, schemas |
| `anysearch.py` | Async REST client, request validation, configuration, structured errors, ordered batching |
| `search.py` | Convert REST results to `SearchPage`; add the optional AnySearch provider |
| `anysearch_tools.py` | Adapt all four native operations to `FunctionTool` and JSON results |
| `knowledge_base.py` | Seed native tools only when explicitly enabled |
| MCP `tools/search.py` | Support `MCP_WEB_SEARCH_PROVIDER=anysearch` for existing web search |

No CLI files, command parser, dotenv loader, JSON-repair command, or stream
rewrapping belong in the consolidated implementation. User documentation replaces
the sample's generated documentation command. Existing HTTP/UI search behavior
and provider fallbacks remain unchanged.

## Public contracts

`AnySearchClient(api_key=None, base_url=None, timeout_seconds=30)` reads runtime
configuration when constructed. Explicit arguments override environment values;
an explicit empty API key means anonymous access. `ANYSEARCH_API_BASE_URL`
defaults to `https://api.anysearch.com`. Every request has a finite timeout and
closes its session on success, failure, or cancellation.

- `search(query, **options)` → REST envelope from `POST /v1/search`.
- `get_sub_domains(domains)` → REST envelope from `GET /v1/sub-domains`, using
  repeated `domain` parameters for one to five normalized domains.
- `extract(url)` → REST envelope from `POST /v1/extract`; HTTP(S) URLs only.
- `batch_search(queries, **shared_options)` → ordered list of REST envelopes or
  per-item exceptions for one to five concurrent requests.

`AnySearchError` carries status, request ID, and response data. The client redacts
its API key from provider errors and does not expose raw transport exceptions.
Invalid input fails before its request is sent. One bad batch item does not
prevent valid siblings from returning; cancellation of the batch cancels workers.

Native capability tags have canonical domain prefixes, e.g. `finance.quote`.
The caller discovers available tags and required query/parameter formats first.
A native `domain` validates that prefix and requires a tag. It does not append
query hints. `sub_domain` aliases `tag`; `sub_domain_params` aliases `params`.
Native parameters accept dictionaries, JSON objects, key=value pairs, and the
sample's {key:value} format. Single and batch calls share normalization. Results
are capped at 10 per request. Batch fields override shared defaults; an item-level
tag replaces the whole inherited route. Caller dictionaries are not modified.

The existing `search`/`web_search` tools continue to append general-purpose topic
hints when given a domain; these tools select AnySearch by provider, without
pretending that hints are native capability routing. The AnySearch provider has
no page-based pagination in the sample contract, and rejects later pages. Its
responses bypass the shared public-web cache because credentials/endpoints can
change the accessible data.

## Tool registration and return shapes

`build_anysearch_tools(client=None)` returns four read-only FunctionTools.
`AGENTIC_SEARCH_ANYSEARCH_ENABLED=true` adds them through the existing seed path;
the default catalog is unchanged. Python callers may instead register them in a
specific `ToolRegistry`.

- `anysearch_search`: JSON array of `{title, content, url}` documents; citeable.
- `anysearch_get_sub_domains`: capability-directory JSON object; not citeable.
- `anysearch_extract`: JSON document array for the extracted page; citeable.
- `anysearch_batch_search`: `{"queries": [{query, results}, {query, error, ...}]}`
  in input order; not citeable because it is a nested response.

Provider failures return JSON errors with status/request ID. Validation failures
return JSON error messages. Canonical search-page conversion is shared with the
provider adapter. Client callers still receive the original envelopes and metadata.
MCP's existing `search_web` can select AnySearch; this change does not add four
new MCP-native endpoints or alter the dynamic bridge.

## Verification

Use focused tests for pure imports, request aliases, invalid native routing,
configuration precedence, REST envelopes and errors, cancellation, ordered batch
failures, shared option precedence, registry invocation, citation-compatible JSON,
opt-in seeding, provider adaptation, credential-sensitive cache bypass, pagination,
and MCP provider selection. Run existing search/domain/cache/ACL/MCP/seed regressions.
Run Ruff lint/format checks and Git whitespace checks. Mocked tests establish
repository behavior against the supplied contract, not external API availability
or search relevance.
