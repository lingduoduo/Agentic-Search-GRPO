# Native search features

## Goal

Implement domain-aware search, local capability discovery, page extraction, and
ordered batch search using the repository's existing functionality. No additional
search service, credentials, provider adapter, or CLI is introduced.

## Architecture

- `search_domains.py` remains the single taxonomy and query-hint definition.
- `domain_search.py` composes existing public-data tools, the existing web
  cascade, `fetch_url`, and argument validation. It owns routing and batching.
- `domain_search_tools.py` adapts the shared methods to `FunctionTool`, using the
  existing JSON/error wrapper from the public-data tools.
- `knowledge_base.py` seeds the four feature tools and shares the existing web
  callback and public-data tool instances with the domain service.
- MCP's search module exposes the same operations using `DomainSearch`. Its web
  callback uses the existing MCP provider selection and `search_tool` dispatcher.

## Capabilities and discovery

Every domain offers `<domain>.web`. Those routes use current topic hints and do
not guarantee category membership. The specialized route table includes only
implemented, read-only public-data tools:

| Tag | Tool | Query argument |
| --- | --- | --- |
| general.wikipedia | search_wikipedia | query |
| academic.arxiv | search_arxiv | query |
| resource.wayback | search_wayback | url |
| finance.quote | get_stock_quote | symbol |
| finance.crypto | get_crypto_price | symbol |
| finance.currency | convert_currency | from_currency |
| environment.weather | get_weather | location |
| travel.location | search_location | query |
| travel.nearby | search_nearby_places | query |

`get_sub_domains(domains)` accepts one to five domain names and returns a local
catalog with descriptions, query formats, full tool parameter schemas, and extra
parameters. Schemas are copied from actual tool definitions so callers cannot
mutate the underlying tools. Discovery performs no network requests and does not
promise upstream availability. Routes to arbitrary registry entries or private
corpus tools are excluded.

## Shared operations

`DomainSearch` accepts injectable web-search, page-fetch, and public-tool inputs.
Its defaults reuse existing repository implementations.

- `search(query, domain=None, tag=None, params=None, max_results=5, ...)` returns
  original query/domain/tag metadata and results. Untagged searches use the
  domain's web route; tagged searches infer or validate the domain. Specialized
  queries are passed unchanged to the tool's documented query parameter.
- `extract(url, max_length=5000)` uses the existing fetcher and returns a document
  array. Accept HTTP(S) URLs and character limits from 1 through 50,000; surface
  fetch failures as errors. Existing extraction format support is preserved.
- `batch_search(queries, **shared_options)` runs one to five searches concurrently,
  returning one result/error per input in input order. Each item overrides shared
  defaults; route overrides remove conflicting shared tags. Inputs are not mutated.
  Shared params remain unless the item supplies params or its alias. Parent
  cancellation propagates to all workers.

`sub_domain` aliases `tag`; `sub_domain_params` aliases `params`. Conflicting
aliases, unsupported tags, incompatible domains, and unknown/missing/incorrectly
typed capability params fail before dispatch. Python params accept objects, JSON,
key=value, or {key:value} forms; structured tool schemas advertise object params.
The query argument is supplied from `query`; a duplicate param must agree.
Search result limits are capped at ten, including tool-specific limit arguments.

## Interfaces and compatibility

Native registry and MCP tool names are `search_domain`, `get_sub_domains`,
`extract_page`, and `batch_search`.

Registry search returns the capability's document array or facts object;
extraction returns a document array. Those two tools are citeable, and the existing
source-card collector only uses document arrays. Discovery returns a directory
object; batch returns `{"queries": [...]}` and is not citeable.

MCP search returns the service envelope, including executed query text for web
routes. MCP batch returns the ordered queries envelope. The existing `search_web`,
`open_urls`, corpus ACLs, and web-fallback contracts stay intact. All providers and
public-data transports retain their existing configuration and error handling.

## Validation and limits

Test real schema discovery, all-domain web routes, unchanged specialized queries,
required params, route allowlisting, result conversion, ordered mixed failures,
cancellation, extraction delegation, registry seeding/invocation, and MCP reuse.
Mock existing upstream helpers; no live service or relevance claim is required.
Run affected search/domain/cache/cascade/ACL/MCP/agent/seed regressions and Ruff.

This feature does not implement new specialized backends for unsupported domains,
new file-format extraction, new credentials, or HTTP/UI category selectors.
