# MCP server

[← Back to README](../README.md)

This guide explains how to install, run, configure, and connect to the Agentic Search MCP server.

## Overview

The MCP server exposes Agentic Search capabilities as [Model Context Protocol](https://modelcontextprotocol.io/) tools, letting any MCP-compatible client (Claude Desktop, Cursor, etc.) query your knowledge base directly. It delegates data access to the main API server, which manages access controls.

## Install and run

Install the `mcp` extra and launch the HTTP server on port 8090:

```bash
pip install -e ".[mcp]"
uvicorn src.internal.mcp_server.api:mcp_app --port 8090
```

The server also launches with the `Run All Services` task from the default `launch.json`, and can be launched independently through the VS Code debugger.

## Authentication and transport

Provide a Personal Access Token or API key in the `Authorization` header as a Bearer token. FastMCP verifies the credential, and indexed-document tools forward that request token only to the authenticated web search endpoint. The web backend—not the MCP caller—derives the user's and groups' access-control filters. A caller may narrow retrieval further with document sets, but cannot broaden the server-derived ACL.

```text
MCP bearer token → FastMCP verification → authenticated web search endpoint
→ server-derived user/group ACL + optional document-set narrowing
→ internal retrieval → MCP result or grounded synthesis
```

Authentication, authorization, transport, and backend failures fail closed. They return an error or empty result and never trigger an unfiltered raw-retrieval fallback.

- **Transport:** HTTP POST (MCP over HTTP)
- **Port:** 8090 (shares the API server's domain)
- **Framework:** FastMCP with a FastAPI wrapper
- **Database:** none; all work delegates to the API server

OAuth and stdio transport may be supported in the future.

## Configure MCP clients

### Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS:

```json
{
  "mcpServers": {
    "agentic-search": {
      "type": "http",
      "url": "http://localhost:8090/",
      "transport": "http",
      "headers": { "Authorization": "Bearer YOUR_TOKEN_HERE" }
    }
  }
}
```

For a remote deployment, replace the URL with `https://[YOUR_DOMAIN]:8090/`. Other MCP clients generally support HTTP transport with custom headers; refer to the client's configuration documentation.

## Tools available to the LLM client

| Tool | What it does |
|------|-------------|
| `search_indexed_documents` | Search the private knowledge base with optional document-set narrowing |
| `search_web` | Web search via Google Custom Search, SerpAPI, or Serper, with optional topic hints |
| `open_urls` | Fetch full page text from a list of URLs |
| `get_sub_domains` | Discover local domain routes and their real parameter schemas |
| `search_domain` | Route a query through existing web search or a supported public-data tool |
| `extract_page` | Extract readable page text with a configurable character limit |
| `batch_search` | Run 1–5 domain searches with ordered results and per-query errors |
| `ask_agentic_search` | Synthesizes an answer from authenticated evidence; validates citation labels and answer/evidence overlap |
| `retrieve_documents` | Returns authenticated document content and relevance scores without answer synthesis |
| `expand_query` | LLM-backed keyword expansion for improved recall |

The memory tools (`save_memory`, `search_memories`,
`update_memory_from_conversation`, …) resolve the caller from the bearer token's
`sub`. Without one they fall back to a **shared** `default_user` bucket — fine
for a single-operator local setup, wrong for a shared deployment. Setting
`AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH=1` makes them refuse an unauthenticated
caller instead. The same variable governs the web backend's `/api/memory/*`
routes; MCP is a second door to the same store, so honouring it in only one
place would leave the other unguarded.

MCP tool selection is independent of the web UI's `/api/agent` auto-router. An MCP client explicitly invokes an exposed tool according to its own model and client policy; it does not pass through the web backend's internal → SerpAPI → browser fallback sequence. For the web API contract, see [API request routing](request-routing.md).

Grounding verification checks citation labels and lexical overlap with retrieved evidence. It reduces unsupported output but is not a hard guarantee that an answer contains no hallucinations.

Dynamic tools registered via `FunctionTool` / `ApiToolRegistry` can be mirrored to MCP by calling `sync_tool_to_mcp(name)` after registration (`src/internal/mcp_server/tools/dynamic.py`).

## Web-search domain hints

`search_web` accepts an optional `domain` from the
[search topic taxonomy](search-engine.md#search-topic-domains). For example:

```json
{"query": "battery recycling", "limit": 5, "domain": "academic"}
```

The provider receives `battery recycling academic research`. A non-general
response includes the original and executed queries; an empty result looks like:

```json
{
  "results": [],
  "query": "battery recycling",
  "domain": "academic",
  "executed_query": "battery recycling academic research"
}
```

The same metadata accompanies provider-exception responses. Omitted or `general`
domains preserve the existing response shape (`results` and `query`, plus `error`
when the provider raises). Invalid domain values produce a tool error before any
provider call. The tool description lists the accepted domains; MCP exposes a
string parameter and validates it at execution time.

Domain hints do not select providers or guarantee category membership.
`search_indexed_documents` does not accept a domain argument; its document-set
and authorization behavior is unchanged.

## Semantic tool discovery (server-side)

The **`ToolRegistry` is the single source of truth** for the web/agent process's
runnable tools. `src/internal/tools/knowledge_base.py` provides the built-in seed set, and
`seed_tools(tool_registry)` loads it into the registry at web startup; OpenAPI
tools are added at runtime via `register_from_openapi`. Discovery covers the
union of both:

- `discover_tools(request)` returns the tools most relevant to a natural-language
  request, using a two-stage TF-IDF match (rank servers, then tools).
- `default_tool_catalog()` is `catalog_from_registry(tool_registry)`, read at call
  time — so it reflects whatever is registered (empty until seeding runs). Built-in
  tools group into a `local` server; each OpenAPI provider gets its own server.

Built-in seed tools: `web_search`, `search`, `search_routing_tool` (and
`rag_routing_tool` when an LLM is configured).

This does not change how MCP clients invoke tools — MCP tool selection stays
client-driven, as described above. Discovery is a ranking aid, not a dispatcher.

**Relationship to the MCP tools:** the `@mcp_server.tool()` functions
(`search_web`, `open_urls`, `ask_agentic_search`, `retrieve_documents`,
`expand_query`, `search_indexed_documents`) are registered with FastMCP, not the
`ToolRegistry`, and several are bound to the MCP request's auth context. The
`dynamic.py` bridge mirrors the `ToolRegistry` **into** the MCP server
(`sync_tool_to_mcp`). Seeding is per-process; exposing the built-ins over MCP
would require seeding the MCP server process's own registry (a separate
follow-up). For the opposite direction — MCP tools becoming registry tools —
see below.

## Pulling MCP tools into the web process

The bridge above runs outward: it publishes our tools to MCP hosts. The web
process can also run as an MCP **client**, so tools from any MCP server become
ordinary `ToolRegistry` tools — callable by the tool agent, listed by
`/admin/tools`, invocable from the Dev Console.

Off by default. Configure servers as `name=url` pairs:

```bash
AGENTIC_SEARCH_MCP_SERVERS="agentic=http://127.0.0.1:8090/"
AGENTIC_SEARCH_MCP_TOKEN="<bearer token>"
```

The token must be one the web backend's `/me` accepts, since the MCP server
delegates authentication there. Multiple servers are comma separated; each gets
its own group in the Dev Console catalog. A server that cannot be reached is
logged and skipped — MCP is additive and never blocks startup.

Three things worth knowing:

- **No export loop.** Tools pulled in register with `source="mcp"`, which the
  outbound bridge skips, so a server is never offered its own tools back.
- **No recursion.** `ask_agentic_search` runs an agent, so the tool agent's
  shadow list excludes it. It stays callable through `/admin/tools`.
- **Discovery runs after startup, not during it.** Our MCP server authenticates
  by calling back into this process's `/me`, which does not answer until lifespan
  startup finishes. Discovery is a background task for that reason.

Remote tools carry `effect=UNSPECIFIED`, so the tool agent's approval gate
applies to them — only `READ_ONLY` tools are auto-approved. Configure only MCP
servers you trust: a server controls the tool descriptions the model sees.

## Resources

| Resource | What it exposes |
|----------|----------------|
| `indexed_sources` | Available retrieval source types based on configured API keys |
| `document_sets` | Document sets scoped for search |

## Debug with MCP Inspector

```bash
npx @modelcontextprotocol/inspector http://localhost:8090/
```

In Inspector, ignore the OAuth menus, open **Authentication**, select **Bearer Token**, paste the token, and connect. You can then browse tools, test calls, inspect payloads, and debug authentication.

Check server health with:

```bash
curl http://localhost:8090/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "mcp_server"
}
```

## Environment variables

| Var | Default | Description |
|-----|---------|-------------|
| `MCP_SERVER_ENABLED` | `false` | Enable the MCP server |
| `MCP_SERVER_PORT` | `8090` | MCP server port |
| `MCP_SERVER_CORS_ORIGINS` | — | Comma-separated allowed origins for CORS |
| `API_SERVER_HOST` | `127.0.0.1` | Host of the web backend |
| `API_SERVER_PROTOCOL` | `http` | Protocol for the web backend URL |
| `API_SERVER_URL_OVERRIDE_FOR_HTTP_REQUESTS` | — | Override the full web backend URL; takes precedence over protocol and host |
| `AGENTIC_SEARCH_MCP_SERVERS` | — | MCP servers the **web process** pulls tools from, as `name=url` pairs. Unset disables it |
| `AGENTIC_SEARCH_MCP_TOKEN` | — | Bearer token sent to those servers |


## Native domain operations

`get_sub_domains`, `search_domain`, `extract_page`, and `batch_search` share the
repository's `DomainSearch` implementation with the tool registry. No additional
provider, API key, or CLI is needed. See [native domain features](search-engine.md#native-domain-search-features)
for tags, aliases, schemas, examples, and limits.

`search_domain` returns query/domain/tag metadata and results, while
`batch_search` returns a `queries` array of these results or errors. For web routes,
MCP uses its existing `MCP_WEB_SEARCH_PROVIDER` setting (Google, SerpAPI, or Serper).
Specialized tags use the repository's public-data tools. Capability discovery is
local and makes no network requests. Invalid single calls surface as tool errors;
batch failures are reported per query.
