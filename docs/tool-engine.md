# Tool engine

[← Back to README](../README.md)

This guide covers the tool agent: multi-turn function calling with structured
tool dispatch. For the full routing contract, see
[API request routing](request-routing.md); for exposing tools to external MCP
clients, see [MCP server](mcp.md).

## Capabilities

- **Structured tool dispatch** — `ToolAgentLoop` (`src/agents/tool/tool_calling.py`)
  runs a generic multi-turn function-calling loop over registered tools, threading
  tool results back into the conversation.
- **Routing and memory** — tool requests are auto-routed and carry bounded session
  history, so follow-ups keep context across turns.
- **Registered tool catalog** — a `ToolRegistry` seeded at startup, extensible at
  runtime with OpenAPI-backed tools.

## Routing into the tool engine

With `mode` omitted, `/api/agent` classifies each request as `chat`, `search`, or
`tool`. Action commands such as “send”, “deploy”, or “create a ticket” route to
`tool`, which runs `ToolAgentLoop`; when no tool or policy model is available it
falls back to grounded chat. The explicit `tool_agent` mode selects the loop
directly. See [API request routing](request-routing.md) for the full decision
order and response metadata (tool calls are surfaced as `tool_calls`).

## Dedicated tool-agent surface (`/tool/*`)

Diagrams in the README:
- [Tool-calling sequence](../README.md#tool-calling-sequence): generate, then
  approval, then parallel execution, then recovery or escalation, then the SSE
  events.
- [Tool-call states](../README.md#tool-call-states): each call's path to its
  `TaskStatus`.

Beyond the unified `/api/agent`, the tool engine has its own conversational
surface, parallel to `/search/*` and `/chat/*`:

- `POST /tool/send-tool-message` — runs `ToolAgentLoop` and streams Server-Sent
  Events: `progress` (per turn), `tool_call` (each completed call), `answer`,
  and a final `done`. Requires a local model (`SEARCH_AGENT_MODEL` /
  `SEARCH_AGENT_SERVER_URL`); returns **400** otherwise. Pass `stream:false` for
  a single JSON response.

  The response and the `done` event carry `truncated`. It is `true` when a
  generation hit the wall-clock stop (`AGENTIC_SEARCH_GENERATION_TIMEOUT`) and
  the answer is a fragment — otherwise a cut-off answer is indistinguishable
  from a complete one. They also carry `tool_recovery` (see
  [When a tool call fails](#when-a-tool-call-fails)).

  If the local model becomes unreachable mid-request (connection error,
  timeout or open circuit), the endpoint answers from a corpus-only search
  under the caller's ACL instead of failing: the answer lists what was found,
  and the response and `done` event carry `degraded: "model_unavailable"`.
  Any other failure still reports `error`.
- `GET /tool/tool-history` — past sessions for the caller (session proxy, like
  `/search/search-history`).

The router (`create_tool_router`, `src/internal/servers/query_and_chat/tool_backend.py`)
reuses the shared loop runner in `src/internal/servers/web/tool_agent_runner.py`.
In the web UI, the **Tool Agent** tab drives this endpoint and renders a running
transcript: each turn interleaves its live tool-call trace, and gated tools show
an inline approval prompt (Approve / Deny) before they run.

Gated tools prompt for approval on the streaming path, at parity with the
auto-router: the endpoint emits `approval_required` events through the
`ToolApprovalBroker` and waits for the user's decision via the shared
`POST /api/agent/approvals/{approval_id}` endpoint (authenticated users only).
A failed side-effecting call is escalated the same way, through
`escalation_required` events and `POST /api/agent/escalations/{escalation_id}`.

## Tool registry and discovery

The **`ToolRegistry` is the single source of truth** for the web/agent process's
runnable tools. `src/internal/tools/knowledge_base.py` provides the built-in seed set, and
`seed_tools(tool_registry)` loads it at web startup; OpenAPI tools are added at
runtime via `register_from_openapi`.

- `discover_tools(request)` returns the tools most relevant to a natural-language
  request, using a two-stage TF-IDF match (rank servers, then tools).
- `default_tool_catalog()` reads the registry at call time, so it reflects
  whatever is registered. Built-in tools group into a `local` server; each OpenAPI
  provider gets its own server.

Built-in seed tools: `web_search`, `search`, the nine keyless public
data-source tools (see [README.md](../README.md#built-in-public-data-tools)),
the four native domain tools `search_domain`, `get_sub_domains`,
`extract_page` and `batch_search` (see
[native domain features](search-engine.md#native-domain-search-features)),
and `rag_routing_tool` when an LLM is configured. Discovery is a ranking aid,
not a dispatcher.

### Not every registered tool is offered to an agent

`ToolEntry` carries two registration properties that decide who may call a tool,
so the decision travels with registration instead of being re-derived at each
call site:

- **`agent_callable=False`** — listed and invocable through `/admin/tools` and
  the Dev Console, but never handed to an agent loop. The seeded `search` is
  registered this way: it is built at process start, where no request identity
  exists, so it could only ever hold an unfiltered view of the corpus.
  `rag_routing_tool` is excluded too, because it generates a whole answer rather
  than returning evidence. So are `search_domain`, `get_sub_domains` and
  `batch_search`: every capability they route to is a public-data tool the agent
  already has, so offering them gave a small model two paths to the same nine
  tools. `extract_page` stays offered, since nothing else seeded fetches a URL.
- **`user_scoped=True`** — withheld when the request has no user, so an
  anonymous caller cannot write into a shared bucket. Set at MCP registration
  from `AGENTIC_SEARCH_MCP_USER_SCOPED`.

The tool agent therefore builds its **own** corpus `search` per request, bound to
that caller's ACL — it sends the filters to retrieval *and* drops anything that
fails them on the way back, because a backend may ignore the field. An agent loop
can never receive the unfiltered seeded instance, by construction rather than by
each call site remembering.

### `web_search` fetches the real web

The seeded `web_search` tool uses a sequential cascade: SerpAPI first
(`SERP_API_KEY`), falling back to the browser search server
(`AGENTIC_SEARCH_BROWSER_SEARCH_URL`, a `/retrieve`-shaped playwright server)
when SerpAPI is empty or unavailable. This applies to both the Tool Agent tab
and the `/api/agent` tool path.

`web_search` takes one to five `queries`, runs them concurrently, and returns a
JSON list of `{title, content, url}` deduplicated by URL. If some queries
succeed, their results are returned and the failures dropped. When no query
produces a usable result the call is a typed failure carrying the first error
(`{"error": ...}`) rather than an empty result, because "No results found." is
indistinguishable from a working search over a topic with no hits — a missing
key, an exhausted quota (SerpAPI's free tier is 100 searches/month, after which
it returns `429`) and an unreachable browser server all used to look identical.
In the tool agent that failure makes `web_search` unavailable for the rest of
the turn (see [When a tool call fails](#when-a-tool-call-fails)), so the model
sees that it is unavailable rather than the error text.

Each leg sits behind a circuit breaker (`serpapi`, `browser_search`): after
repeated failures the leg is skipped with a "circuit open" error for a while,
then one probe call is let through
([`[circuit_breaker]`](configuration/timeouts.md#circuit_breaker)). A
failed live lookup is answered from a stale serving-cache entry, marked
`metadata.stale: true`, while one is inside
`AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS`.

To run the fallback leg:

```bash
python3 -m src.internal.servers.web_search.browser --port 8003
# then, on the web backend:
AGENTIC_SEARCH_BROWSER_SEARCH_URL=http://localhost:8003/retrieve
```

It needs no API key and drives a real browser, so expect ~30-50s per query and
weaker relevance than SerpAPI. Use 8003 rather than its 8000 default: 8000 and
8001 are the retrieval servers, 8002 the reranker.

## Tool contract and argument validation

Every tool on the global registry declares `effect` (`READ_ONLY`,
`SIDE_EFFECTING` or `UNSPECIFIED`), `result_kind` (`DOCUMENTS`, `JSON` or
`TEXT`), `citeable` and `retries_internally`. The registry refuses a tool that
leaves `result_kind` undeclared, is `UNSPECIFIED` without coming from MCP, or
is `citeable` without returning `DOCUMENTS`: `POST /admin/tools/openapi`
answers **422**, and MCP discovery logs and skips the tool. An OpenAPI
operation is `READ_ONLY` for GET/HEAD/OPTIONS and `SIDE_EFFECTING` otherwise.

Arguments are validated against the tool's full JSON Schema (Draft 2020-12)
before the tool runs, on every path through the registry: the tool agent,
`/admin/tools/{name}/invoke`, MCP-mirrored tools and memory curation.
Ranges, enums, patterns, array lengths and closed objects are enforced, and an
invalid call is **rejected, not clamped**, with a message naming the argument —
`Argument 'limit': 500 is greater than the maximum of 10`, or
`Arguments: Additional properties are not allowed ('limti' was unexpected)` for a
misspelled key. The built-in tools declare their limits as schema keywords, so
for example `web_search.queries` takes at most 5 items and a `domain` must match
an allowed value exactly. A schema the validator cannot use (malformed, or an
unresolvable `$ref`) falls back to the old required-keys-and-types check with a
warning.

## When a tool call fails

Recovery is always on, owned by `ToolAgentLoop`, and ordered **retry →
degrade → escalate**:

- **Bad input goes back to the model.** A schema violation, or an input the
  tool itself rejects (an unknown ticker or place),
  returns the error text to the model as `invalid_arguments` so it can correct
  the call.
- **Retry.** A transient failure (HTTP 429 or 5xx, a timeout, a connection
  error) of a `READ_ONLY` tool is
  retried up to 2 times with jittered backoff, honouring `Retry-After` up to
  4 s, within a 10 s retry budget shared by the whole run. A tool that declares
  `retries_internally` — the corpus `search` and the public-data tools that
  retry their own GETs — is not retried again.
- **Degrade.** A `READ_ONLY` tool that still fails becomes unavailable for the
  rest of the turn: the model is told to answer from what it has, later calls
  to it are short-circuited, and the answer ends with
  `Note: <tool> was unavailable, so this answer may be incomplete.`
- **Escalate.** A failed `SIDE_EFFECTING` or `UNSPECIFIED` call is never
  replayed automatically. The run pauses with an `escalation_required` event
  and the Assist and Tools pages offer **Retry**, **Skip** or **Cancel**. An
  unanswered escalation (120 s), the per-run escalation cap (3), or a caller
  that cannot be asked (non-streaming JSON, an anonymous caller) stops the run
  as `unresolved` — never as a success.

`/tool` responses and `/api/agent` response metadata report what happened in
`tool_recovery`:
`outcome` (`recovered`, `degraded`, `cancelled` or `unresolved`),
`needs_user`, `retries`, `degraded` and `escalations`. The retry and
escalation limits are the `[tool_loop]` and `[tool_loop.recovery]` tables in
[timeouts](configuration/timeouts.md).

## Inspecting the registry

Three UI surfaces read the registry, and it is worth knowing which is which —
they were all once called "Tools":

| Surface | Component | Reads | For |
|---|---|---|---|
| `/tools` page | `ToolAgentView` → `ToolCatalog` | `/admin/tools` | The agent, plus the inventory it chooses from |
| Header **Manage tools** (wrench) | `ToolAdminPanel` | `/admin/tools` | Register/delete OpenAPI providers, test-invoke a tool |
| `/assist` → Dev Console | `debug/ToolCatalogPanel` → `ToolCatalog` | `/api/debug/tools` | Dev observability |

`ToolCatalog` is shared by the first and third: one renderer, two data sources.
It badges `agent_callable: false` as "not offered to agents" and
`user_scoped: true` as "needs sign-in" — usually the answer to "the tool is
registered, so why did the agent ignore it".

### Endpoints

Admin-gated (`require_admin`), always mounted:

- `GET /admin/tools` — every registered tool with `parameters`, `source`,
  `provider_id`, `agent_callable`, `user_scoped`, `effect`, `result_kind`, and
  `citeable`. Group by
  `source`/`provider_id` to get the catalog shape.
- `POST /admin/tools/discover` — ranks tools for a `query` via the TF-IDF
  `SemanticRouter`, returning the per-stage routing details. No LLM.
- `POST /admin/tools/{name}/invoke` — run one tool directly with arguments.
  Note it invokes the registry, so test-invoking `search` uses the seeded
  unfiltered instance and can return more than an agent run would. Invalid
  arguments, and a tool that raises, come back in `errors` rather than as a 500.

Dev-only, **mounted only when `AGENTIC_SEARCH_DEBUG_PANELS` is set** — these do
not exist in a normal deployment, which is why the `/tools` page uses the admin
endpoints instead:

- `GET /api/debug/tools` — `registered` plus the `catalog` already grouped by
  server. An empty registry yields empty lists, never a 500.
- `POST /api/debug/tools/discover` — the same ranking as the admin twin.

### Ranking is not how the agent chooses

`SemanticRouter` is only reachable through those two discover endpoints; nothing
in the agent path calls it. `ToolAgentLoop` is handed every agent-callable tool
and the model picks one itself, so a tool ranking first here is not evidence the
agent will call it. Both discover endpoints are diagnostics.

## Relationship to MCP

The [MCP server](mcp.md) exposes a set of tools to external MCP clients (Claude
Desktop, Cursor, etc.). That selection is client-driven and independent of this
auto-router: an MCP client invokes an exposed tool by its own policy and does not
pass through the web backend's routing.

Both directions exist. The `dynamic.py` bridge mirrors registry tools **into**
MCP via `sync_tool_to_mcp(name)`. In the other direction, this process can run as
an MCP **client**, so tools from a configured MCP server become ordinary
`ToolRegistry` tools — callable by the tool agent and listed by
`default_tool_catalog()`. See [MCP](mcp.md) for the client configuration and the
guards that keep a server from being offered its own tools back.
