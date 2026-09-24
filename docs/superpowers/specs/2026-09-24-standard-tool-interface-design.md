# Standard tool interface — design

## Problem

Every tool in `ToolRegistry` subclasses `Tool` (or wraps a function in
`FunctionTool`), but nothing standardises what a tool must declare or how it
reports failure. Today there are:

- **Five failure conventions.** A typed `ToolErrorText` (public-data, via
  `guarded`); a plain-string `{"error": ...}` (`routing_tools.py` corpus `search`
  and `rag_routing_tool`), which the recovery loop records as **success**; prose
  `"Error: ..."` (`web_search`); raising (OpenAPI tools, MCP transport errors); and
  success-shaped text for a logical failure (memory tools' "not found").
- **Undeclared or contradictory metadata.** `web_search` never declares `effect`,
  so it is `UNSPECIFIED`: it needs approval and is denied to callers without an
  approval channel, and its failures escalate instead of degrading. It is
  `citeable` but returns prose, which the source-card code cannot read. `stopping`
  is declared on every tool and read nowhere. `built_in_tools.CITEABLE_TOOLS_NAMES`
  duplicates `citeable` as a name set that lists a non-existent `open_url` and
  misses four citeable tools.
- **Misclassified raises.** `_failure_from_exception` treats every
  `aiohttp.ClientError` as transient, and `ClientResponseError` is one, so an
  OpenAPI 404 is retried. `DomainSearch` raises `ValueError` for bad arguments,
  which becomes `unknown` (degrade) instead of `invalid_input` (feed back);
  `extract_page` alone has a one-off wrapper for it. httpx transport errors from
  the MCP client are `unknown`.
- **Retry ownership inferred, not declared.** The recovery loop learns that a
  provider already retried only if the failure carries `provider_attempts`; the
  corpus `search` tool's client retries 3× and says nothing.
- **Parallel abstractions** that duplicate the real one: `ChatTool`
  (`tools/interface.py`, no implementation), the `built_in_tools` name sets and
  empty `TOOL_NAME_TO_CLASS`, and `tool_evidence.ToolSafety` (a copy of
  `ToolEffect`).

This spec standardises every **registry** tool on one declared contract,
enforced at registration, and retires the duplicates.

**Out of scope:** the MCP server's own FastMCP tools and dynamic mirror (a
separate process with its own client contract — follow-up); `tool_evidence`'s
separate registry protocol (no production caller); all timeout and retry
**values** (sub-project B: configuration files); `Tool.execute`'s
`(response_text, raw, metadata)` signature, which four callers depend on.

## 1. The contract

### Declarations

Every tool declares four properties. `Tool` gives none of them a silent default
that registration would accept:

| declaration | type | rule |
|---|---|---|
| `effect` | `ToolEffect` (`READ_ONLY`, `SIDE_EFFECTING`, `UNSPECIFIED`) | `UNSPECIFIED` only for tools registered with `source="mcp"` (remote behaviour is unknowable). |
| `result_kind` (new) | `ResultKind` (`DOCUMENTS`, `JSON`, `TEXT`) | Must be set. `DOCUMENTS` = the success response is a JSON array of objects with string keys `title`, `content`, `url`. `JSON` = any other JSON value. `TEXT` = free text. |
| `citeable` | `bool` | `citeable` ⇒ `result_kind is DOCUMENTS` (the only shape the source-card extractor reads). |
| `retries_internally` (new) | `bool` | `True` when the tool's provider already retries transient failures. |

`ResultKind` and the new properties live in `src/internal/tools/base.py` beside
`ToolEffect`. On `Tool`, `effect` keeps returning `UNSPECIFIED` and `result_kind`
returns `None` by default, so an undeclared subclass is caught by registration,
not silently accepted. `FunctionTool` and `FunctionTool.from_fn` take
`result_kind` and `retries_internally` keyword arguments; `result_kind` has no
default there either (`None` until set).

### Failure convention

A tool reports a failure in exactly one of two ways:

1. return a `ToolErrorText` carrying a `ToolFailure`, or
2. raise, and let `ToolRegistry.invoke_detailed` classify the exception.

Never: a plain-string `{"error": ...}`, a whole-result `"Error: ..."` prose, or a
success-shaped description of a failure. A **partial** success (some items
failed, some succeeded) is a success: it returns what it got and may note what
is missing inside the result.

### Classifying raised exceptions

`_failure_from_exception` (`registry.py`) becomes:

| exception | category |
|---|---|
| `InvalidToolInput` (new, `base.py`) | `invalid_input` |
| `aiohttp.ClientResponseError` 400, 404, 422 | `invalid_input` |
| `aiohttp.ClientResponseError` 429, 5xx | `transient` |
| `aiohttp.ClientResponseError` other 4xx (incl. 401, 403) | `permanent` |
| other `aiohttp.ClientError`, `asyncio.TimeoutError`, `TimeoutError`, `ConnectionError` | `transient` |
| `httpx.TransportError` (only if httpx is importable) | `transient` |
| anything else | `unknown` |

`InvalidToolInput(message)` is what any tool raises for a bad argument; its
failure message is the exception's message (it describes the caller's own
input, and it goes back to the model, never to a user card).

### Retry ownership

`ToolAgentLoop` passes the tool's `retries_internally` to the recovery policy:
a transient failure of a tool that retries internally is treated as already
retried (no outer retry, degrade). This replaces inferring ownership from
`ToolFailure.provider_attempts` alone; `provider_attempts` stays on the failure
for reporting.

### Enforcement

`ToolRegistry.register(tool, *, source=..., ...)` raises
`ValueError("tool <name>: <rule>")` when:

- `result_kind` is `None`;
- `effect is UNSPECIFIED` and `source != "mcp"`;
- `citeable` and `result_kind is not DOCUMENTS`.

`register_from_openapi` and `register_from_schema` go through `register`, so
dynamic tools are checked too. The registry `tool()` decorator (method and module
shorthand) forwards `effect`, `result_kind`, `citeable` and `retries_internally`.

## 2. Per-tool changes

| tool | change |
|---|---|
| `web_search` (`MultiQueryWebSearchTool`) | `effect=READ_ONLY`, `result_kind=DOCUMENTS`, `retries_internally=False`. Success returns a JSON array of `{title, content, url}` from all queries' pages (content = page summary; de-duplicated by URL, first occurrence wins). When every query fails, return `ToolErrorText` with category `unknown`. Queries that fail while others succeed are left out of the array. `raw` and metadata unchanged. |
| corpus `search` (`routing_tools.build_search_routing_tool`) | `DOCUMENTS`, `retries_internally=True`. When no page succeeds and at least one errored, return `ToolErrorText(json.dumps({"error": <first error>}), ToolFailure(transient, "search backend unavailable", provider_attempts=<client max_retries>))`. |
| `rag_routing_tool` | Stop catching exceptions (the registry classifies them); `JSON`. |
| 9 public-data tools | Declare `result_kind` (`search_wikipedia`, `search_arxiv`, `search_wayback`: `DOCUMENTS`; the rest `JSON`). `retries_internally=True` for the eight GET-based tools (`_fetch` retries GETs 3×); `search_nearby_places` POSTs to Overpass, which `_fetch` never retries, so it declares `False`. |
| domain tools (`search_domain`, `get_sub_domains`, `extract_page`, `batch_search`) | `DomainSearch` raises `InvalidToolInput` instead of `ValueError` for bad arguments; `guarded` maps `InvalidToolInput` to `invalid_input` with the same `{"error": ...}` text; the `extract_page` wrapper is removed. `batch_search` keeps per-query errors inside a successful result. `result_kind`: `extract_page` `DOCUMENTS`; the others `JSON`. `retries_internally=False`. |
| OpenAPI tools (`ApiRequestTool`) | Keep raising; classification above fixes categories. `result_kind=JSON`, `retries_internally=False`; effect by method unchanged. |
| MCP client tools | Stay `UNSPECIFIED` (allowed for `source="mcp"`); `result_kind=TEXT`; `retries_internally=False`. |
| memory tools | "not found" and "empty" become `ToolErrorText` `invalid_input` (the model corrects); `result_kind=TEXT`; `retries_internally=False`. |
| `build_search_tool` (prose corpus `search`) | Deleted, with its export; no caller in `src/`. |

**Behaviour changes to note:**

- `web_search` no longer needs approval. Anonymous and non-streaming callers
  have no approval channel, so the tool agent could never web-search for them;
  now it can. A total web failure degrades quietly instead of escalating.
- The model sees `web_search` results as a JSON document list instead of
  per-query prose sections; web results also produce source cards. JSON-array
  truncation keeps the best-ranked items.

## 3. Retirements and visibility

- **`ChatTool`** (`src/internal/tools/interface.py`) is deleted. Its only use is
  a type in `chat/tool_call_args_streaming._get_tool_class`, which looks names
  up in the empty `TOOL_NAME_TO_CLASS`, so `maybe_emit_argument_delta` returns
  on its first line for every call. Delete `_get_tool_class` and
  `maybe_emit_argument_delta`, and in `chat/llm_step.py` remove its import, the
  `arg_parsers` dict and the `yield from maybe_emit_argument_delta(...)` block.
  Keep the module's `Parser` export if `llm_step` still needs it for anything
  else; otherwise remove that import too.
- **`built_in_tools`**: delete `CITEABLE_TOOLS_NAMES`, `STOPPING_TOOLS_NAMES`
  and `TOOL_NAME_TO_CLASS` (the module is left empty and deleted if nothing else
  remains). `observability/admin_surface` counts citeable tools from
  `tool_registry` (`sum(t.citeable for t in tool_registry.list_tools())`).
  `tests/unit/test_built_in_tools.py` is updated to assert the retirement.
- **`stopping`** is removed from `Tool`, `FunctionTool` and `from_fn` (read
  nowhere).
- **`ToolSafety`** (`context/tool_evidence.py`) becomes an alias of `ToolEffect`
  (identical values), keeping `src.context.ToolSafety` importable.
- **Visibility**: the `/admin/tools` tool view (`servers/tools/api.py` `ToolView`
  and `ToolRegistry.tool_summary`) gains `effect`, `result_kind`, `citeable` and
  `retries_internally`.

## Testing

- **Registration rules:** each rule rejects a violating tool with a message naming
  the tool and the rule; an `UNSPECIFIED` tool registered with `source="mcp"` is
  accepted.
- **Conformance:** every tool the app registers passes `register` — the seeded
  `tool_knowledge_base()` set, the request-bound corpus search, the memory tools,
  a sample OpenAPI tool and a sample MCP tool. Every `DOCUMENTS` tool's success
  response (with network faked) parses as a list of `{title, content, url}` with
  string values.
- **Classification:** `InvalidToolInput` → `invalid_input`; `ClientResponseError`
  400/404/422, 401/403/other 4xx, 429/503 → `invalid_input`, `permanent`,
  `transient`; other `ClientError` and timeouts → `transient`; httpx
  `TransportError` → `transient`; anything else → `unknown`.
- **Per tool:** corpus search outage → typed transient failure (previously a
  success string); `web_search` all queries failed → typed failure, partial
  failure → success array without the failed queries; memory not-found →
  `invalid_input`; `search_domain` bad argument → `invalid_input` with the same
  text; `extract_page` bad URL → `invalid_input` without the wrapper.
- **Behaviour:** `web_search` runs without an approval callback; a
  `retries_internally` tool's transient failure is not retried by the loop; a
  corpus-search failure reaches the model as a failure.
- **Retirements:** `ChatTool`, the name sets, `stopping` and
  `maybe_emit_argument_delta` are gone; `llm_step` streaming still works (its
  existing tests pass); admin surface counts citeable tools from the registry.
- Every new test mutation-checked; the unit suite passes with torch blocked.
