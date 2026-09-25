# Timeout and retry policies

Every agent-facing timeout and retry count is read from one place:
`src/internal/configs/timeouts.py` loads the bundled
`src/internal/configs/timeouts.toml`, deep-merges an optional operator file,
applies four legacy env vars, validates the result, and caches a frozen
`TimeoutPolicies` tree for the process. Sites call `get_timeout_policies()` at
construct or call time — never at import time. The file and env vars are read
once per process, on first use; changing either needs a restart.

## Precedence

Highest to lowest:

1. An explicit constructor/call kwarg at the site (e.g. passing
   `timeout_seconds=` directly) always wins.
2. The four legacy env vars (below).
3. The operator file named by `AGENTIC_SEARCH_TIMEOUTS_PATH`.
4. The bundled `src/internal/configs/timeouts.toml` defaults.

## Operator override file

Set `AGENTIC_SEARCH_TIMEOUTS_PATH` to a TOML file containing only the keys you
want to change; every key you omit keeps the bundled default, including keys
omitted from a table you do provide. An unknown key (such as a typo) or a value
that fails validation raises at load time and names the offending key — it
never silently falls back.

```toml
# /etc/agentic/timeouts.toml — only the keys you change
[llm]
socket_read_timeout_seconds = 60.0

[tool_loop.recovery]
max_retries = 1
```

## Legacy env vars

These four names predate the policy file and still win over it (but lose to
an explicit call-site kwarg):

| env var | overrides |
|---|---|
| `TOOL_APPROVAL_TIMEOUT_SECONDS` | `tool_loop.approval_timeout_seconds` |
| `AGENTIC_SEARCH_GENERATION_TIMEOUT` | `llm.local_generation_timeout_seconds` |
| `LLM_SOCKET_READ_TIMEOUT` | `llm.socket_read_timeout_seconds` |
| `AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS` | `sse.heartbeat_seconds` |

## Validation rules

- Every `*_seconds` timeout must be a finite number `> 0`, **except**
  `sse.heartbeat_seconds` and `llm.local_generation_timeout_seconds`, which
  accept `0` to mean disabled (no heartbeat / no wall-clock cap).
- Every integer count must be `>= 1` (it is an attempt count, and a retry loop
  indexed `range(max_retries)` would make no request at `0`), **except**
  `tool_loop.recovery.max_retries` and `llm.grounded_max_retries`, which
  accept `0` to mean no retries/regenerations.
- `backoff_seconds` lists must be non-empty lists of numbers `>= 0`; the retry
  loop reuses the last element once the attempt index exceeds the list.

## Schema

One table per section of `timeouts.toml`; defaults are today's values (no
behaviour change).

### `[tools.public_data]`

| key | default | bounds |
|---|---|---|
| `timeout_seconds` | 10.0 | one keyless public-data HTTP call |
| `max_attempts` | 3 | GET only; a POST is never retried |
| `backoff_seconds` | `[0.4, 0.8]` | sleep before attempt 2, 3 (last step reused) |
| `retry_budget_seconds` | 15.0 | no new attempt starts after this wall time |
| `overpass_timeout_seconds` | 30.0 | `search_nearby_places` HTTP call |
| `overpass_query_timeout_seconds` | 25.0 | Overpass server-side `[timeout:N]` |

### `[tools.web_search]`

| key | default | bounds |
|---|---|---|
| `google_timeout_seconds` | 15.0 | `google_custom_search` |
| `serpapi_timeout_seconds` | 15.0 | `serpapi_search` and the cascade SerpAPI leg |
| `serper_timeout_seconds` | 10.0 | `serper_dev_search` |
| `fetch_page_timeout_seconds` | 15.0 | `fetch_url`, one page |
| `fetch_pages_timeout_seconds` | 10.0 | `fetch_pages_concurrently`, per page |
| `query_timeout_seconds` | 15.0 | `web_search` tool, per query |

### `[tools.search_router]`

| key | default | bounds | env override |
|---|---|---|---|
| `timeout_seconds` | 15.0 | `search_tool`, any provider | — |
| `max_retries` | 3 | attempts; also the corpus search tool's reported attempts | — |

### `[tools.openapi]`

| key | default | bounds |
|---|---|---|
| `timeout_seconds` | 15.0 | one OpenAPI-registered tool call |

### `[tools.mcp]`

| key | default | bounds |
|---|---|---|
| `timeout_seconds` | 30.0 | MCP streamable-HTTP request |
| `sse_read_timeout_seconds` | 300.0 | MCP stream read |

### `[retrieval.client]`

| key | default | bounds |
|---|---|---|
| `timeout_seconds` | 10.0 | `SearchClient` default, `retrieval_search`, search agents |
| `max_retries` | 3 | attempts |
| `backoff_base_seconds` | 0.5 | sleep base, `base * 2**attempt` between attempts |

### `[retrieval.search_runner]`

| key | default | bounds |
|---|---|---|
| `timeout_seconds` | 15.0 | context retrieval `search_runner` |
| `max_retries` | 3 | attempts |

### `[retrieval.web_hybrid]`

| key | default | bounds |
|---|---|---|
| `provider_timeout_seconds` | 5.0 | `/api/agent` hybrid search, per provider request |
| `provider_max_retries` | 1 | attempts per provider |
| `provider_wait_seconds` | 8.0 | wall clock per provider leg |

### `[rerank]`

| key | default | bounds |
|---|---|---|
| `timeout_seconds` | 10.0 | HTTP cross-encoder reranker call |

### `[llm]`

| key | default | bounds | env override |
|---|---|---|---|
| `socket_read_timeout_seconds` | 120.0 | max gap between streamed chunks | `LLM_SOCKET_READ_TIMEOUT` |
| `remote_total_timeout_seconds` | 120.0 | remote (OpenAI-compatible) server call | — |
| `local_generation_timeout_seconds` | 120.0 | local generation wall clock, `0` = none | `AGENTIC_SEARCH_GENERATION_TIMEOUT` |
| `local_heartbeat_seconds` | 10.0 | local generation progress heartbeat | — |
| `sufficiency_timeout_seconds` | 5.0 | AgenticRAG sufficiency judge | — |
| `grounded_max_retries` | 1 | grounded-answer regenerations, `0` = none | — |

### `[tool_loop]`

| key | default | bounds | env override |
|---|---|---|---|
| `approval_timeout_seconds` | 60.0 | wait for a user's tool approval | `TOOL_APPROVAL_TIMEOUT_SECONDS` |
| `escalation_timeout_seconds` | 120.0 | wait for a user's failure escalation decision | — |
| `max_escalations` | 3 | escalations per run before it stops unresolved | — |
| `tool_evidence_timeout_seconds` | 5.0 | one tool call feeding answer evidence | — |

### `[tool_loop.recovery]`

| key | default | bounds |
|---|---|---|
| `max_retries` | 2 | extra attempts after the first, read-only transient only |
| `backoff_seconds` | `[0.5, 1.0]` | jittered × `U(0.5, 1.5)`; last step reused |
| `retry_after_cap_seconds` | 4.0 | cap on a provider's `Retry-After` |
| `retry_budget_seconds` | 10.0 | total retry sleep per run |

### `[sse]`

| key | default | bounds | env override |
|---|---|---|---|
| `heartbeat_seconds` | 15.0 | keepalive on an idle stream, `0` = off | `AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS` |

### `[circuit_breaker]`

One policy for every serving-dependency breaker (`serpapi`, `browser_search`,
`rerank`, `remote_llm:<base_url>` (one per inference server); see `src/internal/resilience/circuit_breaker.py`).
A breaker opens after `failure_threshold` consecutive failures — a transport
error, a timeout, HTTP 5xx or 429 — and then fails fast for `open_seconds`
before letting one probe call through. Breaker state is per process and is
reported under `circuits` by `GET /api/admin/metrics`.

| key | default | bounds |
|---|---|---|
| `failure_threshold` | 5 | consecutive failures that open a breaker, `>= 1` |
| `open_seconds` | 30.0 | how long an open breaker fails fast before one probe |
