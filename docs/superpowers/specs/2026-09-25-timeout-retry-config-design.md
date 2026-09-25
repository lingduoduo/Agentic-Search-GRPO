# Timeout and retry policies in a configuration file — design

Sub-project B of "standardize the tool interfaces and put timeout/retry policies
into configuration files" (A shipped as #644).

## Problem

About 50 timeout/retry values sit as literals on the serving path. Only four are
tunable (env vars), about a dozen policies are defined in two to four places, and
some copies disagree:

- LLM socket read timeout: 120 s where it is used (`multi_llm.py:59`), 60 s in the
  unused `chat_configs.py:32` copy and in `default_config.py:76`.
- Tool approval timeout: defined in `AppSettings`, `ToolAgentLoopConfig`, a
  `getattr(..., 60.0)` fallback in `tool_agent_runner.py`, and `ToolApprovalBroker`.
- Tool escalation timeout (120 s): two copies, overridable nowhere.
- `search_runner.py` repeats 15 s / 3 retries three times; `routing_tools.py`
  hand-copies `search_tool`'s retry count as `_SEARCH_ATTEMPTS`.
- `agent_configs.py` holds 26 `AGENT_TIMEOUT_*` constants nothing reads.

**Goal:** every agent-facing timeout/retry policy lives in one versioned TOML file,
one key per logical policy, tunable by an operator without a code change.

**Decisions made with the user:** scope is agent-facing calls (tools, retrieval,
reranker, LLM, tool loop, SSE); format is TOML; **no behaviour change** — every site
keeps the value it runs with today, duplicates collapse, sites that differ on
purpose keep their own keys.

**Out of scope:** the standalone web-search servers (`servers/web_search/serp.py`,
`google.py`, `browser.py` — separate processes with their own CLI config), the MCP
server's document parser, the debug router, Redis/SQLite/locks, billing, license,
JWKS, hooks and poll intervals (infrastructure, not policy a request waits on);
a per-tool execution timeout in `ToolAgentLoop` (a behaviour change, follow-up);
unifying values across sites (e.g. one retrieval policy, one backoff shape).

## 1. File, loading and precedence

**Bundled file:** `src/internal/configs/timeouts.toml`, today's values, each with a
one-line comment on what it bounds. Packaged with `src`: `pyproject.toml`
`[tool.setuptools.package-data]` gains `*.toml` (today it ships only `*.json`), and
the loader reads it through `importlib.resources`, never a repo-relative path.

**Operator override:** `AGENTIC_SEARCH_TIMEOUTS_PATH` names a TOML file that may be
partial; it is deep-merged over the bundled file (tables merge, scalars and lists
replace).

**Module:** `src/internal/configs/timeouts.py`:

- A frozen dataclass per section (§2) and a root `TimeoutPolicies`.
- `load_timeout_policies(env: Mapping[str, str] | None = None) -> TimeoutPolicies` —
  pure: bundled → override file → env vars.
- `get_timeout_policies() -> TimeoutPolicies` — process-wide cached
  `load_timeout_policies(os.environ)`.
- `use_timeout_policies(policies)` — context manager replacing the cached value
  (tests); `reset_timeout_policies()` clears the cache.
- `AppSettings` gains `timeouts: TimeoutPolicies`, filled by `load_app_settings`
  from the same loader.

**Validation (fail fast, at load):**

- Unknown table or key → `ValueError` naming its dotted path (`llm.socket_read_timout_seconds`).
- `*_seconds` > 0; `max_retries` / `grounded_max_retries` integers ≥ 0;
  `max_attempts` / `max_escalations` integers ≥ 1; `backoff_seconds` a non-empty
  list of numbers ≥ 0. A bool is not a number.
- `AGENTIC_SEARCH_TIMEOUTS_PATH` set to a missing or unparsable file → `ValueError`
  naming the path. Unset or empty → bundled only.

**Precedence:** explicit constructor/call kwarg > existing env var > operator file >
bundled file. The four env vars keep their names and meaning:

| env var | key |
|---|---|
| `TOOL_APPROVAL_TIMEOUT_SECONDS` | `tool_loop.approval_timeout_seconds` |
| `AGENTIC_SEARCH_GENERATION_TIMEOUT` | `llm.local_generation_timeout_seconds` |
| `LLM_SOCKET_READ_TIMEOUT` | `llm.socket_read_timeout_seconds` |
| `AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS` | `sse.heartbeat_seconds` |

**Resolution time:** sites read the policies at construct or call time, never at
import time. A kwarg or dataclass default that is a literal today becomes `None`
(or a `default_factory`) resolved from `get_timeout_policies()`; module constants
become reads inside the function that uses them.

**Dependency:** `tomli; python_version < "3.11"` in `requirements.txt` and
`pyproject.toml` dependencies; `tomllib` on 3.11+.

## 2. Schema

Values are today's. "Replaces" lists every copy the key becomes the source for.

| table | keys | replaces |
|---|---|---|
| `tools.public_data` | `timeout_seconds = 10`, `max_attempts = 3` (GET only), `backoff_seconds = [0.4, 0.8]`, `retry_budget_seconds = 15`, `overpass_timeout_seconds = 30`, `overpass_query_timeout_seconds = 25` | `public_data/_http.py` `DEFAULT_TIMEOUT_SECONDS`, `_MAX_ATTEMPTS`, `_RETRY_BACKOFF_SECONDS`, `_RETRY_BUDGET_SECONDS`; `geo.py` `OVERPASS_TIMEOUT_SECONDS` and the `[timeout:25]` in the Overpass query |
| `tools.web_search` | `google_timeout_seconds = 15`, `serpapi_timeout_seconds = 15`, `serper_timeout_seconds = 10`, `fetch_page_timeout_seconds = 15`, `fetch_pages_timeout_seconds = 10`, `query_timeout_seconds = 15` | `search.py` `google_custom_search`, `serpapi_search`, the cascade SerpAPI leg, `serper_dev_search`, `fetch_url`, `fetch_pages_concurrently`, `MultiQueryWebSearchTool` |
| `tools.search_router` | `timeout_seconds = 15`, `max_retries = 3` | `search.py` `search_tool` defaults and the cascade browser leg; `routing_tools._SEARCH_ATTEMPTS` is derived from `max_retries` |
| `tools.openapi` | `timeout_seconds = 15` | `tools/api.py` `ClientTimeout(total=15)` |
| `tools.mcp` | `timeout_seconds = 30`, `sse_read_timeout_seconds = 300` | `mcp_client.py` `streamablehttp_client(...)` library defaults, now passed explicitly |
| `retrieval.client` | `timeout_seconds = 10`, `max_retries = 3`, `backoff_base_seconds = 0.5` | `SearchClientConfig`, the `0.5 * 2**attempt` backoff, `search.py` `retrieval_search`, `agents/search/search.py` and `agents/generation/single_turn.py` defaults |
| `retrieval.search_runner` | `timeout_seconds = 15`, `max_retries = 3` | the three copies in `context/retrieval/search_runner.py` |
| `retrieval.web_hybrid` | `provider_timeout_seconds = 5`, `provider_max_retries = 1`, `provider_wait_seconds = 8` | `servers/web/app.py` per-provider client and `wait_for` |
| `rerank` | `timeout_seconds = 10` | `search/stages.py` `RerankHTTPRankingStage` |
| `llm` | `socket_read_timeout_seconds = 120`, `remote_total_timeout_seconds = 120`, `local_generation_timeout_seconds = 120`, `local_heartbeat_seconds = 10`, `sufficiency_timeout_seconds = 5`, `grounded_max_retries = 1` | `multi_llm.py`; `serving.py` `OpenAIServerManager` (now forwarded by `build_server_manager`) and `LocalServerManager`; `AppSettings.generation_timeout_seconds`; `agentic_rag.py` `sufficiency_timeout_s`; `GroundedGenerationConfig.max_retries` and the `pipeline.py` clamp |
| `tool_loop` | `approval_timeout_seconds = 60`, `escalation_timeout_seconds = 120`, `max_escalations = 3`, `tool_evidence_timeout_seconds = 5` | `AppSettings.tool_approval_timeout_seconds`, `ToolAgentLoopConfig`, the `tool_agent_runner` fallback, `ToolApprovalBroker`, `ToolEscalationBroker`; `tool_evidence.py` and `pipeline.py` defaults |
| `tool_loop.recovery` | `max_retries = 2`, `backoff_seconds = [0.5, 1.0]`, `retry_after_cap_seconds = 4`, `retry_budget_seconds = 10` | `RecoveryPolicy()` defaults |
| `sse` | `heartbeat_seconds = 15` | `servers/sse.py` |

Existing `AppSettings` fields (`tool_approval_timeout_seconds`,
`generation_timeout_seconds`) stay as fields; `load_app_settings` fills them from
`timeouts.tool_loop.approval_timeout_seconds` and
`timeouts.llm.local_generation_timeout_seconds`, so current callers and tests that
construct `AppSettings` directly keep working. `grounded_max_retries` replaces the
hard-coded clamp value in `pipeline.py` (the clamp reads the key).

**Deleted as dead duplicates:** `chat_configs.LLM_SOCKET_READ_TIMEOUT` (60, unused);
the 60 s documented in `default_config.py` (becomes a pointer to the file); the 26
`AGENT_TIMEOUT_*` constants in `agent_configs.py`; the `getattr(..., 60.0)` fallback
in `tool_agent_runner.py`. Each deletion is preceded by an AST-level reachability
check (no importer of the name anywhere in `src`, `examples`, `tests`).

## 3. Testing

- **Snapshot:** `load_timeout_policies({})` equals today's values key by key; the
  expected values are literals in the test, not read from the file.
- **Loader:** partial override deep-merges; unknown key errors with its dotted path;
  wrong type / out-of-range / bool errors; missing and unparsable override files
  error; each of the four env vars beats the file; `use_timeout_policies` and
  `reset_timeout_policies` isolate tests.
- **Per site:** under `use_timeout_policies(...)` with a changed value, the component
  uses it (e.g. `_http` makes 2 attempts with `max_attempts = 2`; `RecoveryPolicy()`
  takes the new budget; `ToolEscalationBroker` the new timeout); an explicit kwarg
  still wins. Every per-site test is mutation-checked by restoring the literal.
- **Drift guard:** an AST scan of the migrated modules fails if a schema-owned value
  reappears as a literal default or module constant.
- **Packaging:** the bundled file resolves through `importlib.resources`; built
  wheel contains `timeouts.toml`.
- **3.10 path:** the loader is exercised with `tomllib` blocked so the `tomli`
  fallback runs; the unit suite passes with torch unimportable.

## 4. Docs

`docs/configuration/timeouts.md`: each key, what it bounds, and its env override.
One line under Configuration in `.claude/CLAUDE.md`; `.env.example` gains a
commented `AGENTIC_SEARCH_TIMEOUTS_PATH`.
