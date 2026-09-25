# Model unavailable: degrade to search-only instead of 502: design

## Problem

When the model is unavailable, `/api/agent` usually fails the whole request.
The investigation of 2026-09-25 traced these outcomes:

| Failure | Where it surfaces | Outcome today |
|---|---|---|
| Remote LLM provider connection error, HTTP 5xx or 429 | `llm.complete` / `stream_complete` (`src/internal/llm/providers.py`), final answer synthesis on auto CHAT, `chat_loop`, `chat_once` | Re-raised (`requests.ConnectionError` / `requests.HTTPError`), which becomes **HTTP 502** at the dispatch `except Exception` in `src/internal/servers/web/app.py` (~:2113-2122) |
| Remote inference server down, or its circuit breaker open | `OpenAIServerManager.generate` / `generate_stream` (`src/model/serving.py`), on the auto SEARCH escalation and the explicit `search_agent` / `tool_agent` modes | Plain `RuntimeError`, which becomes **HTTP 502** |
| Same, auto TOOL route | `app.py` ~:1343-1371 | Already caught and degraded (`route_degraded="tool_unavailable"`, then CHAT) |
| `LLMTimeoutError` during synthesis | `generate_answer` (`src/context/pipeline.py` ~:322) | Already degraded (`TIMEOUT_DEGRADED_ANSWER`) |

The search-only degraded answer already exists: `_auto_search_pipeline`
(`app.py` ~:656), used today for `no_llm` / `no_local_model`. It is not used
when a configured model fails at runtime.

## Decision (approved by the user)

### 1. One typed error for "the model is unavailable"

`src/context/models.py`:

```python
class ModelUnavailableError(RuntimeError):
    """The model could not be reached or refused service (connect error,
    timeout, 5xx/429, circuit open). Not raised for request/config errors."""


class LLMTimeoutError(ModelUnavailableError):  # was RuntimeError
    """The LLM call exceeded its timeout."""
```

- `LLMTimeoutError` becomes a subclass, so every existing `except
  LLMTimeoutError` handler keeps working unchanged. Where a handler catches
  both, it must list `LLMTimeoutError` first.
- Export `ModelUnavailableError` wherever `LLMTimeoutError` is exported.

### 2. Who raises it

- **Remote LLM providers** (`src/internal/llm/providers.py`, both `complete`
  and the streaming path):
  - `requests.ConnectionError` → `ModelUnavailableError` (`from exc`).
  - `requests.HTTPError` with status ≥ 500 or 429 → `ModelUnavailableError`.
  - Any other `requests.HTTPError` (4xx, including a context-too-long 400) is
    a request or config error and keeps raising unchanged.
  - The `SchemaUnsupportedError` mapping is unchanged and is checked first.
- **`OpenAIServerManager`** (`src/model/serving.py`): the connect-error,
  timeout and circuit-open paths now raise `ModelUnavailableError` instead of
  a bare `RuntimeError`, with the same messages. It is still a `RuntimeError`,
  so the existing auto TOOL catch and the circuit-breaker tests keep passing.
  Breaker success and failure accounting is unchanged.
- **Error text:** messages are the existing ones, with no URLs or secrets
  beyond what is logged today.

### 3. One handler: `/api/agent` dispatch

The existing `except Exception` → 502 block in `_run_agent_impl` (the one
shared by `POST /api/agent`, `/api/agent/stream` and `/api/agent/ws`) gains an
arm before it:

```python
except ModelUnavailableError as exc:
    logger.warning("Model unavailable, degrading to search-only: %s", exc)
    # run _auto_search_pipeline for this query with the request's search
    # settings, and mark extra["route_degraded"] = "model_unavailable"
```

- **The degraded response:** the normal success shape (HTTP 200) with the
  search-only answer, its documents, and `route_degraded="model_unavailable"`.
  The UI already renders `route_degraded` as the ⚠ degraded pill.
- **Where the degradation starts:** the handler re-uses `_auto_search_pipeline`
  exactly as the `no_llm` path does, so ACL and filter enforcement come from
  the same code. It must never skip `_enforce_access`.
- **If the fallback itself raises**, that exception goes through the existing
  502 path. There is no fallback chain.
- **Coverage:** every mode that reaches this dispatch. That is auto (SEARCH
  escalation and CHAT synthesis), `search_agent`, `tool_agent`, `chat_loop`
  and `chat_once`. The auto TOOL route keeps its own existing degradation.
- **Existing precedence is kept:** an earlier `route_degraded` value, for
  example `tool_unavailable`, is overwritten only when this handler actually
  runs the fallback. The final value reports the degradation that produced the
  answer.

## Out of scope

- The direct surfaces `/chat/send-chat-message` and `/tool/send-tool-message`.
  Today they return `answer=""` plus `error`, and that stays for now.
- Extractive answers instead of search-only answers.
- Context-too-long handling, which belongs to the summarization work.
- Routing-LLM blocking on the event loop.
- Readiness checks for model state.

## Testing

- **Unit tests for the new error type:**
  - `LLMTimeoutError` is a `ModelUnavailableError`.
  - Provider mapping: `ConnectionError`, 500, 503 and 429 raise
    `ModelUnavailableError`.
  - A 400 re-raises `requests.HTTPError`.
  - A schema-unsupported 400 still raises `SchemaUnsupportedError`.
  - Streaming and non-streaming are covered alike.
  - `OpenAIServerManager`: connect error, timeout and circuit open each raise
    `ModelUnavailableError`, and each is still a `RuntimeError`.
- **App tests** (`create_web_app` + `TestClient`, using the stubbing patterns
  of existing web tests):
  - For each mode (auto CHAT, `chat_once`, `chat_loop`, `search_agent`), a
    model that raises `ModelUnavailableError` gives 200, a non-empty answer
    and `route_degraded == "model_unavailable"`.
  - A model raising a plain `ValueError`, or a 4xx `HTTPError`, still gives
    502.
  - Documents in the degraded response pass the same ACL filtering. Reuse an
    existing ACL fixture if one is available.
- **Regression:** the existing timeout-degraded tests and the auto TOOL
  degradation tests are unchanged and must pass.
- **Mutation checks:**
  - Remove the new `except` arm and watch the app tests go red.
  - Map 4xx to unavailable and watch the 4xx test go red.
