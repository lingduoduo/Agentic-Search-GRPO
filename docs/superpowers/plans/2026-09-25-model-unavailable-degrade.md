# Model Unavailable Degrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the configured model is unreachable or refuses service at runtime, `/api/agent` answers with the search-only degraded answer (HTTP 200, `route_degraded="model_unavailable"`) instead of HTTP 502.

**Architecture:** One typed error, `ModelUnavailableError(RuntimeError)`, raised by the two model transports (`OpenAICompatibleLLM` over `requests`, `OpenAIServerManager` over `aiohttp`) for connect errors, 5xx/429 and an open circuit. `LLMTimeoutError` becomes its subclass. One new `except ModelUnavailableError` arm in `_run_agent_impl` re-runs the query through the existing `_auto_search_pipeline` (the `no_llm` fallback) and finalizes it like any other turn.

**Tech Stack:** Python 3.12, FastAPI + TestClient, requests, aiohttp, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-model-unavailable-degrade-design.md`

## Global Constraints

- `class ModelUnavailableError(RuntimeError)` in `src/context/models.py`; `class LLMTimeoutError(ModelUnavailableError)` (was `RuntimeError`).
- Export `ModelUnavailableError` wherever `LLMTimeoutError` is exported (`src/context/__init__.py` import + `__all__`).
- Providers: `requests.ConnectionError` → `ModelUnavailableError` (`from exc`); `requests.HTTPError` with status ≥ 500 or 429 → `ModelUnavailableError`; any other `HTTPError` (4xx, incl. context-too-long 400) re-raises unchanged; the `SchemaUnsupportedError` mapping is unchanged and is checked first.
- `OpenAIServerManager`: connect-error, timeout and circuit-open paths raise `ModelUnavailableError` with the **same messages**; breaker accounting unchanged.
- Error text: existing messages only, no new URLs or secrets.
- `/api/agent` degraded response: HTTP 200, normal success shape, `hook_metadata["route_degraded"] == "model_unavailable"`.
- The fallback is `_auto_search_pipeline`, the same code the `no_llm` path uses; never skip `_enforce_access`.
- If the fallback raises, it goes through the existing 502 path. No fallback chain.
- The auto TOOL route keeps its own degradation; existing timeout-degraded tests are unchanged.
- Out of scope: `/chat/send-chat-message`, `/tool/send-tool-message`, extractive answers, context-too-long, routing-LLM blocking, readiness.

## Deviations from the spec (recorded, minimal)

1. **Existing contract tests change.** Four existing tests pin the old behaviour the spec deliberately reverses and are updated, not deleted:
   `tests/unit/test_llm_providers.py::test_plain_connection_error_still_propagates` (ConnectionError now becomes `ModelUnavailableError`),
   `tests/unit/test_llm_structured_output.py::test_connection_error_propagates_unchanged` (same),
   `tests/unit/test_llm_structured_output.py::test_other_http_errors_propagate_unchanged` (its 429 and 500 cases now become `ModelUnavailableError`; the 400 case stays), and `tests/unit/test_llm_structured_output.py::test_stream_complete_other_http_errors_propagate_unchanged` (found during execution: it used a 500 as its "other" error; now a non-schema 400, which keeps its intent — 5xx on `stream_complete` is covered by the new tests). The circuit-breaker tests that `pytest.raises(RuntimeError, ...)` keep passing unchanged, as the spec says.
2. **The fallback runs with `llm=None`.** The spec says to reuse `_auto_search_pipeline` "exactly as the `no_llm` path does"; that path has `llm is None`. Passing the request's `llm` would make `_expanded_queries` call the dead model again for query expansion (up to another 30 s timeout per request before degrading). So the arm passes `llm=None`.
3. **Streaming mid-body errors in the provider are not mapped.** The spec maps `ConnectionError` "in both `complete` and the streaming path". The mapping is placed around the request phase of `stream()` (connect + status), where no text has been yielded yet. A `ConnectionError` raised while iterating an already-open body keeps propagating (it would arrive after tokens were streamed to the client; see Review Focus 5).
4. **Source provider for the fallback.** Explicit modes never validated `request.source_provider`; the arm normalizes it with `_normalize_source_provider` exactly as the auto path does (default `"auto"`).
5. **5xx from the inference server (`OpenAIServerManager`) is not mapped.** The spec lists only connect-error, timeout and circuit-open for the manager; a `ClientResponseError` 5xx/429 keeps re-raising `aiohttp.ClientResponseError` (the breaker still counts it, and once open the next call raises `ModelUnavailableError`). Kept as specced.

## Review Focus

1. **The fallback itself fails** (retrieval server also down → `_auto_search_pipeline` raises): expect 502 with the fallback's message, not a hang or a 200 with an empty answer. Pinned in Task 4 (`test_fallback_failure_still_returns_502`).
2. **The fallback must not call the dead model again** (query expansion through `llm`): expect `llm=None` passed to `_auto_search_pipeline`. Pinned in Task 4 (`test_degrade_passes_request_scope_and_no_llm`).
3. **ACL on the degraded documents**: a private document returned by the retrieval server must not reach an anonymous caller through the fallback. Pinned in Task 4 (`test_degraded_documents_are_acl_filtered`) with the real pipeline and a stubbed `search_tool`.
4. **Streaming transport** (`/api/agent/stream`): the `done` event must carry `route_degraded == "model_unavailable"` and the degraded answer, not an `error` event. Pinned in Task 4 (`test_stream_done_event_reports_model_unavailable`).
5. **Degraded turn persistence**: the degraded answer is stored as the assistant turn so a follow-up in the same session sees it. Pinned in Task 4 (asserted in `test_degrades_per_mode` via `messages[-1]`).

---

### Task 1: `ModelUnavailableError` type and exports

**Files:**
- Modify: `src/context/models.py:41-42`
- Modify: `src/context/__init__.py` (import next to `LLMTimeoutError` ~:21, `__all__` next to `"LLMTimeoutError"` ~:78)
- Test: `tests/unit/test_model_unavailable_error.py` (new)

**Interfaces:**
- Produces: `src.context.models.ModelUnavailableError(RuntimeError)`, re-exported as `src.context.ModelUnavailableError`; `LLMTimeoutError` now subclasses it.

- [x] **Step 1: Write the failing test**

```python
"""ModelUnavailableError: the one typed "model is unavailable" error."""

from __future__ import annotations


def test_llm_timeout_is_a_model_unavailable_error():
    from src.context.models import LLMTimeoutError, ModelUnavailableError

    assert issubclass(ModelUnavailableError, RuntimeError)
    assert issubclass(LLMTimeoutError, ModelUnavailableError)


def test_model_unavailable_error_is_exported_with_llm_timeout_error():
    import src.context as context

    assert context.ModelUnavailableError is context.models.ModelUnavailableError
    assert "ModelUnavailableError" in context.__all__
```

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_model_unavailable_error.py -q -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'ModelUnavailableError'`.

- [x] **Step 3: Implement**

`src/context/models.py`, replacing the `LLMTimeoutError` class:

```python
class ModelUnavailableError(RuntimeError):
    """The model could not be reached or refused service (connect error,
    timeout, 5xx/429, circuit open). Not raised for request/config errors."""


class LLMTimeoutError(ModelUnavailableError):
    """The LLM call exceeded its timeout."""
```

`src/context/__init__.py`: add `from .models import ModelUnavailableError` after `from .models import LLMTimeoutError`, and `"ModelUnavailableError",` after `"LLMTimeoutError",` in `__all__`.

- [x] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_model_unavailable_error.py tests/unit/test_context_pipeline.py -q -p no:cacheprovider`
Expected: PASS (existing `except LLMTimeoutError` handlers unaffected).

- [x] **Step 5: Commit**

```bash
git add src/context/models.py src/context/__init__.py tests/unit/test_model_unavailable_error.py
git commit -m "ModelUnavailableError: one typed error for an unavailable model"
```

---

### Task 2: Remote LLM provider maps unavailability

**Files:**
- Modify: `src/internal/llm/providers.py` (import ~:21; helper next to `_is_schema_unsupported_response` ~:112; `stream` request phase ~:201-222; `complete` ~:360-376)
- Modify: `tests/unit/test_llm_providers.py` (`test_plain_connection_error_still_propagates`)
- Modify: `tests/unit/test_llm_structured_output.py` (`test_other_http_errors_propagate_unchanged`, `test_connection_error_propagates_unchanged`)
- Test: `tests/unit/test_model_unavailable_error.py`

**Interfaces:**
- Consumes: `ModelUnavailableError` (Task 1).
- Produces: `OpenAICompatibleLLM.complete` / `stream` / `stream_complete` raise `ModelUnavailableError` for connect errors and 5xx/429; 4xx re-raise `requests.HTTPError`; schema-unsupported 400 raises `SchemaUnsupportedError` when a schema was applied.

- [x] **Step 1: Write the failing tests** (append to `tests/unit/test_model_unavailable_error.py`)

```python
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.context.models import ModelUnavailableError
from src.context.structured_output import (
    SchemaUnsupportedError,
    StructuredOutputRequest,
)
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM

MESSAGES = [{"role": "user", "content": "hi"}]


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        LLMConfig(model_provider="openai", model_name="m", api_key="k")
    )


def _http_error(status: int, body: str = "provider says no") -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    response.url = "https://provider.invalid/chat/completions"
    return requests.HTTPError(f"{status} {body}", response=response)


def _failing_response(error: requests.HTTPError) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.side_effect = error
    return response


def _complete(llm, **kwargs):
    return llm.complete(MESSAGES, **kwargs)


def _stream_complete(llm, **kwargs):
    return list(llm.stream_complete(MESSAGES, **kwargs))


def _stream(llm, **kwargs):
    return list(llm.stream(prompt="hi"))


PATHS = [_complete, _stream_complete, _stream]


@pytest.mark.parametrize("call", PATHS)
def test_connection_error_is_model_unavailable(call):
    llm = _llm()
    cause = requests.ConnectionError("connection refused")
    with patch.object(llm._session, "post", side_effect=cause):
        with pytest.raises(ModelUnavailableError) as caught:
            call(llm)
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("call", PATHS)
@pytest.mark.parametrize("status", [500, 503, 429])
def test_5xx_and_429_are_model_unavailable(call, status):
    llm = _llm()
    error = _http_error(status)
    with patch.object(llm._session, "post", return_value=_failing_response(error)):
        with pytest.raises(ModelUnavailableError) as caught:
            call(llm)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("call", PATHS)
def test_4xx_reraises_http_error(call):
    llm = _llm()
    error = _http_error(400, "context length exceeded")
    with patch.object(llm._session, "post", return_value=_failing_response(error)):
        with pytest.raises(requests.HTTPError) as caught:
            call(llm)
    assert caught.value is error
    assert not isinstance(caught.value, ModelUnavailableError)


@pytest.mark.parametrize("call", [_complete, _stream_complete])
def test_schema_unsupported_400_still_raises_schema_unsupported(call):
    llm = _llm()
    request = StructuredOutputRequest(
        name="answer",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    error = _http_error(400, "unknown parameter: response_format")
    with patch.object(llm._session, "post", return_value=_failing_response(error)):
        with pytest.raises(SchemaUnsupportedError):
            call(llm, structured_output=request)
```

(`StructuredOutputRequest` field names: confirm against `tests/unit/test_llm_structured_output.py::schema_request` before running and match it.)

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_model_unavailable_error.py -q -p no:cacheprovider`
Expected: connection/5xx/429 cases FAIL (`requests.ConnectionError` / `HTTPError` raised, not `ModelUnavailableError`); 4xx and schema cases PASS already.

- [x] **Step 3: Implement** in `src/internal/llm/providers.py`

Import: `from src.context.models import LLMResponse, LLMTimeoutError, ModelUnavailableError`.

Helper below `_is_schema_unsupported_response`:

```python
def _is_unavailable_status(response: requests.Response | None) -> bool:
    """5xx or 429: the provider refused service, not the request."""
    return response is not None and (
        response.status_code >= 500 or response.status_code == 429
    )
```

`stream`, request phase (the `try` around `post` + `raise_for_status`):

```python
        except requests.Timeout:
            raise LLMTimeoutError("LLM request timed out") from None
        except requests.ConnectionError as exc:
            raise ModelUnavailableError("LLM provider is unreachable") from exc
        except requests.HTTPError as exc:
            logger.error(...)  # unchanged
            if _is_unavailable_status(exc.response):
                raise ModelUnavailableError(
                    "LLM provider is unavailable"
                ) from exc
            raise
```

`complete`:

```python
        except requests.Timeout:
            raise LLMTimeoutError("LLM request timed out") from None
        except requests.ConnectionError as exc:
            raise ModelUnavailableError("LLM provider is unreachable") from exc
        except requests.HTTPError as exc:
            if schema_applied and _is_schema_unsupported_response(exc.response):
                raise SchemaUnsupportedError(
                    "Provider does not support JSON Schema response formatting"
                ) from None
            if _is_unavailable_status(exc.response):
                raise ModelUnavailableError("LLM provider is unavailable") from exc
            raise
```

`requests.Timeout` stays first: `ConnectTimeout` subclasses both `Timeout` and `ConnectionError` and must remain `LLMTimeoutError`. `stream_complete` needs no change: the schema check only fires on 400, and 5xx/429 now arrive from `stream` as `ModelUnavailableError`, which its `except requests.HTTPError` does not catch. The messages are fixed strings (no endpoint, no provider body); the original exception is the `__cause__`, which is what gets logged today.

Update the three pinned tests (Deviation 1):

`tests/unit/test_llm_providers.py` — rename and flip:

```python
def test_plain_connection_error_is_model_unavailable():
    # A host that is down or refusing means the model is unavailable: the web
    # dispatch degrades to search-only on this type instead of a 502.
    llm = _timeout_llm()
    with patch.object(
        llm._session,
        "post",
        side_effect=requests.ConnectionError("connection refused"),
    ):
        with pytest.raises(ModelUnavailableError):
            llm.complete([{"role": "user", "content": "hi"}])
```

(import `ModelUnavailableError` next to `LLMTimeoutError`.)

`tests/unit/test_llm_structured_output.py` — drop the 429/500 cases from `test_other_http_errors_propagate_unchanged` (keep `(400, "invalid API key")`) and add:

```python
@pytest.mark.parametrize(
    ("status", "body"),
    [(429, "response_format is unsupported"), (500, "json_schema failed")],
)
def test_unavailable_statuses_become_model_unavailable(schema_request, status, body):
    llm = configured_llm()
    error = http_error(status, body)
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(ModelUnavailableError) as caught:
            llm.complete(MESSAGES, structured_output=schema_request)
    assert caught.value.__cause__ is error
```

and turn `test_connection_error_propagates_unchanged` into:

```python
def test_connection_error_becomes_model_unavailable(schema_request):
    llm = configured_llm()
    error = requests.ConnectionError("down")
    with patch.object(llm._session, "post", side_effect=error):
        with pytest.raises(ModelUnavailableError) as caught:
            llm.complete(MESSAGES, structured_output=schema_request)
    assert caught.value.__cause__ is error
```

- [x] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_model_unavailable_error.py tests/unit/test_llm_providers.py tests/unit/test_llm_structured_output.py -q -p no:cacheprovider`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/internal/llm/providers.py tests/unit/test_model_unavailable_error.py tests/unit/test_llm_providers.py tests/unit/test_llm_structured_output.py
git commit -m "LLM provider: connect errors and 5xx/429 raise ModelUnavailableError"
```

---

### Task 3: `OpenAIServerManager` raises `ModelUnavailableError`

**Files:**
- Modify: `src/model/serving.py` (import; `_admit` ~:314-324; `_connect_error` ~:326-330)
- Test: `tests/unit/resilience/test_circuit_breaker_sites.py` (append; reuses `_llm`, `_call`, `threshold`)

**Interfaces:**
- Consumes: `ModelUnavailableError` (Task 1).
- Produces: `OpenAIServerManager.generate` / `generate_stream` raise `ModelUnavailableError` (a `RuntimeError`) on connect error, timeout and open circuit, same messages.

- [x] **Step 1: Write the failing test** (append to `tests/unit/resilience/test_circuit_breaker_sites.py`)

```python
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "outcome",
    [
        asyncio.TimeoutError(),
        aiohttp.ClientConnectorError(
            connection_key=None, os_error=OSError(61, "refused")
        ),
    ],
    ids=["timeout", "connect"],
)
def test_remote_llm_unavailable_paths_raise_model_unavailable(
    monkeypatch, stream, outcome
):
    from src.context.models import ModelUnavailableError

    m, _ = _llm(monkeypatch, outcome)
    with threshold(1):
        with pytest.raises(ModelUnavailableError, match="Cannot connect") as first:
            _call(m, stream)
        with pytest.raises(ModelUnavailableError, match="circuit open") as second:
            _call(m, stream)
    assert isinstance(first.value, RuntimeError)
    assert isinstance(second.value, RuntimeError)
```

(`ClientConnectorError(connection_key, os_error)` — if the installed aiohttp's `__str__` needs a real `ConnectionKey`, build it with `MagicMock()` for `connection_key`.)

- [x] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/resilience/test_circuit_breaker_sites.py -q -p no:cacheprovider -k model_unavailable`
Expected: FAIL — plain `RuntimeError` is not `ModelUnavailableError`.

- [x] **Step 3: Implement** in `src/model/serving.py`

Add `from src.context.models import ModelUnavailableError` to the imports, then:

```python
    def _admit(self):
        """This server's breaker, or ModelUnavailableError while it is being skipped."""
        breaker = get_breaker(f"remote_llm:{self.base_url}")
        try:
            breaker.before_call()
        except CircuitOpenError:
            raise ModelUnavailableError(
                f"Inference server at {self.base_url} is temporarily skipped "
                "after repeated failures (circuit open)."
            ) from None
        return breaker

    def _connect_error(self) -> ModelUnavailableError:
        return ModelUnavailableError(
            f"Cannot connect to inference server at {self.base_url}. "
            f"Start one first, e.g.: mlx_lm.server --model {self.model} --port 8080"
        )
```

No change to `generate` / `generate_stream` bodies or breaker accounting.

- [x] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/resilience/test_circuit_breaker_sites.py tests/unit/test_model_serving.py tests/unit/test_timeout_policy_sites.py -q -p no:cacheprovider`
Expected: PASS (existing `pytest.raises(RuntimeError, ...)` tests still pass). Also run `.venv/bin/python -c "import src.model.serving"` to confirm no import cycle.

- [x] **Step 5: Commit**

```bash
git add src/model/serving.py tests/unit/resilience/test_circuit_breaker_sites.py
git commit -m "OpenAIServerManager: connect, timeout and circuit-open raise ModelUnavailableError"
```

---

### Task 4: `/api/agent` degrades to search-only on `ModelUnavailableError`

**Files:**
- Modify: `src/internal/servers/web/app.py` (import `ModelUnavailableError` ~:54; new module helper `_dispatch_failure` near `_enforce_access` ~:987; dispatch `except` block ~:2113-2122)
- Test: `tests/unit/servers/web/test_model_unavailable_degrade.py` (new)

**Interfaces:**
- Consumes: `ModelUnavailableError` (Task 1); `_auto_search_pipeline(query, *, llm, search_url, browser_search_url, rerank_url, top_k, filters, history, source_provider, extra, domain="general", retrieval_query=None) -> (answer, citations, documents, intent, extra)`; `_finalize_response(db, session_id, *, query, answer, citations, documents, intent, hook_metadata, extra, mode)`.
- Produces: `_dispatch_failure(exc: Exception) -> HTTPException` (502, detail = `str(exc)` or the generic message) used by both the existing arm and the fallback failure.

- [x] **Step 1: Write the failing tests** (`tests/unit/servers/web/test_model_unavailable_degrade.py`)

```python
"""/api/agent degrades to search-only when the model is unavailable."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests
from fastapi.testclient import TestClient

from src.context.models import ContextDocument, ModelUnavailableError
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.servers.web.intent import RouteDecision, RouteStrategy

APP = "src.internal.servers.web.app"


async def _unavailable(*args, **kwargs):
    raise ModelUnavailableError("Cannot connect to inference server")


def _fake_pipeline(calls: list):
    async def fake(query, **kwargs):
        calls.append({"query": query, **kwargs})
        doc = ContextDocument(id="D1", title="Doc", content="body", score=0.5)
        return "search-only answer", ["D1"], [doc], "search", kwargs["extra"]

    return fake


def _post(app, body: dict, *, local_model: bool = False):
    with TestClient(app) as client:
        if local_model:
            app.state.search_agent_manager = MagicMock()
            app.state.search_agent_tokenizer = MagicMock()
        return client.post("/api/agent", json=body)


def _app(tmp_path):
    return create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )


# (mode, runner patched to raise, needs a local model)
MODES = [
    (None, "_run_agentic_rag", False),  # auto, CHAT route
    ("chat_once", "answer_with_retrieval", False),
    ("chat_loop", "_run_agentic_rag", False),
    ("search_agent", "_run_search_agent", True),
    ("tool_agent", "_run_tool_agent", True),
]


@pytest.mark.parametrize(("mode", "runner", "local_model"), MODES)
def test_degrades_per_mode(monkeypatch, tmp_path, mode, runner, local_model):
    monkeypatch.setattr(
        f"{APP}.recognize_intent", lambda *a, **k: RouteDecision(RouteStrategy.CHAT)
    )
    monkeypatch.setattr(f"{APP}.{runner}", _unavailable)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))
    body = {"query": "explain FAISS"}
    if mode:
        body["mode"] = mode

    response = _post(_app(tmp_path), body, local_model=local_model)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["answer"] == "search-only answer"
    assert data["hook_metadata"]["route_degraded"] == "model_unavailable"
    assert [d["id"] for d in data["documents"]] == ["D1"]
    assert data["messages"][-1]["content"] == "search-only answer"
    assert len(calls) == 1


def test_degrade_passes_request_scope_and_no_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(f"{APP}.answer_with_retrieval", _unavailable)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))

    response = _post(
        _app(tmp_path), {"query": "explain FAISS", "mode": "chat_once", "top_k": 3}
    )

    assert response.status_code == 200
    (call,) = calls
    assert call["query"] == "explain FAISS"
    assert call["llm"] is None
    assert call["top_k"] == 3
    assert call["filters"].access_acl  # anonymous is ["public"], never unfiltered
    assert call["source_provider"] == "auto"


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad input format"),
        requests.HTTPError(
            "400 context length exceeded",
            response=MagicMock(status_code=400),
        ),
    ],
    ids=["value_error", "http_4xx"],
)
def test_non_availability_errors_still_502(monkeypatch, tmp_path, error):
    async def explode(*args, **kwargs):
        raise error

    monkeypatch.setattr(f"{APP}.answer_with_retrieval", explode)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))

    response = _post(_app(tmp_path), {"query": "q", "mode": "chat_once"})

    assert response.status_code == 502
    assert str(error) in response.json()["detail"]
    assert calls == []


def test_fallback_failure_still_returns_502(monkeypatch, tmp_path):
    monkeypatch.setattr(f"{APP}.answer_with_retrieval", _unavailable)

    async def pipeline_down(query, **kwargs):
        raise RuntimeError("retrieval server unreachable")

    monkeypatch.setattr(f"{APP}._auto_search_pipeline", pipeline_down)

    response = _post(_app(tmp_path), {"query": "q", "mode": "chat_once"})

    assert response.status_code == 502
    assert "retrieval server unreachable" in response.json()["detail"]


def test_degraded_documents_are_acl_filtered(monkeypatch, tmp_path):
    """Real fallback pipeline: a private document never reaches an anonymous caller."""
    from src.internal.tools import SearchPage

    monkeypatch.setattr(f"{APP}.answer_with_retrieval", _unavailable)

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider != "retrieval":
            return []
        return [
            SearchPage(title="Public", summary="ok", url="http://r/pub",
                       metadata={"acl": ["public"]}),
            SearchPage(title="Private", summary="secret", url="http://r/priv",
                       metadata={"acl": ["user:alice"]}),
        ]

    async def no_browser(*args, **kwargs):
        return []

    monkeypatch.setattr(f"{APP}.search_tool", fake_search_tool)
    monkeypatch.setattr(f"{APP}._run_browser_search", no_browser)

    response = _post(_app(tmp_path), {"query": "q", "mode": "chat_once"})

    assert response.status_code == 200, response.text
    data = response.json()
    titles = [d["title"] for d in data["documents"]]
    assert "Public" in titles
    assert "Private" not in titles
    assert "secret" not in data["answer"]
    assert data["hook_metadata"]["route_degraded"] == "model_unavailable"


def test_stream_done_event_reports_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        f"{APP}.recognize_intent", lambda *a, **k: RouteDecision(RouteStrategy.CHAT)
    )
    monkeypatch.setattr(f"{APP}._run_agentic_rag", _unavailable)
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline([]))

    with TestClient(_app(tmp_path)) as client:
        response = client.post("/api/agent/stream", json={"query": "explain FAISS"})

    assert response.status_code == 200
    events = [
        json.loads(line[len("data:"):].strip())
        for line in response.text.splitlines()
        if line.startswith("data:") and line[len("data:"):].strip()
    ]
    assert not [e for e in events if e.get("type") == "error"]
    done = next(e for e in events if e["type"] == "done")
    assert done["route_degraded"] == "model_unavailable"
    answer = next(e for e in events if e["type"] == "answer")
    assert answer["text"] == "search-only answer"
```

(The answer text rides the `answer` event; `done` carries `route_degraded` — see `_terminal_events` ~:2476. If the ACL test's documents pass through a different provider set for `"auto"`, keep the assertion — only the stubs change.)

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/servers/web/test_model_unavailable_degrade.py -q -p no:cacheprovider`
Expected: degrade/ACL/stream tests FAIL with status 502 (or an `error` event); the two 502 tests PASS already.

- [x] **Step 3: Implement** in `src/internal/servers/web/app.py`

Import: `from src.context.models import ModelUnavailableError` next to the other `src.context.models` imports.

Helper after `_enforce_access`:

```python
def _dispatch_failure(exc: Exception) -> HTTPException:
    """The 502 an agent dispatch failure becomes, carrying its message."""
    logger.exception("Agent dispatch error: %s", exc)
    detail = str(exc) if str(exc).strip() else "Unexpected error during agent dispatch"
    return HTTPException(status_code=502, detail=detail)
```

Dispatch block (replacing the current `except Exception` body, adding the arm before it):

```python
            except HTTPException:
                raise
            except ModelUnavailableError as exc:
                logger.warning("Model unavailable, degrading to search-only: %s", exc)
                try:
                    (
                        answer,
                        citations,
                        documents,
                        intent,
                        extra,
                    ) = await _auto_search_pipeline(
                        query,
                        # Not the request's llm: that model is what just failed,
                        # and query expansion would call it again.
                        llm=None,
                        search_url=search_url,
                        browser_search_url=settings.browser_search_url,
                        rerank_url=settings.rerank_url,
                        top_k=top_k,
                        filters=filters,
                        history=history,
                        source_provider=_normalize_source_provider(
                            request.source_provider
                        ),
                        extra={"route_degraded": "model_unavailable"},
                        domain=domain,
                        retrieval_query=retrieval_query,
                    )
                except Exception as fallback_exc:
                    raise _dispatch_failure(fallback_exc) from fallback_exc
                extra.update(follow_up_meta)
                _cap = _capture.active()
                if _cap is not None:
                    _cap.route_degraded = extra.get("route_degraded")
                return _finalize_response(
                    db,
                    session_id,
                    query=query,
                    answer=answer,
                    citations=citations,
                    documents=documents,
                    intent=intent,
                    hook_metadata=hook_metadata,
                    extra=extra,
                    mode=normalized_mode or "auto",
                )
            except Exception as exc:
                raise _dispatch_failure(exc) from exc
```

`_auto_search_pipeline` retrieves through `_run_hybrid_search`, whose retrieval leg applies `_enforce_access` — the same code the `no_llm` path trusts; no enforcement is skipped. Because `extra` is a fresh dict built here, an earlier `route_degraded` (e.g. `tool_unavailable`) is replaced only when this arm actually produced the answer.

- [x] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/servers/web/test_model_unavailable_degrade.py tests/unit/servers/web/test_web_experience_app.py tests/unit/servers/web/test_sse_streaming.py -q -p no:cacheprovider`
Expected: PASS (incl. existing `test_agent_other_exception_returns_502_with_message`, `test_agent_no_llm_chat_degrades_to_pipeline`, and auto TOOL degradation tests).

- [x] **Step 5: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_model_unavailable_degrade.py
git commit -m "/api/agent: model unavailable degrades to search-only instead of 502"
```

---

### Task 5: Mutation checks and final verification

**Files:** none changed permanently.

- [x] **Step 1: Mutation — remove the arm.** Delete the `except ModelUnavailableError` arm in `app.py`; run `tests/unit/servers/web/test_model_unavailable_degrade.py`. Expected: degrade, ACL and stream tests go RED (502). Restore with `git checkout src/internal/servers/web/app.py`; `find src tests -name __pycache__ -type d -prune -exec rm -rf {} +`; re-run → green.
- [x] **Step 2: Mutation — map 4xx as unavailable.** Change `_is_unavailable_status` to `return response is not None and response.status_code >= 400`; run `tests/unit/test_model_unavailable_error.py`. Expected: `test_4xx_reraises_http_error` RED (and the schema cases for `_stream_complete`). Restore via `git checkout`, delete `__pycache__`, re-run → green.
- [x] **Step 3: Full verification.**
  - `.venv/bin/python -m pytest tests/unit/ -q -p no:cacheprovider` → 0 failures (~149 skipped).
  - `ruff check . && ruff format --check .`
  - `git diff --check origin/main...HEAD`
- [x] **Step 4: Commit** any formatting fixes: `git commit -am "Format"` (only if ruff changed something).
