# /chat and /tool model-unavailable degrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the local model raises `ModelUnavailableError`, `/tool/send-tool-message` answers with the corpus-only search-only fallback and `/chat/send-chat-message` answers with a fixed unavailable message, instead of `answer=""` plus `error`.

**Architecture:** Each endpoint gains one `except ModelUnavailableError` arm ahead of its existing `except Exception` arm, in both the non-stream and the SSE branch. `/tool` lazily imports `_auto_search_pipeline` from `src/internal/servers/web/app.py` (the module already lazily imports `_request_tool_approval` from there, so no import cycle), and runs it with `llm=None`, `source_provider="retrieval"` and the caller's ACL. Both response models gain an additive `degraded: str | None = None`.

**Tech Stack:** FastAPI, pydantic, pytest + TestClient.

**Spec:** `docs/superpowers/specs/2026-09-25-direct-surfaces-degrade-design.md`

## Global Constraints

- The degrade marker is exactly `"model_unavailable"` (response field `degraded`, SSE `done` key `degraded`, pipeline `extra={"route_degraded": "model_unavailable"}`).
- `CHAT_MODEL_UNAVAILABLE_MESSAGE = "The chat model is temporarily unavailable. Please try again in a moment."`, a module constant in `chat_backend.py`.
- `/tool` fallback: `llm=None`, `source_provider="retrieval"`, the surface's `search_url`, `filters=SearchFilters(access_acl=capabilities.access_acl)`, `history`, `top_k` = the surface's existing default.
- The `/tool` degraded assistant turn is persisted; the `/chat` unavailable message is NOT persisted (the user turn still is).
- Any other exception, and a failing fallback, keep today's behavior: `answer=""` plus `error`, or an SSE `error` event.
- The no-local-model 400 and `/api/agent` behavior are unchanged.

## Deviations from the spec (recorded)

1. **`top_k`.** The spec says "the surface's existing default". `/tool` has no request `top_k`; its only corpus-search size is `_CORPUS_SEARCH_TOP_K = 5` in `tool_agent_runner.py`, which the tool loop's corpus search uses. The fallback imports and uses that constant.
2. **`rerank_url` / `browser_search_url`.** The spec does not name them, but `_auto_search_pipeline` requires both. `browser_search_url=None` (corpus-only never reaches the browser); `rerank_url=resolved.services.rerank_url`, the same value `/api/agent` passes (`SearchExperienceSettings.from_app_settings` maps it from there), so both surfaces rerank alike.
3. **No shared helper was factored out of `app.py`**: a lazy import works, as `_request_tool_approval` already proves.

## Review Focus

1. A private corpus document reaching an anonymous `/tool` caller through the fallback. Expect it filtered by the real pipeline. Pinned by `test_tool_degrade_acl_filters_private_document` (Task 1).
2. A fallback that itself fails in the SSE branch. Expect an SSE `error` event, never a hang or an empty `done`. Pinned by `test_tool_fallback_failure_keeps_error[stream]` (Task 1).
3. A `ModelUnavailableError` subclass (`LLMTimeoutError`) raised by the model. Expect the same degrade, since it is caught by the base class. Covered by construction (`except ModelUnavailableError`); no separate test.
4. Chat streaming where the failure surfaces from `await task`. Expect `answer` then `done` with `degraded`, no `error`. Pinned by `test_chat_stream_degrades` (Task 2).
5. A degraded `/chat` reply polluting later context. Expect only the user turn stored. Pinned by `test_chat_degrades_with_unavailable_message` (Task 2).

---

### Task 1: `/tool/send-tool-message` search-only degrade

**Files:**
- Modify: `src/internal/servers/query_and_chat/models.py` (`ToolAgentMessageResponse`)
- Modify: `src/internal/servers/query_and_chat/tool_backend.py` (`send_tool_message`)
- Test: `tests/unit/test_direct_surfaces_degrade.py` (create)

**Interfaces:**
- Consumes: `_auto_search_pipeline(query, *, llm, search_url, browser_search_url, rerank_url, top_k, filters, history, source_provider, extra, ...) -> (answer, citations, documents, intent, extra)` from `src.internal.servers.web.app`; `_CORPUS_SEARCH_TOP_K` from `src.internal.servers.web.tool_agent_runner`; `ModelUnavailableError` from `src.context.models`.
- Produces: `ToolAgentMessageResponse.degraded: str | None = None`.

- [x] **Step 1: Write the failing tests**

```python
"""/tool and /chat degrade when the local model is unavailable."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.context.models import ContextDocument, ModelUnavailableError
from src.internal.configs import load_app_settings
from src.internal.db import AgenticSearchStore
from src.internal.servers.query_and_chat.tool_backend import create_tool_router

APP = "src.internal.servers.web.app"
TOOL_RUNNER = "src.internal.servers.web.tool_agent_runner._run_tool_agent"


async def _unavailable(*args, **kwargs):
    raise ModelUnavailableError("Cannot connect to inference server")


def _fake_pipeline(calls: list):
    async def fake(query, **kwargs):
        calls.append({"query": query, **kwargs})
        doc = ContextDocument(id="D1", title="Doc", content="body", score=0.5)
        return "search-only answer", ["D1"], [doc], "search", kwargs["extra"]

    return fake


def _events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data:") :].strip())
        for line in text.splitlines()
        if line.startswith("data:") and line[len("data:") :].strip()
    ]


def _roles(store, session_id) -> list[str]:
    return [m.role for m in store.list_chat_messages(session_id)]


def _tool_app():
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(
        create_tool_router(
            store, search_url="http://x/retrieve", resolved=load_app_settings()
        )
    )
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    return app, store


def _send_tool(app, *, stream: bool):
    return TestClient(app).post(
        "/tool/send-tool-message", json={"message": "explain FAISS", "stream": stream}
    )


def test_tool_degrades_to_search_only_answer(monkeypatch):
    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline([]))
    app, store = _tool_app()

    resp = _send_tool(app, stream=False)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "search-only answer"
    assert data["degraded"] == "model_unavailable"
    assert data["error"] is None
    assert data["tool_calls"] == [] and data["num_turns"] == 0
    assert _roles(store, data["session_id"]) == ["user", "assistant"]
    assert store.list_chat_messages(data["session_id"])[-1].content == (
        "search-only answer"
    )


def test_tool_degrade_is_corpus_only_with_caller_acl(monkeypatch):
    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))
    app, _ = _tool_app()

    assert _send_tool(app, stream=False).status_code == 200

    (call,) = calls
    assert call["query"] == "explain FAISS"
    assert call["llm"] is None
    assert call["source_provider"] == "retrieval"
    assert call["search_url"] == "http://x/retrieve"
    assert call["top_k"] == 5
    assert call["filters"].access_acl == ["public"]  # anonymous caller
    assert call["extra"] == {"route_degraded": "model_unavailable"}


def test_tool_stream_degrades(monkeypatch):
    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline([]))
    app, store = _tool_app()

    resp = _send_tool(app, stream=True)

    assert resp.status_code == 200
    events = _events(resp.text)
    assert not [e for e in events if e["type"] == "error"]
    assert [e["type"] for e in events] == ["answer", "done"]
    assert events[0]["text"] == "search-only answer"
    done = events[-1]
    assert done["degraded"] == "model_unavailable"
    assert _roles(store, done["session_id"]) == ["user", "assistant"]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_tool_other_errors_keep_error(monkeypatch, stream):
    async def explode(*args, **kwargs):
        raise ValueError("bad input format")

    monkeypatch.setattr(TOOL_RUNNER, explode)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))
    app, _ = _tool_app()

    resp = _send_tool(app, stream=stream)

    assert calls == []
    if stream:
        assert _events(resp.text)[-1] == {"type": "error", "detail": "bad input format"}
    else:
        assert resp.json()["answer"] == ""
        assert resp.json()["error"] == "bad input format"
        assert resp.json()["degraded"] is None


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_tool_fallback_failure_keeps_error(monkeypatch, stream):
    monkeypatch.setattr(TOOL_RUNNER, _unavailable)

    async def pipeline_down(query, **kwargs):
        raise RuntimeError("retrieval server unreachable")

    monkeypatch.setattr(f"{APP}._auto_search_pipeline", pipeline_down)
    app, store = _tool_app()

    resp = _send_tool(app, stream=stream)

    if stream:
        events = _events(resp.text)
        assert events[-1] == {"type": "error", "detail": "retrieval server unreachable"}
    else:
        data = resp.json()
        assert data["answer"] == ""
        assert data["error"] == "retrieval server unreachable"
        assert _roles(store, data["session_id"]) == ["user"]


def test_tool_degrade_acl_filters_private_document(monkeypatch):
    """Real fallback pipeline: a private row never reaches an anonymous caller."""
    from src.context.search import SearchResult

    rows = [
        ("Public", "ok", "http://r/pub", ["public"]),
        ("Private", "secret", "http://r/priv", ["user:alice"]),
    ]

    class _Client:
        def __init__(self, config):
            pass

        async def retrieve_one(self, query, **kwargs):
            return [
                SearchResult(contents=c, title=t, url=u, metadata={"acl": acl})
                for t, c, u, acl in rows
            ]

        async def aclose(self):
            return None

    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    monkeypatch.setattr("src.context.retrieval.search_runner.SearchClient", _Client)
    app, _ = _tool_app()

    data = _send_tool(app, stream=False).json()

    assert data["degraded"] == "model_unavailable"
    # The answer reports a count, not titles: of the two stubbed rows only the
    # public one may be counted.
    assert "returned 1 result(s)" in data["answer"]
    assert "secret" not in data["answer"]
```

- [x] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_direct_surfaces_degrade.py -q -p no:cacheprovider`
Expected: the degrade tests FAIL (`answer == ""`, no `degraded` key); `test_tool_other_errors_keep_error[stream]` passes already and `[json]` fails only on the missing `degraded` key.

- [x] **Step 3: Implement**

`models.py`, in `ToolAgentMessageResponse` after `tool_recovery`:

```python
    # "model_unavailable": the model was down and `answer` is the corpus-only
    # search-only fallback.
    degraded: str | None = None
```

`tool_backend.py`: import `ModelUnavailableError` next to `SearchFilters`; inside `send_tool_message`, after `_run`, add:

```python
        async def _search_only_answer() -> str:
            # Lazy for the same cycle as tool_agent_runner above.
            from src.internal.servers.web.app import _auto_search_pipeline
            from src.internal.servers.web.tool_agent_runner import (
                _CORPUS_SEARCH_TOP_K,
            )

            answer, *_ = await _auto_search_pipeline(
                body.message,
                # Not the model that just failed: query expansion would call it.
                llm=None,
                search_url=search_url,
                browser_search_url=None,
                rerank_url=resolved.services.rerank_url,
                top_k=_CORPUS_SEARCH_TOP_K,
                filters=SearchFilters(access_acl=capabilities.access_acl),
                history=history,
                # Corpus-only: an outage never sends the query to a web provider.
                source_provider="retrieval",
                extra={"route_degraded": "model_unavailable"},
            )
            return answer
```

Non-stream branch, before `except Exception`:

```python
            except ModelUnavailableError as exc:
                logger.warning("Model unavailable, degrading /tool to search: %s", exc)
                try:
                    answer = await _search_only_answer()
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.exception("Search-only fallback failed: %r", body.message)
                    return ToolAgentMessageResponse(
                        session_id=session_id, answer="", error=str(fallback_exc)
                    )
                store.add_chat_message(session_id, role="assistant", content=answer)
                return ToolAgentMessageResponse(
                    session_id=session_id, answer=answer, degraded="model_unavailable"
                )
```

Stream branch (`_gen`), before `except Exception`:

```python
            except ModelUnavailableError as exc:
                logger.warning("Model unavailable, degrading /tool to search: %s", exc)
                try:
                    answer = await _search_only_answer()
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.exception("Search-only fallback failed: %r", body.message)
                    yield sse_frame({"type": "error", "detail": str(fallback_exc)})
                    return
                store.add_chat_message(session_id, role="assistant", content=answer)
                yield sse_frame({"type": "answer", "text": answer})
                yield sse_frame(
                    {
                        "type": "done",
                        "session_id": session_id,
                        "tool_calls": [],
                        "num_turns": 0,
                        "truncated": False,
                        "tool_recovery": None,
                        "degraded": "model_unavailable",
                    }
                )
```

- [x] **Step 4: Run to verify they pass**, plus `tests/unit/test_tool_backend.py`.

- [x] **Step 5: Commit** — `git add` the three files; message "/tool degrades to a corpus-only search answer when the model is unavailable".

### Task 2: `/chat/send-chat-message` unavailable message

**Files:**
- Modify: `src/internal/servers/query_and_chat/models.py` (`ChatMessageResponse`)
- Modify: `src/internal/servers/query_and_chat/chat_backend.py`
- Test: `tests/unit/test_direct_surfaces_degrade.py`

**Interfaces:**
- Consumes: `ModelUnavailableError`.
- Produces: `CHAT_MODEL_UNAVAILABLE_MESSAGE: str` in `chat_backend`; `ChatMessageResponse.degraded: str | None = None`.

- [x] **Step 1: Write the failing tests** (append; add `from src.internal.servers.query_and_chat import chat_backend` and `create_chat_router` imports at the top)

```python
def _chat_app():
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(create_chat_router(store))
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    return app, store


def _send_chat(app, *, stream: bool):
    return TestClient(app).post(
        "/chat/send-chat-message", json={"message": "hello", "stream": stream}
    )


def test_chat_degrades_with_unavailable_message(monkeypatch):
    monkeypatch.setattr(chat_backend, "_run_plain_chat", _unavailable)
    app, store = _chat_app()

    data = _send_chat(app, stream=False).json()

    assert data["answer"] == CHAT_MODEL_UNAVAILABLE_MESSAGE
    assert data["degraded"] == "model_unavailable"
    assert data["error"] is None
    # The message is not content: only the user turn is stored.
    assert _roles(store, data["session_id"]) == ["user"]


def test_chat_stream_degrades(monkeypatch):
    monkeypatch.setattr(chat_backend, "_run_plain_chat", _unavailable)
    app, store = _chat_app()

    events = _events(_send_chat(app, stream=True).text)

    assert [e["type"] for e in events] == ["answer", "done"]
    assert events[0]["text"] == CHAT_MODEL_UNAVAILABLE_MESSAGE
    assert events[1]["degraded"] == "model_unavailable"
    assert _roles(store, events[1]["session_id"]) == ["user"]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_chat_other_errors_keep_error(monkeypatch, stream):
    async def explode(*args, **kwargs):
        raise ValueError("bad input format")

    monkeypatch.setattr(chat_backend, "_run_plain_chat", explode)
    app, _ = _chat_app()

    resp = _send_chat(app, stream=stream)

    if stream:
        assert _events(resp.text)[-1] == {"type": "error", "detail": "bad input format"}
    else:
        assert resp.json()["answer"] == ""
        assert resp.json()["error"] == "bad input format"
        assert resp.json()["degraded"] is None
```

- [x] **Step 2: Run to verify they fail** (ImportError on `CHAT_MODEL_UNAVAILABLE_MESSAGE` first).

- [x] **Step 3: Implement**

`models.py`, `ChatMessageResponse`: add `degraded: str | None = None`.

`chat_backend.py`: import `ModelUnavailableError` from `src.context.models`; module constant after `logger`:

```python
# Plain chat does no retrieval, so an outage has no search answer to fall back to.
CHAT_MODEL_UNAVAILABLE_MESSAGE = (
    "The chat model is temporarily unavailable. Please try again in a moment."
)
```

Non-stream, before `except Exception`:

```python
            except ModelUnavailableError as exc:
                logger.warning("Chat model unavailable: %s", exc)
                # Not persisted: it would pollute later context and summaries.
                return ChatMessageResponse(
                    session_id=session_id,
                    answer=CHAT_MODEL_UNAVAILABLE_MESSAGE,
                    degraded="model_unavailable",
                )
```

Stream (`_gen`), before `except Exception`:

```python
            except ModelUnavailableError as exc:
                logger.warning("Chat model unavailable: %s", exc)
                yield sse_frame(
                    {"type": "answer", "text": CHAT_MODEL_UNAVAILABLE_MESSAGE}
                )
                yield sse_frame(
                    {
                        "type": "done",
                        "session_id": session_id,
                        "degraded": "model_unavailable",
                    }
                )
```

- [x] **Step 4: Run to verify they pass**, plus `tests/unit/test_chat_backend.py`.

- [x] **Step 5: Commit** — message "/chat answers with a clear unavailable message when the model is down".

### Task 3: Mutation checks and final verification

- [x] Remove each of the four new `except ModelUnavailableError` arms in turn; the matching tests must go red. Restore, `find src tests -name __pycache__ -type d -exec rm -rf {} +`.
- [x] Add `store.add_chat_message(session_id, role="assistant", content=CHAT_MODEL_UNAVAILABLE_MESSAGE)` to the chat non-stream arm; `test_chat_degrades_with_unavailable_message` must go red. Restore, clear `__pycache__`.
- [x] Replace `SearchFilters(access_acl=capabilities.access_acl)` in `_search_only_answer` with `SearchFilters()`; the ACL tests must go red. Restore, clear `__pycache__`.
- [x] `.venv/bin/python -m pytest tests/unit/ -q -p no:cacheprovider`, `ruff check . && ruff format --check .`, `git diff --check origin/main...HEAD`.

## Execution note: streaming /chat wrapper (added during review)

**The bug.** `chat_backend._run_plain_chat` accepted `on_turn` but not
`on_token`. The SSE branch passes `on_token=`, so **every real streaming
`/chat` request failed with a `TypeError`**, which surfaced as an SSE
`error`. That made the new streaming degrade arm unreachable. Existing tests
monkeypatched the wrapper itself, so they never exercised it.

**The fix, folded into this PR because the spec's streaming behavior depends
on it.** The wrapper now forwards `on_token`. Two tests go through the real
wrapper by patching `plain_chat_runner._run_plain_chat`: one for tokens, and
one for the degrade path. Both were RED before the fix and GREEN after it.
