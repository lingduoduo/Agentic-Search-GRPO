# Standard Tool Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every registry tool declares `effect`, `result_kind`, `citeable` and `retries_internally`, reports failure one way (typed `ToolErrorText` or a raise the registry classifies), and strict registries reject tools that break the contract; the duplicate abstractions are retired.

**Architecture:** New declarations live on `Tool` (`src/internal/tools/base.py`); `validate_tool_contract` and a `strict` flag on `ToolRegistry` enforce them. Raised exceptions are classified centrally in `registry._failure_from_exception`. Each nonconforming tool gets a targeted fix; enforcement is switched on for the global and memory registries only after every tool conforms (Task 8), so app startup never breaks mid-branch.

**Tech Stack:** Python 3.10+, aiohttp, httpx (optional import), FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-standard-tool-interface-design.md`

## Global Constraints

- `Tool.execute(instance_id, arguments) -> (response_text, raw, metadata)` is unchanged.
- `ToolRegistry.invoke()` keeps its signature and `(response, raw, errors)` tuple; its callers see the same response text as before for every tool whose failure text is unchanged.
- Failure is reported only by returning `ToolErrorText` or raising; never a plain-string `{"error": ...}`, whole-result `"Error: ..."` prose, or success-shaped failure text. Partial success is success.
- Nothing classifies failure by searching text for "error".
- `ResultKind` values: `DOCUMENTS` (JSON array of objects with string `title`, `content`, `url`), `JSON`, `TEXT`.
- Contract rules (checked by `validate_tool_contract`): `result_kind` is not `None`; `effect is UNSPECIFIED` only when `source == "mcp"`; `citeable` ⇒ `result_kind is DOCUMENTS`.
- Exception classification: `InvalidToolInput` → `invalid_input`; `aiohttp.ClientResponseError` 400/404/422 → `invalid_input`, 429/5xx → `transient`, other 4xx → `permanent`; other `aiohttp.ClientError`, `asyncio.TimeoutError`, `TimeoutError`, `ConnectionError` → `transient`; `httpx.TransportError` (if importable) → `transient`; else `unknown`.
- `InvalidToolInput` subclasses `ValueError`.
- `asyncio.CancelledError` always propagates.
- Unit tests must pass with torch unimportable. Branch `feat/standard-tool-interface`; never commit to `main`; commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`; run `ruff check --fix` and `ruff format` on touched Python files and confirm each commit with `git log -1`. Never add `.planning/` or the untracked `docs/superpowers/{specs,plans}/2026-09-24-tool-failure-recovery*` files.

## Review Focus

1. **A tool registered into the global registry at runtime** (`POST /admin/tools/openapi`, MCP discovery at startup) that violates the contract must be rejected with a clear message, not crash app startup or the admin request with an unhandled 500 — test in Task 8.
2. **A citeable `DOCUMENTS` tool whose items carry a missing or non-string field** (e.g. a web page without a URL) must still produce a valid document array (empty string, not `null`) so the source-card extractor does not drop the whole result — test in Task 4.
3. **`web_search` where some queries fail and others succeed** must return the successful documents, not a failure, and the failed queries must not appear as items — test in Task 4.
4. **A domain capability whose own failure is `invalid_input`** (a bad ticker via `search_domain`) must surface as `invalid_input` through `DomainSearch`, and an upstream capability failure must not — test in Task 5.
5. **Memory curation's `registry.invoke` text for "not found"** must stay byte-identical (`memory not found`) after the tool starts returning `ToolErrorText` — test in Task 6.

---

### Task 1: Contract types, validation and exception classification

**Files:**
- Modify: `src/internal/tools/base.py` (`ResultKind`, `InvalidToolInput`, `Tool.result_kind`, `Tool.retries_internally`, `FunctionTool`/`from_fn` kwargs)
- Modify: `src/internal/tools/registry.py` (`validate_tool_contract`, `ToolRegistry(strict=False)`, `_failure_from_exception`, `tool()` decorators forward, summaries)
- Modify: `src/internal/tools/__init__.py` (exports)
- Modify: `src/internal/servers/tools/api.py` (`ToolView` fields)
- Test: `tests/unit/test_tool_contract.py`

**Interfaces:**
- Produces: `ResultKind` (str Enum `DOCUMENTS="documents"`, `JSON="json"`, `TEXT="text"`); `InvalidToolInput(ValueError)`; `Tool.result_kind -> ResultKind | None` (default `None`); `Tool.retries_internally -> bool` (default `False`); `FunctionTool(..., result_kind: ResultKind | None = None, retries_internally: bool = False)` and the same kwargs on `FunctionTool.from_fn`; `validate_tool_contract(tool: Tool, *, source: str) -> list[str]`; `ToolRegistry(strict: bool = False)`; summaries gain `effect`, `result_kind`, `citeable`, `retries_internally` (string values for enums, `None` for an undeclared `result_kind`).

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_tool_contract.py`:

```python
import asyncio

import aiohttp
import pytest

from src.internal.tools import (
    FailureCategory,
    FunctionTool,
    InvalidToolInput,
    ResultKind,
    ToolEffect,
    ToolRegistry,
)
from src.internal.tools.registry import _failure_from_exception, validate_tool_contract


def _tool(**kwargs):
    return FunctionTool(lambda: "ok", name="t", description="t", **kwargs)


def test_a_fully_declared_tool_has_no_violations():
    tool = _tool(effect=ToolEffect.READ_ONLY, result_kind=ResultKind.JSON)
    assert validate_tool_contract(tool, source="function") == []


@pytest.mark.parametrize(
    ("kwargs", "source", "fragment"),
    [
        ({"effect": ToolEffect.READ_ONLY}, "function", "result_kind"),
        ({"result_kind": ResultKind.JSON}, "function", "effect"),
        (
            {"effect": ToolEffect.READ_ONLY, "result_kind": ResultKind.TEXT, "citeable": True},
            "function",
            "citeable",
        ),
    ],
)
def test_each_rule_reports_a_violation(kwargs, source, fragment):
    problems = validate_tool_contract(_tool(**kwargs), source=source)
    assert problems and any(fragment in p for p in problems)


def test_unspecified_is_allowed_only_for_mcp():
    tool = _tool(result_kind=ResultKind.TEXT)
    assert validate_tool_contract(tool, source="mcp") == []
    assert validate_tool_contract(tool, source="function")


def test_strict_registry_rejects_and_names_the_tool():
    with pytest.raises(ValueError, match=r"tool t: .*result_kind"):
        ToolRegistry(strict=True).register(_tool(effect=ToolEffect.READ_ONLY))


def test_default_registry_still_accepts_undeclared_tools():
    registry = ToolRegistry()
    registry.register(_tool())
    assert registry.get("t") is not None


def test_decorator_forwards_the_declarations():
    registry = ToolRegistry(strict=True)

    @registry.tool(effect=ToolEffect.READ_ONLY, result_kind=ResultKind.JSON, retries_internally=True)
    def double(n: int) -> int:
        return n * 2

    tool = registry.get("double")
    assert (tool.result_kind, tool.retries_internally) == (ResultKind.JSON, True)


def test_summaries_expose_the_declarations():
    registry = ToolRegistry()
    registry.register(_tool(effect=ToolEffect.READ_ONLY, result_kind=ResultKind.DOCUMENTS, citeable=True))
    [summary] = registry.all_summaries()
    assert {k: summary[k] for k in ("effect", "result_kind", "citeable", "retries_internally")} == {
        "effect": "read_only", "result_kind": "documents", "citeable": True, "retries_internally": False,
    }
    assert registry.tool_summary("t")["result_kind"] == "documents"


def _response_error(status):
    return aiohttp.ClientResponseError(request_info=None, history=(), status=status)


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (InvalidToolInput("bad ticker"), FailureCategory.INVALID_INPUT),
        (_response_error(400), FailureCategory.INVALID_INPUT),
        (_response_error(404), FailureCategory.INVALID_INPUT),
        (_response_error(422), FailureCategory.INVALID_INPUT),
        (_response_error(401), FailureCategory.PERMANENT),
        (_response_error(403), FailureCategory.PERMANENT),
        (_response_error(418), FailureCategory.PERMANENT),
        (_response_error(429), FailureCategory.TRANSIENT),
        (_response_error(503), FailureCategory.TRANSIENT),
        (aiohttp.ClientConnectionError(), FailureCategory.TRANSIENT),
        (asyncio.TimeoutError(), FailureCategory.TRANSIENT),
        (ConnectionError(), FailureCategory.TRANSIENT),
        (KeyError("x"), FailureCategory.UNKNOWN),
    ],
)
def test_exception_classification(exc, category):
    assert _failure_from_exception(exc).category is category


def test_invalid_tool_input_keeps_its_message_and_is_a_value_error():
    exc = InvalidToolInput("max_results must be an integer")
    assert isinstance(exc, ValueError)
    assert _failure_from_exception(exc).message == "max_results must be an integer"


def test_httpx_transport_errors_are_transient():
    httpx = pytest.importorskip("httpx")
    assert _failure_from_exception(httpx.ConnectError("down")).category is FailureCategory.TRANSIENT
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/unit/test_tool_contract.py`
Expected: collection error — `ImportError: cannot import name 'InvalidToolInput'`.

- [ ] **Step 3: Implement**

`base.py`, after `ToolEffect`:

```python
class ResultKind(str, Enum):
    """What a tool's successful response text is."""

    DOCUMENTS = "documents"  # JSON array of {title, content, url} (string values)
    JSON = "json"
    TEXT = "text"


class InvalidToolInput(ValueError):
    """A tool argument is wrong; the model should correct it and call again."""
```

On `Tool`, next to `effect`/`citeable`:

```python
    @property
    def result_kind(self) -> "ResultKind | None":
        """What a successful response is. None = undeclared (a strict registry rejects it)."""
        return None

    @property
    def retries_internally(self) -> bool:
        """True when the provider already retries transient failures itself."""
        return False
```

`FunctionTool.__init__` gains `result_kind: ResultKind | None = None, retries_internally: bool = False` (store as `self._result_kind`, `self._retries_internally`) with matching properties; `from_fn` gains the same two kwargs and passes them through. Leave `stopping` alone in this task (Task 7 removes it).

`registry.py`:

```python
def validate_tool_contract(tool: Tool, *, source: str) -> list[str]:
    """The ways *tool* breaks the standard tool contract (empty = conforms)."""
    problems: list[str] = []
    if tool.result_kind is None:
        problems.append("result_kind must be declared")
    if tool.effect is ToolEffect.UNSPECIFIED and source != "mcp":
        problems.append("effect must be declared (UNSPECIFIED is only for source='mcp')")
    if tool.citeable and tool.result_kind is not ResultKind.DOCUMENTS:
        problems.append("citeable tools must declare result_kind DOCUMENTS")
    return problems
```

`ToolRegistry.__init__(self, *, strict: bool = False)` stores `self._strict`; `register` first does:

```python
        if self._strict:
            problems = validate_tool_contract(tool, source=source)
            if problems:
                raise ValueError(f"tool {tool.name}: " + "; ".join(problems))
```

`ToolRegistry.tool(...)` replaces its `stopping` kwarg handling to additionally accept and forward `result_kind: ResultKind | None = None` and `retries_internally: bool = False` (keep `stopping` until Task 7); the module-level `tool(...)` shorthand accepts and forwards `effect`, `citeable`, `result_kind` and `retries_internally`.

Replace `_failure_from_exception`:

```python
_INPUT_STATUSES = frozenset({400, 404, 422})


def _failure_from_exception(exc: Exception) -> ToolFailure:
    if isinstance(exc, InvalidToolInput):
        return ToolFailure(FailureCategory.INVALID_INPUT, str(exc))
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - aiohttp is a declared dependency
        aiohttp = None
    if aiohttp is not None and isinstance(exc, aiohttp.ClientResponseError):
        status = exc.status
        if status in _INPUT_STATUSES:
            category = FailureCategory.INVALID_INPUT
        elif status == 429 or status >= 500:
            category = FailureCategory.TRANSIENT
        else:
            category = FailureCategory.PERMANENT
        return ToolFailure(category, type(exc).__name__)
    transient = isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError))
    if aiohttp is not None:
        transient = transient or isinstance(exc, aiohttp.ClientError)
    try:
        import httpx

        transient = transient or isinstance(exc, httpx.TransportError)
    except ImportError:
        pass
    category = FailureCategory.TRANSIENT if transient else FailureCategory.UNKNOWN
    return ToolFailure(category, type(exc).__name__)
```

`tool_summary` and `all_summaries` add (build both from one private `_summary(entry)` helper so they cannot drift):

```python
            "effect": t.effect.value,
            "result_kind": t.result_kind.value if t.result_kind is not None else None,
            "citeable": t.citeable,
            "retries_internally": t.retries_internally,
```

`servers/tools/api.py` `ToolView` adds `effect: str = "unspecified"`, `result_kind: str | None = None`, `citeable: bool = False`, `retries_internally: bool = False`.

Export `ResultKind` and `InvalidToolInput` from `src/internal/tools/__init__.py` next to `ToolEffect`.

- [ ] **Step 4: Run tests**

Run: `pytest -q tests/unit/test_tool_contract.py tests/unit/test_tool_registry.py tests/unit/test_tool_failures.py tests/unit/servers/web/test_tool_admin_api.py`
Expected: all PASS.

- [ ] **Step 5: Mutation-check** — (a) drop the `source != "mcp"` clause → `test_unspecified_is_allowed_only_for_mcp` FAILS; (b) remove the `if self._strict:` block → `test_strict_registry_rejects_and_names_the_tool` FAILS; (c) move the `ClientResponseError` branch after the generic `ClientError` check → the 404 case FAILS. Restore, re-run.

- [ ] **Step 6: Commit**

```bash
git add src/internal/tools/base.py src/internal/tools/registry.py src/internal/tools/__init__.py src/internal/servers/tools/api.py tests/unit/test_tool_contract.py docs/superpowers/plans/2026-09-24-standard-tool-interface.md
git commit -m "tools: declared tool contract, strict registries and status-aware classification"
```

---

### Task 2: Recovery honours declared retry ownership

**Files:**
- Modify: `src/agents/tool/recovery.py` (`RecoveryPolicy.decide`)
- Modify: `src/agents/tool/tool_calling.py` (`_call_tool` passes the tool's declaration)
- Test: `tests/unit/test_tool_recovery_policy.py`, `tests/unit/test_tool_recovery_loop.py` (append)

**Interfaces:**
- Consumes: `Tool.retries_internally` (Task 1).
- Produces: `RecoveryPolicy.decide(failure, effect, retries_so_far, budget_left, uniform=random.uniform, *, retries_internally: bool = False)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tool_recovery_policy.py`:

```python
def test_a_tool_that_retries_internally_is_not_retried_again():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 0, 10.0, MID, retries_internally=True
    )
    assert got.action is Action.UNAVAILABLE
```

Append to `tests/unit/test_tool_recovery_loop.py` (reuse its `_loop`, `_trace`, `CALL` helpers):

```python
def test_loop_does_not_retry_a_tool_that_retries_internally():
    calls = []

    @FunctionTool.from_fn(
        name="lookup", effect=ToolEffect.READ_ONLY, result_kind=ResultKind.JSON, retries_internally=True
    )
    async def lookup():
        calls.append(1)
        return ToolErrorText('{"error": "x"}', ToolFailure(FailureCategory.TRANSIENT, "upstream temporarily unavailable"))

    loop, _ = _loop([lookup], [CALL, "done"])
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    assert len(calls) == 1
    assert output.tool_recovery["degraded"] == ["lookup"]
```

(add `ResultKind` to that file's `src.internal.tools` import).

- [ ] **Step 2: Run to verify failure**

Run: `pytest -q tests/unit/test_tool_recovery_policy.py tests/unit/test_tool_recovery_loop.py -k retries_internally`
Expected: FAIL (`unexpected keyword argument 'retries_internally'`; loop test sees 3 calls).

- [ ] **Step 3: Implement** — in `decide`, add the keyword-only parameter and change the retry condition to `failure.category is FailureCategory.TRANSIENT and not retries_internally and failure.provider_attempts <= 1 and retries_so_far < self.max_retries` (comment: "declared ownership first; provider_attempts covers tools that report it"). In `_call_tool`, read `retries_internally = tool.retries_internally if tool is not None else False` next to `effect`, and pass `retries_internally=retries_internally` to `self._recovery.decide(...)`.

- [ ] **Step 4: Run tests** — `pytest -q tests/unit/test_tool_recovery_policy.py tests/unit/test_tool_recovery_loop.py` → all PASS.

- [ ] **Step 5: Mutation-check** — drop `not retries_internally and` → both new tests FAIL. Restore.

- [ ] **Step 6: Commit** — `git add src/agents/tool/recovery.py src/agents/tool/tool_calling.py tests/unit/test_tool_recovery_policy.py tests/unit/test_tool_recovery_loop.py` and commit `agents: recovery honours a tool's declared retry ownership`.

---

### Task 3: Corpus search, RAG routing and public-data tools conform

**Files:**
- Modify: `src/internal/tools/routing_tools.py`
- Modify: `src/internal/tools/public_data/knowledge.py`, `geo.py`, `market.py` (declarations only)
- Test: `tests/unit/test_tool_conformance.py` (new; grows in Tasks 4-6 and 8)

**Interfaces:**
- Consumes: `ResultKind`, `validate_tool_contract` (Task 1).
- Produces: `tests/unit/test_tool_conformance.py` with helper `assert_documents(text: str)` used by later tasks.

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_tool_conformance.py`:

```python
import asyncio
import json

import pytest

from src.internal.tools import FailureCategory, ResultKind, ToolErrorText
from src.internal.tools.registry import validate_tool_contract
from src.internal.tools.search import SearchPage


def assert_documents(text: str) -> list[dict]:
    items = json.loads(text)
    assert isinstance(items, list)
    for item in items:
        assert set(item) >= {"title", "content", "url"}
        assert all(isinstance(item[k], str) for k in ("title", "content", "url"))
    return items


def _run(tool, **arguments):
    async def go():
        instance = await tool.create()
        try:
            return await tool.execute(instance, arguments)
        finally:
            await tool.release(instance)

    return asyncio.run(go())


def test_public_data_tools_conform():
    from src.internal.tools.public_data import public_data_tools

    tools = public_data_tools()
    assert len(tools) == 9
    for tool in tools:
        assert validate_tool_contract(tool, source="function") == [], tool.name
    by_name = {t.name: t for t in tools}
    assert {n for n, t in by_name.items() if t.result_kind is ResultKind.DOCUMENTS} == {
        "search_wikipedia", "search_arxiv", "search_wayback",
    }
    assert {n for n, t in by_name.items() if not t.retries_internally} == {"search_nearby_places"}


def _corpus_tool(monkeypatch, pages):
    from src.internal.tools import routing_tools

    async def fake_search_tool(query, **kwargs):
        return pages

    monkeypatch.setattr(routing_tools, "search_tool", fake_search_tool)
    return routing_tools.build_search_routing_tool(search_url="http://x/retrieve", top_k=5)


def test_corpus_search_conforms_and_returns_documents(monkeypatch):
    tool = _corpus_tool(monkeypatch, [SearchPage(title="A", summary="a", url="http://a")])
    assert validate_tool_contract(tool, source="function") == []
    assert tool.retries_internally is True
    response, _raw, meta = _run(tool, query="q")
    assert assert_documents(response)[0]["url"] == "http://a"
    assert meta == {}


def test_corpus_search_outage_is_a_typed_transient_failure(monkeypatch):
    tool = _corpus_tool(monkeypatch, [SearchPage(error="connection refused")])
    response, _raw, meta = _run(tool, query="q")
    assert isinstance(response, ToolErrorText)
    assert json.loads(response) == {"error": "connection refused"}
    assert meta["failure"].category is FailureCategory.TRANSIENT


def test_corpus_search_with_no_hits_is_an_empty_success(monkeypatch):
    tool = _corpus_tool(monkeypatch, [])
    response, _raw, meta = _run(tool, query="q")
    assert json.loads(response) == [] and meta == {}


def test_rag_routing_tool_conforms_and_lets_errors_raise(monkeypatch):
    from src.internal.tools import routing_tools

    tool = routing_tools.build_rag_routing_tool(llm=object(), search_url="http://x", top_k=3)
    assert validate_tool_contract(tool, source="function") == []
    assert tool.result_kind is ResultKind.JSON

    async def boom(*args, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr("src.context.answer_with_retrieval", boom)
    with pytest.raises(RuntimeError):
        _run(tool, query="q")
```

`public_data_tools()` (`src/internal/tools/public_data/__init__.py`) builds the nine tools; `SearchPage`'s `title`/`summary`/`url` default to `""` and `error` to `None`.

- [ ] **Step 2: Run to verify failure** — `pytest -q tests/unit/test_tool_conformance.py` → FAIL (violations: `result_kind must be declared`; outage returns a plain `str`).

- [ ] **Step 3: Implement**

`routing_tools.build_search_routing_tool`: `result_kind=ResultKind.DOCUMENTS, retries_internally=True` on the `FunctionTool`; replace the outage branch with:

```python
        if not results and any(p.error for p in pages):
            errors = [p.error for p in pages if p.error]
            return ToolErrorText(
                json.dumps({"error": errors[0]}),
                ToolFailure(
                    FailureCategory.TRANSIENT,
                    "search backend unavailable",
                    provider_attempts=_SEARCH_ATTEMPTS,
                ),
            )
```

with `_SEARCH_ATTEMPTS = 3` defined beside `_SEARCH_TOOL_PARAMS` and a comment that it mirrors `search_tool`'s default `max_retries`. (Read `search_tool`'s signature in `search.py` and confirm the default is 3; if it differs, use the real value.)

`build_rag_routing_tool`: delete the `try/except` so exceptions propagate; add `result_kind=ResultKind.JSON`.

Public-data: add `result_kind=ResultKind.DOCUMENTS, retries_internally=True` to `search_wikipedia`, `search_arxiv`, `search_wayback`; `result_kind=ResultKind.JSON, retries_internally=True` to `get_weather`, `search_location`, `get_stock_quote`, `get_crypto_price`, `convert_currency`; `result_kind=ResultKind.JSON, retries_internally=False` to `search_nearby_places` with a comment that it POSTs to Overpass and `_fetch` never retries a POST.

- [ ] **Step 4: Run tests** — `pytest -q tests/unit/test_tool_conformance.py tests/unit/test_public_data_seeding.py tests/unit/test_knowledge_base.py tests/unit/test_agent_callable_tools.py` → all PASS.

- [ ] **Step 5: Mutation-check** — (a) return `json.dumps({"error": ...})` again in the outage branch → the outage test FAILS; (b) set `search_nearby_places` to `retries_internally=True` → the public-data test FAILS. Restore.

- [ ] **Step 6: Commit** — stage the changed modules and the new test; commit `tools: corpus search, RAG routing and public-data tools declare the contract`.

---

### Task 4: `web_search` returns documents and typed failures

**Files:**
- Modify: `src/internal/tools/search.py` (`MultiQueryWebSearchTool`)
- Modify: existing web_search tests in `tests/unit/test_search_tools.py` whose assertions encode the prose format (update them to the document format)
- Test: `tests/unit/test_tool_conformance.py` (append)

**Interfaces:**
- Consumes: `ResultKind`, `ToolErrorText`, `ToolFailure`, `FailureCategory`; `assert_documents` (Task 3).

- [ ] **Step 1: Write the failing tests** (append):

```python
from src.internal.tools.search import MultiQueryWebSearchTool


def _web_tool(pages_by_query):
    async def fake_search(query, **kwargs):
        return pages_by_query[query]

    return MultiQueryWebSearchTool(search_fn=fake_search)


def test_web_search_conforms():
    tool = _web_tool({})
    assert validate_tool_contract(tool, source="function") == []
    assert (tool.effect.value, tool.result_kind, tool.citeable) == ("read_only", ResultKind.DOCUMENTS, True)


def test_web_search_returns_deduplicated_documents():
    tool = _web_tool({
        "a": [SearchPage(title="A", summary="sa", url="http://1"), SearchPage(title="B", summary="sb", url="http://2")],
        "b": [SearchPage(title="A again", summary="x", url="http://1")],
    })
    response, raw, meta = _run(tool, queries=["a", "b"])
    items = assert_documents(response)
    assert [i["url"] for i in items] == ["http://1", "http://2"]
    assert items[0] == {"title": "A", "content": "sa", "url": "http://1"}
    assert meta["queries"] == ["a", "b"]


def test_web_search_partial_failure_keeps_the_successes():
    tool = _web_tool({
        "a": [SearchPage(title="A", summary="sa", url="http://1")],
        "b": [SearchPage(error="rate limited")],
    })
    response, _raw, meta = _run(tool, queries=["a", "b"])
    assert "failure" not in meta
    assert [i["url"] for i in assert_documents(response)] == ["http://1"]


def test_web_search_total_failure_is_typed_unknown():
    tool = _web_tool({"a": [SearchPage(error="no provider")], "b": [SearchPage(error="rate limited")]})
    response, _raw, meta = _run(tool, queries=["a", "b"])
    assert isinstance(response, ToolErrorText)
    assert meta["failure"].category is FailureCategory.UNKNOWN
    assert json.loads(response) == {"error": "no provider"}


def test_web_search_missing_fields_become_empty_strings():
    tool = _web_tool({"a": [SearchPage(title="", summary="", url="")]})
    response, _raw, _meta = _run(tool, queries=["a"])
    assert assert_documents(response) == [{"title": "", "content": "", "url": ""}]


def test_web_search_runs_without_an_approval_callback():
    from src.agents import ToolAgentLoop, ToolAgentLoopConfig
    from tests.unit.test_tool_recovery_loop import _Manager, _Tokenizer

    tool = _web_tool({"q": [SearchPage(title="A", summary="s", url="http://1")]})
    tokenizer = _Tokenizer()
    manager = _Manager(tokenizer, ['{"name":"web_search","arguments":{"queries":["q"]}}', "done"])
    loop = ToolAgentLoop(tokenizer, manager, [tool], ToolAgentLoopConfig(response_length=8192))
    output = asyncio.run(loop.run([{"role": "user", "content": "go"}], {}))
    trace = [json.loads(line) for line in output.action_trace.splitlines()]
    assert trace[0]["status"] == str(TaskStatus.COMPLETED)
```

(add `from src.agents.core.state import TaskStatus` to the file's imports.)

- [ ] **Step 2: Run to verify failure** — `pytest -q tests/unit/test_tool_conformance.py -k web_search` → FAIL (violations; prose response; approval-denied SKIPPED).

- [ ] **Step 3: Implement** — in `MultiQueryWebSearchTool`: add properties `effect` → `ToolEffect.READ_ONLY`, `result_kind` → `ResultKind.DOCUMENTS` (`retries_internally` inherits `False`). In `execute`, after `results_per_query` is gathered:

```python
        seen_urls: set[str] = set()
        merged: list[SearchPage] = []
        errors: list[str] = []
        for pages in results_per_query:
            for page in pages:
                if page.error:
                    errors.append(page.error)
                    continue
                if page.url and page.url in seen_urls:
                    continue
                if page.url:
                    seen_urls.add(page.url)
                merged.append(page)

        metadata: dict[str, Any] = {"queries": queries}
        if domain != "general":
            metadata.update(domain=domain, executed_queries=executed_queries)
        if not merged and errors:
            # A Tool subclass (unlike FunctionTool) must put the failure into
            # its own metadata: that is how invoke_detailed learns about it.
            failure = ToolFailure(FailureCategory.UNKNOWN, "web search unavailable")
            return (
                ToolErrorText(json.dumps({"error": errors[0]}), failure),
                merged,
                {**metadata, "failure": failure},
            )
        documents = [
            {"title": p.title or "", "content": p.summary or "", "url": p.url or ""}
            for p in merged
        ]
        return json.dumps(documents, ensure_ascii=False), merged, metadata
```

Keep the empty-queries early return (`"No results found."` becomes `json.dumps([])` so the result still matches `DOCUMENTS`). Leave `format_search_pages` in place if anything else still imports it (`grep -rn "format_search_pages" src tests`); otherwise delete it with `_render_sections` if unused.

Update the prose-format assertions in `tests/unit/test_search_tools.py` (and any other test the grep in Step 3 finds) to the document format, and list each changed test in the report.

- [ ] **Step 4: Run tests** — `pytest -q tests/unit/test_tool_conformance.py tests/unit/test_search_tools.py tests/unit/servers/web/test_loop_runners.py` → all PASS.

- [ ] **Step 5: Mutation-check** — (a) keep failed pages in `merged` → the partial-failure test FAILS; (b) drop `"failure"` from the failure metadata → the total-failure test FAILS; (c) remove the `effect` property → the approval test FAILS. Restore.

- [ ] **Step 6: Commit** — stage `search.py` and the tests; commit `tools: web_search returns documents and typed failures, and is read-only`.

---

### Task 5: Domain tools raise `InvalidToolInput`

**Files:**
- Modify: `src/internal/tools/search.py` (`normalize_search_domain`, `DomainSearch`, `build_domain_search_tools`, delete `build_search_tool`)
- Modify: `src/internal/tools/public_data/_http.py` (`guarded`)
- Modify: `src/internal/tools/__init__.py` (drop the `build_search_tool` export)
- Test: `tests/unit/test_tool_conformance.py` (append); update `tests/unit/test_search_domain_tools.py` / `tests/unit/test_tools_package_layout.py` where they encode `ValueError`-by-type or the `build_search_tool` export

**Interfaces:**
- Consumes: `InvalidToolInput`, `ResultKind` (Task 1).

- [ ] **Step 1: Write the failing tests** (append):

```python
from src.internal.tools import InvalidToolInput
from src.internal.tools.search import DomainSearch, build_domain_search_tools


def test_domain_tools_conform():
    tools = {t.name: t for t in build_domain_search_tools(service=DomainSearch())}
    for tool in tools.values():
        assert validate_tool_contract(tool, source="function") == [], tool.name
    assert tools["extract_page"].result_kind is ResultKind.DOCUMENTS
    assert tools["search_domain"].citeable is False
    assert all(not t.retries_internally for t in tools.values())


def test_search_domain_bad_argument_is_invalid_input_with_the_same_text():
    tool = {t.name: t for t in build_domain_search_tools(service=DomainSearch())}["search_domain"]
    response, _raw, meta = _run(tool, query="q", max_results="many")
    assert meta["failure"].category is FailureCategory.INVALID_INPUT
    assert json.loads(response) == {"error": "InvalidToolInput: max_results must be an integer"}


def test_extract_page_dead_link_is_invalid_input():
    async def fetch(url, max_length):
        return "[fetch error] 404"

    tool = {t.name: t for t in build_domain_search_tools(service=DomainSearch(fetch_fn=fetch))}["extract_page"]
    _response, _raw, meta = _run(tool, url="http://dead.example")
    assert meta["failure"].category is FailureCategory.INVALID_INPUT


def _failing_capability(category):
    """A stand-in for the stock-quote capability whose call fails with *category*."""
    from src.internal.tools import FunctionTool, ToolEffect, ToolFailure
    from src.internal.tools.search import iter_capabilities

    tag, capability = next(
        (t, c) for t, c in iter_capabilities() if c.tool_name == "get_stock_quote"
    )

    async def fn(**kwargs):
        return ToolErrorText(
            json.dumps({"error": "invalid ticker symbol 'APPL'"}),
            ToolFailure(category, "m"),
        )

    tool = FunctionTool(
        fn,
        name="get_stock_quote",
        effect=ToolEffect.READ_ONLY,
        result_kind=ResultKind.JSON,
        parameters={
            "type": "object",
            "properties": {capability.query_parameter: {"type": "string"}},
        },
    )
    return DomainSearch(tools=[tool], web_search_fn=lambda *a, **k: None), tag


def test_capability_input_failure_surfaces_as_invalid_input():
    service, tag = _failing_capability(FailureCategory.INVALID_INPUT)
    with pytest.raises(InvalidToolInput, match="invalid ticker"):
        asyncio.run(service.search("APPL", tag=tag))


def test_capability_upstream_failure_is_not_invalid_input():
    service, tag = _failing_capability(FailureCategory.PERMANENT)
    with pytest.raises(ValueError) as caught:
        asyncio.run(service.search("APPL", tag=tag))
    assert not isinstance(caught.value, InvalidToolInput)
```

Also confirm the `{"error": "InvalidToolInput: ..."}` text format against `guarded`'s implementation after Step 3 and adjust the assertion if the chosen format differs — the requirement is that the text names the problem.

- [ ] **Step 2: Run to verify failure** — `pytest -q tests/unit/test_tool_conformance.py -k "domain or extract_page or capability"` → FAIL.

- [ ] **Step 3: Implement**

- `normalize_search_domain`: raise `InvalidToolInput` instead of `ValueError`.
- `DomainSearch`: every `raise ValueError(...)` that validates caller arguments (the `query`/`max_results`/`tag`/`domain`/params/`domains`/`url`/`max_length`/batch-size checks, and `extract`'s `[fetch error]` branch) becomes `raise InvalidToolInput(...)`. The two capability-result checks change as follows:

```python
        payload = json.loads(response)
        if isinstance(payload, dict) and "error" in payload:
            failure = getattr(response, "failure", None)
            if failure is not None and failure.category is FailureCategory.INVALID_INPUT:
                raise InvalidToolInput(str(payload["error"]))
            raise ValueError(str(payload["error"]))
        if not isinstance(payload, (list, dict)):
            raise ValueError("capability returned an unsupported result")
```

- `guarded`: add, before the generic `except Exception`:

```python
        except InvalidToolInput as exc:
            return ToolErrorText(
                json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                ToolFailure(FailureCategory.INVALID_INPUT, str(exc)),
            )
```

- `build_domain_search_tools`: delete the `extract` wrapper's `try/except` (pass `service.extract` directly); the `definitions` tuples gain a `result_kind` element (`search_domain` `JSON`, `get_sub_domains` `JSON`, `extract_page` `DOCUMENTS`, `batch_search` `JSON`) and `search_domain`'s citeable flag becomes `False`; pass `result_kind=result_kind` to each `FunctionTool`.
- Delete `build_search_tool` and its export; if `search_for_tool_string` then has no other caller (`grep`), leave it (out of scope) unless a test imports only via `build_search_tool`.

Update existing tests that assert `ValueError` by exact type where `InvalidToolInput` now appears only if they use `type(exc) is ValueError`; `pytest.raises(ValueError)` keeps passing. Update `test_tools_package_layout.py` if it lists `build_search_tool`.

- [ ] **Step 4: Run tests** — `pytest -q tests/unit/test_tool_conformance.py tests/unit/test_search_domain_tools.py tests/unit/test_domain_search_tools.py tests/unit/test_tools_package_layout.py tests/unit/test_search_tools.py` → all PASS.

- [ ] **Step 5: Mutation-check** — (a) remove `guarded`'s `InvalidToolInput` clause → the search_domain test FAILS (category unknown); (b) make the capability branch always raise `InvalidToolInput` → `test_capability_upstream_failure_is_not_invalid_input` FAILS. Restore.

- [ ] **Step 6: Commit** — stage and commit `tools: domain tools report bad arguments as InvalidToolInput`.

---

### Task 6: OpenAPI, MCP and memory tools conform

**Files:**
- Modify: `src/internal/tools/api.py` (`ApiRequestTool.result_kind`)
- Modify: `src/internal/tools/mcp_client.py` (`_build_tool`)
- Modify: `src/internal/memory/tools.py` (declarations, typed failures)
- Test: `tests/unit/test_tool_conformance.py` (append)

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_mcp_client_tool_conforms_as_unspecified_text():
    from types import SimpleNamespace

    from src.internal.tools.mcp_client import McpServerSpec, _build_tool

    remote = SimpleNamespace(name="remote_op", description="d", inputSchema={"type": "object"})
    tool = _build_tool(McpServerSpec(name="srv", url="http://mcp"), remote)
    assert validate_tool_contract(tool, source="mcp") == []
    assert (tool.effect.value, tool.result_kind) == ("unspecified", ResultKind.TEXT)


def test_openapi_tool_declares_json():
    from src.internal.tools.api import ApiRequestTool

    assert ApiRequestTool.result_kind.fget(object.__new__(ApiRequestTool)) is ResultKind.JSON


class _Store:
    def add_user_memory(self, user_id, content):
        return None

    def update_user_memory(self, user_id, memory_id, content):
        return None

    def delete_user_memory(self, user_id, memory_id):
        return False


def test_memory_tools_conform_and_type_their_failures():
    from src.internal.memory.tools import build_memory_registry

    registry, _counts, _schemas = build_memory_registry(_Store(), "u1")
    for tool in registry.list_tools():
        assert validate_tool_contract(tool, source="function") == [], tool.name
        assert tool.result_kind is ResultKind.TEXT
    for name, args in (("add_memory", {"content": ""}), ("update_memory", {"memory_id": "m", "content": "c"}), ("delete_memory", {"memory_id": "m"})):
        outcome = asyncio.run(registry.invoke_detailed(name, args))
        assert outcome.failure.category is FailureCategory.INVALID_INPUT, name


def test_memory_invoke_text_is_unchanged_for_not_found():
    from src.internal.memory.tools import build_memory_registry

    registry, _counts, _schemas = build_memory_registry(_Store(), "u1")
    response, _raw, errors = asyncio.run(registry.invoke("delete_memory", {"memory_id": "m"}))
    assert (response, errors) == ("memory not found", [])
```

`McpServerSpec(name, url)` is its minimal constructor (the other fields default).

- [ ] **Step 2: Run to verify failure** — `pytest -q tests/unit/test_tool_conformance.py -k "mcp or openapi or memory"` → FAIL.

- [ ] **Step 3: Implement**

- `ApiRequestTool`: `result_kind` property → `ResultKind.JSON`.
- `_build_tool`: pass `result_kind=ResultKind.TEXT` to the `FunctionTool` (effect stays `UNSPECIFIED`; keep the comment).
- Memory tools: each class gets `result_kind` → `ResultKind.TEXT`; the three failure returns become `ToolErrorText("<same text>", ToolFailure(FailureCategory.INVALID_INPUT, "<same text>"))` — `"empty content; nothing added"` and `"memory not found"` — returned as `(ToolErrorText(...), None, {"failure": failure})` (a `Tool` subclass sets `meta["failure"]` itself).

- [ ] **Step 4: Run tests** — `pytest -q tests/unit/test_tool_conformance.py tests/unit/memory tests/unit/test_mcp_client*.py tests/unit/test_api_tools*.py` (adjust globs to the files that exist) → all PASS.

- [ ] **Step 5: Mutation-check** — return a plain `"memory not found"` again from `delete_memory` → the memory typed-failure test FAILS while the invoke-text test still PASSES (that is the point of pairing them). Restore.

- [ ] **Step 6: Commit** — `tools: OpenAPI, MCP and memory tools declare the contract; memory failures are typed`.

---

### Task 7: Retire the duplicate abstractions

**Files:**
- Delete: `src/internal/tools/interface.py`, `src/internal/chat/tool_call_args_streaming.py`, `src/internal/tools/built_in_tools.py`
- Modify: `src/internal/chat/llm_step.py` (remove the import at ~line 26, the `Parser` import at ~line 186, `arg_parsers` at ~line 1260, and the `yield from maybe_emit_argument_delta(...)` block at ~lines 1465-1470 — keep `_update_tool_call_with_delta`)
- Modify: `src/internal/observability/admin_surface.py` (count citeable tools from the registry)
- Modify: `src/internal/tools/base.py` and `registry.py` (remove `stopping`)
- Modify: `src/context/tool_evidence.py` (`ToolSafety = ToolEffect`)
- Modify tests: `tests/unit/test_built_in_tools.py`, and any test passing `stopping=` (`grep -rln "stopping" tests`)

- [ ] **Step 1: Write the failing tests** — replace `tests/unit/test_built_in_tools.py` with:

```python
import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.internal.tools.built_in_tools",
        "src.internal.tools.interface",
        "src.internal.chat.tool_call_args_streaming",
    ],
)
def test_retired_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_stopping_is_retired():
    from src.internal.tools import FunctionTool, Tool

    assert not hasattr(Tool, "stopping")
    with pytest.raises(TypeError):
        FunctionTool(lambda: "x", name="t", stopping=True)


def test_tool_safety_is_tool_effect():
    from src.context import ToolSafety
    from src.internal.tools import ToolEffect

    assert ToolSafety is ToolEffect


def test_admin_surface_counts_citeable_tools_from_the_registry(monkeypatch):
    import src.internal.observability.admin_surface as surface
    from src.internal.tools import FunctionTool, ResultKind, ToolEffect, ToolRegistry

    registry = ToolRegistry()
    registry.register(FunctionTool(lambda: "[]", name="a", effect=ToolEffect.READ_ONLY, result_kind=ResultKind.DOCUMENTS, citeable=True))
    registry.register(FunctionTool(lambda: "{}", name="b", effect=ToolEffect.READ_ONLY, result_kind=ResultKind.JSON))
    monkeypatch.setattr(surface, "tool_registry", registry)
    assert surface.citeable_tool_count() == 1
```

- [ ] **Step 2: Run to verify failure** — `pytest -q tests/unit/test_built_in_tools.py` → FAIL.

- [ ] **Step 3: Implement**

- Delete the three modules. In `llm_step.py` remove exactly the four usages listed above; run `python -c "import src.internal.chat.llm_step"` to confirm it imports.
- `admin_surface.py`: replace the `built_in_tools` imports and the set union at line ~14 with `from src.internal.tools import tool_registry` and

```python
def citeable_tool_count() -> int:
    return sum(1 for t in tool_registry.list_tools() if t.citeable)
```

used where `len(CITEABLE_TOOLS_NAMES)` was; if the line-14 union fed something else, read it first and replace it with the equivalent registry-derived value.
- Remove `stopping` from `Tool`, `FunctionTool.__init__`/property/`from_fn`, and `ToolRegistry.tool`; update tests that pass `stopping=` (drop the argument; if a test asserted stopping behaviour, delete that assertion — the property was never read).
- `tool_evidence.py`: replace the `ToolSafety` class with `ToolSafety = ToolEffect` (import `ToolEffect` from `src.internal.tools.base`), keeping the existing export.

- [ ] **Step 4: Run tests** — `pytest -q tests/unit/test_built_in_tools.py tests/unit/test_tool_categories.py tests/unit/test_llm_agent_generation.py tests/unit/test_run_agentic_search.py tests/unit/test_rag_tool_evidence.py tests/unit/test_rag_pipeline_integration.py` and every test touching `llm_step` (`grep -rln "llm_step" tests`) → all PASS.

- [ ] **Step 5: Mutation-check** — restore `stopping` on `Tool` → `test_stopping_is_retired` FAILS. Restore.

- [ ] **Step 6: Commit** — `git rm` the three modules, stage the rest; commit `tools: retire ChatTool, built_in_tools name sets, stopping and ToolSafety`.

---

### Task 8: Switch on enforcement and prove conformance

**Files:**
- Modify: `src/internal/tools/registry.py` (`tool_registry = ToolRegistry(strict=True)`)
- Modify: `src/internal/memory/tools.py` (`ToolRegistry(strict=True)`)
- Modify: `src/internal/servers/tools/api.py` (`POST /admin/tools/openapi` maps a contract `ValueError` to 422)
- Modify: `src/internal/tools/mcp_client.py` (a remote tool that fails the contract is skipped with a warning, not fatal)
- Modify test fixtures that register non-conforming tools into the global registry: `tests/unit/test_mcp_dynamic_bridge.py`, `tests/unit/servers/web/test_tool_admin_api.py`, `tests/unit/servers/web/test_debug_tools.py`
- Test: `tests/unit/test_tool_conformance.py` (append)

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_the_global_registry_is_strict():
    from src.internal.tools import FunctionTool, ToolEffect
    from src.internal.tools.registry import tool_registry

    with pytest.raises(ValueError, match="result_kind"):
        tool_registry.register(FunctionTool(lambda: "x", name="_undeclared", effect=ToolEffect.READ_ONLY))
    assert tool_registry.get("_undeclared") is None


def test_every_production_tool_registers_strictly():
    from unittest.mock import MagicMock

    from src.internal.memory.tools import build_memory_registry
    from src.internal.tools import ToolRegistry
    from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
    from src.internal.tools.routing_tools import build_search_routing_tool

    registry = ToolRegistry(strict=True)
    seed_tools(registry, tools=tool_knowledge_base(llm=MagicMock()))
    registry.register(build_search_routing_tool(search_url="http://x", top_k=5, name="search_bound"))
    build_memory_registry(_Store(), "u1")  # strict internally; raises if a memory tool breaks the contract
```

Plus: an OpenAPI registration of a spec whose operation would violate the contract cannot happen (effect is derived and `result_kind` is fixed), so instead test the admin endpoint maps a `ValueError` from `register_from_openapi` to 422 (monkeypatch `tool_registry.register_from_openapi` to raise `ValueError("tool x: ...")` and assert the response status and detail); and an MCP discovery test where a remote tool is replaced by a non-conforming one via monkeypatching `_build_tool` asserts that tool is skipped, the others register, and a warning is logged. Read the existing tests for both surfaces first (`grep -rln "register_from_openapi\|parse_mcp_servers\|register_mcp" tests`) and follow their harnesses.

- [ ] **Step 2: Run to verify failure** — `pytest -q tests/unit/test_tool_conformance.py -k "global or production or admin or mcp_discovery"` → FAIL.

- [ ] **Step 3: Implement** — flip the two registries to `strict=True`; wrap the MCP `registry.register(...)` in `try/except ValueError as exc: logger.warning("skipping MCP tool %s: %s", name, exc)`; map `ValueError` in the OpenAPI admin endpoint to `HTTPException(422, detail=str(exc))` (read the endpoint's current exception handling first and extend it). Update the three test fixtures to declare `effect=ToolEffect.READ_ONLY` and a `result_kind` on the tools they register globally (and `tool_registry.tool(double, effect=..., result_kind=...)` in `test_tool_admin_api.py`).

- [ ] **Step 4: Run tests** — the conformance file, the three updated fixture files, `tests/unit/servers/web/test_web_experience_app.py` (app startup seeds the strict global registry), then `pytest -q tests/unit` once. Also run the new/changed test files with torch blocked (a `sys.meta_path` finder raising `ImportError` for `torch`, `sentence_transformers`, `transformers`).

- [ ] **Step 5: Mutation-check** — flip the global registry back to non-strict → `test_the_global_registry_is_strict` FAILS; remove `result_kind` from any one seeded tool → `test_every_production_tool_registers_strictly` FAILS. Restore.

- [ ] **Step 6: Commit** — `tools: enforce the tool contract in the global and memory registries`.
