# Memory Relevant Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Above the 20-memory cap, the injected memory preamble is half relevance hits for the current query and half most-recent fill, instead of recency only; at or below the cap nothing changes.

**Architecture:** `memory_preamble` gains `query`/`encoder` and a `_select_relevant` step built on the existing `search_memories`; `resolve_capabilities` forwards them; the Assist path passes the request query and an encoder the web lifespan built once via `maybe_build_encoder` and stored on `app.state.memory_encoder`.

**Tech Stack:** Python 3.11, FastAPI, SQLite via `AgenticSearchStore`, optional e5 encoder (numpy), pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-memory-relevant-recall-design.md`

## Global Constraints

- `MEMORY_RELEVANT_SLOTS = MEMORY_INJECTION_MAX // 2` (10). Above the cap: up to that many relevance hits (rank order from `search_memories`), then fill newest-first with texts not already selected, up to `max_items`; output in chronological order.
- At or below the cap, or with an empty/whitespace `query`, `memory_preamble` calls only `store.get_user_memories` and returns exactly today's block. The existing capabilities test doubles expose only `get_user_memories` and must keep passing unchanged.
- `search_memories` raising inside selection is logged at warning and degrades to the recency block; nothing in the new code can fail a request.
- No new flag. `maybe_build_encoder` is called exactly once per process, in the web lifespan, never per request.
- Tools and Chat surfaces unchanged. No change to `search_memories`, the manual search endpoint, the MCP tool, `src/agents/`.
- Branch `feat/memory-relevant-recall`; never commit to `main`; never bare `git stash`/`git stash pop`.
- `ruff check . --fix && ruff format .` before each commit; pre-commit aborts if it reformats; re-add and re-commit.
- Every new test gets a mutation check (remove the behavior, watch it go red, restore, `git diff` clean before committing).

---

### Task 1: Relevance-aware `memory_preamble`

**Files:**
- Modify: `src/internal/memory/service.py` (constants near line 24; `memory_preamble` at ~line 35)
- Modify: `tests/unit/memory/test_injection.py`

**Interfaces:**
- Consumes: `search_memories(store, user_id, query, max_results=5, encoder=None) -> list[tuple[UserMemoryRecord, float]]` (same file), `store.get_user_memories(user_id) -> list[str]` (chronological), `Encoder = Callable[[list[str]], Any]`.
- Produces:
  - `MEMORY_RELEVANT_SLOTS: int = 10`
  - `memory_preamble(store, user_id, *, max_items=MEMORY_INJECTION_MAX, query: str | None = None, encoder: Encoder | None = None) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/memory/test_injection.py` (add `import numpy as np` and `from src.internal.memory import service` at the top if not present; `MEMORY_RELEVANT_SLOTS` joins the existing import from `service`):

```python
def _seed_above_cap(store, n=MEMORY_INJECTION_MAX + 5):
    store.add_user_memory("u1", "User is allergic to peanuts")
    for i in range(1, n):
        store.add_user_memory("u1", f"memory number {i}")


def test_relevant_slots_is_half_the_cap():
    assert MEMORY_RELEVANT_SLOTS == MEMORY_INJECTION_MAX // 2


def test_memory_preamble_below_cap_ignores_query():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "User is allergic to peanuts")
    store.add_user_memory("u1", "User prefers window seats")
    assert memory_preamble(store, "u1", query="thai food") == memory_preamble(store, "u1")
    store.close()


def test_memory_preamble_above_cap_keeps_relevant_old_memory_and_recent_fill():
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)
    pre = memory_preamble(store, "u1", query="thai food with peanuts")
    bullets = [line[2:] for line in pre.split("\n") if line.startswith("- ")]
    assert len(bullets) == MEMORY_INJECTION_MAX
    assert len(set(bullets)) == MEMORY_INJECTION_MAX
    assert bullets[0] == "User is allergic to peanuts"  # oldest, kept, first
    assert bullets[-1] == f"memory number {MEMORY_INJECTION_MAX + 4}"  # newest fill
    assert "memory number 1" not in bullets  # old and irrelevant: dropped
    store.close()


def test_memory_preamble_above_cap_without_query_is_recency():
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)
    for q in (None, "", "   "):
        pre = memory_preamble(store, "u1", query=q)
        assert "User is allergic to peanuts" not in pre
        assert pre.count("\n- ") == MEMORY_INJECTION_MAX
    store.close()


def test_memory_preamble_above_cap_uses_encoder_when_given():
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)
    calls: list = []

    def encoder(texts):
        calls.append(list(texts))
        # One-hot: the peanut passage and the query share axis 0; everything else axis 1.
        return np.array(
            [[1.0, 0.0] if ("peanuts" in t or t.startswith("query:")) else [0.0, 1.0] for t in texts]
        )

    pre = memory_preamble(store, "u1", query="anything", encoder=encoder)
    assert "User is allergic to peanuts" in pre
    assert len(calls) == 2
    assert all(t.startswith("passage: ") for t in calls[0])
    assert calls[1] == ["query: anything"]
    store.close()


def test_memory_preamble_search_failure_degrades_to_recency(monkeypatch):
    store = AgenticSearchStore(":memory:")
    _seed_above_cap(store)

    def boom(*a, **k):
        raise RuntimeError("encoder down")

    monkeypatch.setattr(service, "search_memories", boom)
    pre = memory_preamble(store, "u1", query="thai food with peanuts")
    assert pre.count("\n- ") == MEMORY_INJECTION_MAX
    assert "User is allergic to peanuts" not in pre
    store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/memory/test_injection.py -v`
Expected: `ImportError: cannot import name 'MEMORY_RELEVANT_SLOTS'` (whole file fails at import; that is the RED).

- [ ] **Step 3: Implement**

In `src/internal/memory/service.py`, after `MEMORY_INJECTION_MAX = 20` add:

```python
# Above the cap, this many preamble slots go to relevance hits for the
# current query; the rest stay the most recent memories.
MEMORY_RELEVANT_SLOTS = MEMORY_INJECTION_MAX // 2
```

Replace `memory_preamble`:

```python
def memory_preamble(
    store,
    user_id: str,
    *,
    max_items: int = MEMORY_INJECTION_MAX,
    query: str | None = None,
    encoder: Encoder | None = None,
) -> str:
    """Format the user's active memories as a system-prompt preamble.

    Returns an instructional block (leading blank line included) listing up to
    *max_items* memories, or ``""`` when the user has none. The instructional
    wording is what drives proactive use (e.g. warn about a stored allergy).

    At or below the cap every memory is listed. Above it, and given a
    *query*, half the slots go to the memories most relevant to that query
    (``search_memories``) and the rest to the most recent ones, so an old but
    relevant fact still reaches the model while recent facts stay always
    present. Without a query the most recent *max_items* are listed.
    """
    memories = store.get_user_memories(user_id)
    if not memories:
        return ""
    if len(memories) <= max_items or not (query or "").strip():
        chosen = memories[-max_items:]
    else:
        chosen = _select_relevant(store, user_id, query, memories, max_items, encoder)
    return _MEMORY_PREAMBLE_HEADER + "\n".join(f"- {m}" for m in chosen)


def _select_relevant(
    store,
    user_id: str,
    query: str,
    memories: list[str],
    max_items: int,
    encoder: Encoder | None,
) -> list[str]:
    """Relevance hits first, then newest-first fill, returned in chronological order."""
    try:
        hits = search_memories(
            store,
            user_id,
            query,
            max_results=min(MEMORY_RELEVANT_SLOTS, max_items),
            encoder=encoder,
        )
    except Exception as exc:  # noqa: BLE001 — recall is best-effort
        logger.warning("relevant memory recall failed for %s: %s", user_id, exc)
        return memories[-max_items:]
    selected: list[str] = []
    for record, _score in hits:
        if record.memory_text not in selected:
            selected.append(record.memory_text)
    for text in reversed(memories):
        if len(selected) >= max_items:
            break
        if text not in selected:
            selected.append(text)
    keep = set(selected)
    return [m for m in memories if m in keep]
```

`search_memories` is defined later in the same module; that is fine because the call happens at runtime. The test monkeypatches `service.search_memories`, which works because `_select_relevant` resolves the module global at call time.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/memory/test_injection.py tests/unit/access/test_capabilities.py -v`
Expected: all pass, including the three pre-existing preamble tests and every capabilities test (they never pass a query, so they never reach `_select_relevant`).

- [ ] **Step 5: Mutation check**

In `memory_preamble`, change `if len(memories) <= max_items or not (query or "").strip():` to `if True:`. Expected: `test_memory_preamble_above_cap_keeps_relevant_old_memory_and_recent_fill` and `..._uses_encoder_when_given` FAIL. Restore. In `_select_relevant`, remove the `for text in reversed(memories)` fill loop. Expected: the "keeps relevant" test FAILS on the bullet count. Restore. `git diff` clean.

- [ ] **Step 6: Lint and commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/memory/service.py tests/unit/memory/test_injection.py
git commit -m "feat(memory): relevance-aware preamble above the memory cap

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Thread the query and a once-built encoder through capabilities and the Assist path

**Files:**
- Modify: `src/internal/access/capabilities.py` (`resolve_capabilities` ~line 42)
- Modify: `src/internal/servers/web/app.py` (imports; `resolve_capabilities(auth_user, db)` at ~line 1584; lifespan `_app.state` assignments at ~line 1382)
- Modify: `docs/configuration.md` (row after `AGENTIC_SEARCH_MEMORY_AUTO_CURATE`)
- Modify: `tests/unit/access/test_capabilities.py`
- Modify: `tests/unit/servers/web/test_memory_injection.py`

**Interfaces:**
- Consumes: Task 1's `memory_preamble(..., query=, encoder=)`; `maybe_build_encoder() -> Encoder | None` (`src/internal/memory/service.py`).
- Produces: `resolve_capabilities(user, store, *, query: str | None = None, encoder=None) -> RequestCapabilities`; `app.state.memory_encoder`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/access/test_capabilities.py`:

```python
def test_query_and_encoder_are_forwarded_to_the_preamble(monkeypatch):
    seen: dict = {}

    def fake_preamble(store, user_id, *, query=None, encoder=None, **kw):
        seen.update(user_id=user_id, query=query, encoder=encoder)
        return "pre"

    monkeypatch.setattr("src.internal.access.capabilities.memory_preamble", fake_preamble)
    sentinel = object()
    caps = resolve_capabilities(AuthenticatedUser(id="u1"), _Store(), query="thai", encoder=sentinel)
    assert caps.memory_preamble == "pre"
    assert seen == {"user_id": "u1", "query": "thai", "encoder": sentinel}


def test_anonymous_never_reaches_the_preamble_even_with_a_query(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("preamble must not run for anonymous")

    monkeypatch.setattr("src.internal.access.capabilities.memory_preamble", boom)
    assert resolve_capabilities(None, _Store(), query="thai").memory_preamble == ""
```

Append to `tests/unit/servers/web/test_memory_injection.py`:

```python
def test_assist_passes_query_and_app_state_encoder_to_capabilities(monkeypatch):
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    captured: dict = {}
    monkeypatch.setattr(web_app, "answer_with_retrieval", _capturing_awr(captured))
    seen: dict = {}
    real = web_app.resolve_capabilities

    def recording(user, db, *, query=None, encoder=None):
        seen.update(query=query, encoder=encoder)
        return real(user, db, query=query, encoder=encoder)

    monkeypatch.setattr(web_app, "resolve_capabilities", recording)
    app = create_web_app(SearchExperienceSettings(), store=store)
    sentinel = object()
    app.state.memory_encoder = sentinel
    client = TestClient(app)
    client.cookies.set("fastapiusersauth", generate_user_jwt_token(user_id="u1"))
    client.post("/api/agent", json={"query": "Recommend Thai food", "mode": "chat_once"})
    assert seen == {"query": "Recommend Thai food", "encoder": sentinel}


def test_lifespan_builds_the_memory_encoder_once(monkeypatch):
    calls: list = []
    sentinel = object()

    def fake_build():
        calls.append(1)
        return sentinel

    monkeypatch.setattr(web_app, "maybe_build_encoder", fake_build)
    app = create_web_app(SearchExperienceSettings(), store=AgenticSearchStore(":memory:"))
    with TestClient(app):
        assert app.state.memory_encoder is sentinel
    assert calls == [1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/access/test_capabilities.py tests/unit/servers/web/test_memory_injection.py -v -k "forwarded or anonymous_never or app_state_encoder or lifespan_builds"`
Expected: `TypeError: resolve_capabilities() got an unexpected keyword argument 'query'`; `AttributeError: module ... has no attribute 'maybe_build_encoder'`.

- [ ] **Step 3: Implement**

`src/internal/access/capabilities.py`:

```python
def resolve_capabilities(
    user, store, *, query: str | None = None, encoder=None
) -> RequestCapabilities:
```

and in the body `preamble = memory_preamble(store, user_id, query=query, encoder=encoder)`. Extend the docstring: "``query`` and ``encoder`` let the preamble favor memories relevant to the current request once the user is above the injection cap; both are optional and the anonymous short-circuit runs before either is used."

`src/internal/servers/web/app.py`:
- Add `from src.internal.memory.service import maybe_build_encoder` to the module-level imports (the lifespan test patches `web_app.maybe_build_encoder`, so it must be a module attribute).
- In the lifespan, directly after `_app.state.search_agent_tokenizer = None`, add:

```python
        # One encoder per process for relevance-aware memory recall. None
        # unless AGENTIC_SEARCH_MEMORY_SEMANTIC is set; the lexical fallback
        # needs nothing. Never built per request.
        _app.state.memory_encoder = maybe_build_encoder()
```

- In `_run_agent_impl`, replace `capabilities = resolve_capabilities(auth_user, db)` with:

```python
        capabilities = resolve_capabilities(
            auth_user,
            db,
            query=query,
            encoder=getattr(http_request.app.state, "memory_encoder", None),
        )
```

`docs/configuration.md`, directly after the `AGENTIC_SEARCH_MEMORY_AUTO_CURATE` row:

```markdown
| `AGENTIC_SEARCH_MEMORY_SEMANTIC` | `false` | Use the e5 encoder for memory search instead of token overlap: `/api/memory/search`, the MCP `search_memories` tool, and the per-request memory preamble, which above the 20-memory cap fills half its slots with the memories most relevant to the current query. The web app loads the model once at startup; `AGENTIC_SEARCH_MEMORY_EMBED_DEVICE` (default `cpu`) picks the device |
```

(`AGENTIC_SEARCH_MEMORY_SEMANTIC` and `AGENTIC_SEARCH_MEMORY_EMBED_DEVICE` are not documented anywhere in `docs/` today; this row is their first.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/access/test_capabilities.py tests/unit/servers/web/test_memory_injection.py tests/unit/memory -v`
Expected: all pass.

- [ ] **Step 5: Mutation check**

Drop `query=query` from the `resolve_capabilities` call in `_run_agent_impl`. Expected: `test_assist_passes_query_and_app_state_encoder_to_capabilities` FAILS. Restore. Remove the `_app.state.memory_encoder = maybe_build_encoder()` line. Expected: `test_lifespan_builds_the_memory_encoder_once` FAILS (AttributeError or `calls == []`). Restore. `git diff` clean.

- [ ] **Step 6: Lint, full suite, import guard, commit**

```bash
ruff check . --fix && ruff format .
pytest -q
python -c "import src.internal.servers.web.app, src.internal.access.capabilities, src.internal.memory.service"
git add src/internal/access/capabilities.py src/internal/servers/web/app.py docs/configuration.md tests/unit/access/test_capabilities.py tests/unit/servers/web/test_memory_injection.py
git commit -m "feat(web): pass the request query and a once-built encoder into memory recall

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Done criteria

- `pytest -q` green; `ruff check .` clean.
- `grep -rn "maybe_build_encoder()" src/` shows the web lifespan call plus the two pre-existing per-call sites (router, MCP tool) and nothing on a request path.
- Push `feat/memory-relevant-recall`; open a PR titled "feat(memory): relevance-aware memory recall above the injection cap" linking the spec and this plan.
