# Hierarchical Agent Memory + Lifecycle Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the per-user long-term memory in `src/internal/memory/` to a categorized hierarchy with hybrid BM25+dense search, in-process lifecycle hooks for automatic recall and capture, and LLM-backed consolidation. Every `/api/agent` route gets the memory.

**Architecture:**
- **Categories and index:** memories get a fixed category, a cached embedding, and per-category plus root summary nodes in SQLite.
- **Retrieval and recall:** a pure-Python BM25 leg is fused with a cached-e5 leg through the existing `rrf_rank`. A hierarchical preamble is built from it.
- **Lifecycle hooks:** a small registry (`before_turn` / `after_turn` / `after_tool_call`) drives recall, per-turn capture, and the consolidation trigger from `_run_agent_impl`.

**Tech Stack:** Python 3, sqlite3, numpy (optional dense leg), FastAPI, pytest (asyncio_mode=auto). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-05-hierarchical-agent-memory-design.md`

## Global Constraints

- **Memory never fails or slows a request.** Every recall, capture and consolidation call site catches, logs and degrades.
- **No cross-user or anonymous leakage:**
  - every store query is scoped by `user_id`
  - capture requires `user_id is not None` and session ownership via `service._readable`
- **Redaction:** every write goes through `AgenticSearchStore` methods, which run `deterministic_capture`.
- **Torch-free tests:**
  - tests use `AgenticSearchStore(":memory:")`, fake numpy encoders, and scripted fake LLMs
  - `tests/unit/test_no_live_llm_in_tests.py` and `tests/unit/test_no_blocking_calls_in_async.py` stay green
  - sync store/LLM calls from async code go through `asyncio.to_thread`
- **Optional dependencies:** `llm is None` and `encoder is None` are both supported everywhere. Without them the code uses the lexical/deterministic fallback.
- **Fixed categories:** `("constraints", "preferences", "profile", "projects", "decisions", "general")`. The default is `"general"`.
- **Auto capture** is gated by the existing `settings.memory_auto_curate` (`AGENTIC_SEARCH_MEMORY_AUTO_CURATE`), default off.
- **Auto consolidation** threshold: env `AGENTIC_SEARCH_MEMORY_CONSOLIDATE_EVERY`, default `10`.
- **Constraints are protected.** Automatic consolidation never soft-deletes, supersedes or reclassifies a `constraints` memory. Exact-duplicate collapse is allowed.
- **Pinned versions:** test against `requirements-unit-test.txt` versions.
- **Lint and format:** `ruff check . --fix && ruff format .` before each commit.

## Review Focus

1. **A pre-upgrade file DB** (existing `user_memories` rows, no new columns) opens without error. Its rows are backfilled by the classifier. → Task 1, `test_migrates_legacy_db_and_backfills_categories`.
2. **A memory edited after its embedding was cached** must not be ranked by its stale vector. → Task 2, `test_update_invalidates_cached_embedding`.
3. **An encoder swap** (a different model identity) recomputes vectors rather than mixing spaces. → Task 2, `test_encoder_identity_change_recomputes`.
4. **The LLM returns junk** (non-JSON, unknown category, an id that doesn't exist) during capture or consolidation. Nothing is written for that item, and no exception escapes. → Task 7, `test_capture_ignores_unparseable_llm_output`; Task 8, `test_llm_consolidation_ignores_unknown_ids`.
5. **Two turns from one user finishing back-to-back** must not run capture and consolidation concurrently for that user. → Task 7, `test_user_guard_skips_when_busy`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/internal/memory/categories.py` (new) | `CATEGORIES`, `DEFAULT_CATEGORY`, `classify_memory(text)`, `normalize_category(value)` |
| `src/internal/db/store.py` (modify) | new columns/table, migration and backfill, category/embedding/index-node accessors |
| `src/internal/db/models.py` (modify) | `UserMemoryRecord.category`, new `MemoryIndexNodeRecord` |
| `src/internal/memory/search.py` (new) | BM25, embedding cache, RRF fusion, `hybrid_search` |
| `src/internal/memory/service.py` (modify) | `search_memories` delegates; hierarchical `memory_preamble`; `consolidate_memories` delegates |
| `src/internal/memory/consolidation.py` (new) | near-dup merge, contradictions, reclassify, summaries |
| `src/internal/memory/capture.py` (new) | gate, extraction, reconcile |
| `src/internal/memory/lifecycle.py` (new) | memory's hook handlers plus the per-user guard |
| `src/internal/hooks/lifecycle.py` (new) | generic `LifecycleHooks` registry and context dataclasses |
| `src/internal/memory/agent_tool.py` (new) | request-bound read-only `search_memory` Tool |
| `src/internal/memory/router.py` (modify) | `/index`, category params, app encoder, richer consolidate |
| `src/internal/access/capabilities.py` (modify) | `include_memory` kwarg |
| `src/agents/tool/tool_calling.py` (modify) | `on_tool_call` callback |
| `src/internal/servers/web/app.py`, `tool_agent_runner.py` (modify) | emission points; SEARCH/TOOL memory wiring |
| `docs/api-reference.md`, `docs/chat-engine.md` (modify) | docs |

---

### Task 1: Categories, schema migration, store accessors

**Files:**
- Create: `src/internal/memory/categories.py`
- Modify: `src/internal/db/models.py:26-35`, `src/internal/db/store.py` (schema around `:466` and `:534`, `_row_to_user_memory` `:1213`, memory accessors `:2084-2220`)
- Test: `tests/unit/memory/test_categories.py`, `tests/unit/db/test_memory_hierarchy_store.py`

**Interfaces:**
- Produces:
  - `CATEGORIES: tuple[str, ...]`
  - `DEFAULT_CATEGORY = "general"`
  - `classify_memory(text: str) -> str`
  - `normalize_category(value: object) -> str` (unknown or empty becomes `"general"`)
  - `UserMemoryRecord.category: str = "general"`
  - `@dataclass(slots=True) MemoryIndexNodeRecord(user_id: str, node: str, summary: str, memory_count: int, dirty: bool, updated_at: str | None)`
  - `ROOT_NODE = "__root__"` (in `categories.py`)
  - Store methods:
    - `add_user_memory(user_id, memory_text, metadata=None, *, category: str | None = None)`: `None` means `classify_memory`
    - `update_user_memory(user_id, memory_id, new_text, metadata=None, *, category: str | None = None)`: `None` keeps the existing category; clears the embedding
    - `set_user_memory_category(user_id, memory_id, category) -> bool`
    - `get_user_memory_records(user_id, *, category: str | None = None)`
    - `get_memory_embeddings(user_id) -> dict[str, tuple[bytes, str]]` (id → (blob, model))
    - `set_memory_embeddings(user_id, rows: list[tuple[str, bytes, str]]) -> None`
    - `get_memory_index_nodes(user_id) -> list[MemoryIndexNodeRecord]`
    - `upsert_memory_index_node(user_id, node, *, summary, memory_count) -> None` (sets `dirty=0`)
    - `mark_memory_nodes_dirty(user_id, categories: Iterable[str]) -> None` (always includes `ROOT_NODE`)
  - Every add, update, delete and category change marks the affected category nodes plus root dirty.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/memory/test_categories.py
import pytest
from src.internal.memory.categories import CATEGORIES, classify_memory, normalize_category

@pytest.mark.parametrize("text,expected", [
    ("User is allergic to peanuts", "constraints"),
    ("Never suggest paywalled sources", "constraints"),
    ("Prefers peer-reviewed papers", "preferences"),
    ("Decided to go with FAISS over Annoy", "decisions"),
    ("Is researching hybrid retrieval for a thesis project", "projects"),
    ("I'm a data engineer living in Berlin", "profile"),
    ("The sky was cloudy", "general"),
])
def test_classify_memory(text, expected):
    assert classify_memory(text) == expected

def test_normalize_category():
    assert normalize_category("Preferences ") == "preferences"
    assert normalize_category("bogus") == "general"
    assert normalize_category(None) == "general"
    assert set(CATEGORIES) == {"constraints","preferences","profile","projects","decisions","general"}
```

```python
# tests/unit/db/test_memory_hierarchy_store.py
def test_add_classifies_and_marks_dirty():
    s = AgenticSearchStore(":memory:")
    r = s.add_user_memory("u1", "User is allergic to peanuts")
    assert r.category == "constraints"
    dirty = {n.node for n in s.get_memory_index_nodes("u1") if n.dirty}
    assert dirty == {"constraints", "__root__"}

def test_explicit_category_and_filter():
    s = AgenticSearchStore(":memory:")
    s.add_user_memory("u1", "likes tea", category="profile")
    assert [r.category for r in s.get_user_memory_records("u1", category="profile")] == ["profile"]
    assert s.get_user_memory_records("u1", category="preferences") == []

def test_update_keeps_category_and_clears_embedding():
    s = AgenticSearchStore(":memory:")
    r = s.add_user_memory("u1", "Prefers arXiv")
    s.set_memory_embeddings("u1", [(r.id, b"\x00" * 8, "m1")])
    s.update_user_memory("u1", r.id, "Prefers arXiv and ACL anthology")
    assert s.get_user_memory_records("u1")[0].category == "preferences"
    assert s.get_memory_embeddings("u1") == {}

def test_embeddings_and_nodes_scoped_by_user():
    s = AgenticSearchStore(":memory:")
    r = s.add_user_memory("u1", "x")
    s.set_memory_embeddings("u2", [(r.id, b"\x00" * 8, "m1")])  # wrong user: ignored
    assert s.get_memory_embeddings("u1") == {}
    assert s.get_memory_index_nodes("u2") == []

def test_upsert_index_node_clears_dirty():
    s = AgenticSearchStore(":memory:")
    s.add_user_memory("u1", "User is allergic to peanuts")
    s.upsert_memory_index_node("u1", "constraints", summary="allergies", memory_count=1)
    node = {n.node: n for n in s.get_memory_index_nodes("u1")}["constraints"]
    assert (node.summary, node.memory_count, node.dirty) == ("allergies", 1, False)

def test_migrates_legacy_db_and_backfills_categories(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript("""CREATE TABLE user_memories (id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        memory_text TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        INSERT INTO user_memories VALUES ('mem_1','u1','User is allergic to peanuts','{}',1,'t','t');""")
    conn.commit(); conn.close()
    s = AgenticSearchStore(str(path))
    assert s.get_user_memory_records("u1")[0].category == "constraints"
```

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/unit/memory/test_categories.py tests/unit/db/test_memory_hierarchy_store.py -v`
Expected: FAIL (ImportError / TypeError on `category`).

- [ ] **Step 3: Implement `categories.py`**

Use ordered keyword rules: the first category whose word-boundary regex matches wins, in the order `constraints, decisions, preferences, projects, profile`, falling back to `general`. Include the spec's keyword lists (allergy/allergic/intolerant/vegan/never/must not/can't eat; prefer/like/hate/favorite; decided/chose/going with; working on/researching/project/goal; I am/I'm a/I live/my job/is a ... engineer|researcher|student). Matching is case-insensitive. The test table above defines the behavior.

- [ ] **Step 4: Implement the store changes**

- **Schema:** add `category TEXT NOT NULL DEFAULT 'general'`, `embedding BLOB` and `embedding_model TEXT` to `user_memories` in both `_init_schema` and `_migrate_schema`. Use the existing `try: ALTER ... except sqlite3.OperationalError: pass` pattern at `:520-525`.
- **New table:** `memory_index_nodes(user_id TEXT NOT NULL, node TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', memory_count INTEGER NOT NULL DEFAULT 0, dirty INTEGER NOT NULL DEFAULT 1, updated_at TEXT, PRIMARY KEY(user_id, node))`.
- **Backfill:** in `_migrate_schema`, run `classify_memory` once over rows whose category is `'general'` and whose `metadata_json` lacks `"category_backfilled"`. Set the category, and set `metadata.category_backfilled=true` so a reopen doesn't redo the work.
- **Accessors:** every new accessor runs under the store's existing lock decorator or pattern, as the neighbouring methods do. `set_memory_embeddings` updates `WHERE id=? AND user_id=?`.
- **Records:** `_row_to_user_memory` reads `category`.

- [ ] **Step 5: Run the new and existing memory/store tests**

Run: `pytest tests/unit/memory tests/unit/db -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/internal/memory/categories.py src/internal/db/store.py src/internal/db/models.py tests/unit/memory/test_categories.py tests/unit/db/test_memory_hierarchy_store.py
git commit -m "feat(memory): categories, embedding cache columns and index nodes in the store"
```

---

### Task 2: Memory grep: BM25 + cached dense + RRF

**Files:**
- Create: `src/internal/memory/search.py`
- Modify: `src/internal/memory/service.py:160-188` (`search_memories` delegates), `src/internal/memory/router.py` (`/search`)
- Test: `tests/unit/memory/test_memory_search.py`, extend `tests/unit/servers/web/test_memory_router.py`

**Interfaces:**
- Consumes: Task 1 store methods; `src.internal.retrieval.fusion.rrf_rank`.
- Produces:
  - `bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]], *, k1: float = 1.2, b: float = 0.75) -> list[float]`
  - `encoder_identity(encoder) -> str`: returns `getattr(encoder, "model_id", None)` or `f"{module}.{qualname}"` of the callable
  - `ensure_embeddings(store, user_id, records, encoder) -> dict[str, np.ndarray]`: computes only missing or stale vectors in one batch with `passage: ` prefixes, writes them back, returns id → vector
  - `hybrid_search(store, user_id, query, *, max_results=5, encoder=None, category=None) -> list[tuple[UserMemoryRecord, float]]`
  - `service.search_memories(store, user_id, query, max_results=5, encoder=None, *, category=None)` returns `hybrid_search(...)`
- `build_e5_encoder` (`src/internal/servers/retrieval/hybrid.py:89`): set `encode.model_id = model_name` on the returned function so identity is stable.

- [ ] **Step 1: Write failing tests**

```python
def test_bm25_prefers_rarer_term_and_zero_for_no_overlap():
    s = bm25_scores(["peanut"], [["peanut","allergy"], ["thai","food"], ["peanut","butter","peanut"]])
    assert s[1] == 0.0 and s[0] > 0 and s[2] > 0

def test_lexical_only_ranks_and_filters_zero():
    store = _store_with(["User is allergic to peanuts", "Prefers arXiv", "Lives in Berlin"])
    hits = hybrid_search(store, "u1", "peanut allergy")
    assert [r.memory_text for r, _ in hits] == ["User is allergic to peanuts"]

def test_category_filter():
    hits = hybrid_search(store, "u1", "arXiv", category="constraints")
    assert hits == []

def test_dense_leg_fuses_and_caches(fake_encoder):  # fake counts calls, one-hot on "peanut"/query
    hybrid_search(store, "u1", "nuts", encoder=fake_encoder)
    hybrid_search(store, "u1", "nuts", encoder=fake_encoder)
    passage_calls = [c for c in fake_encoder.calls if c[0].startswith("passage: ")]
    assert len(passage_calls) == 1          # vectors cached after first search
    assert len(store.get_memory_embeddings("u1")) == 3

def test_dense_only_match_is_returned(fake_encoder):
    hits = hybrid_search(store, "u1", "nuts", encoder=fake_encoder)   # no lexical overlap
    assert hits[0][0].memory_text == "User is allergic to peanuts"

def test_update_invalidates_cached_embedding(fake_encoder):
    hybrid_search(store, "u1", "x", encoder=fake_encoder)
    rid = store.get_user_memory_records("u1")[1].id
    store.update_user_memory("u1", rid, "Prefers ACL anthology")
    fake_encoder.calls.clear()
    hybrid_search(store, "u1", "x", encoder=fake_encoder)
    assert fake_encoder.calls[0] == ["passage: Prefers ACL anthology"]

def test_encoder_identity_change_recomputes(fake_encoder, other_encoder):
    hybrid_search(store, "u1", "x", encoder=fake_encoder)
    hybrid_search(store, "u1", "x", encoder=other_encoder)
    assert len(other_encoder.calls[0]) == 3   # all passages re-embedded
```

Router test: `POST /api/memory/search` with `{"query": "...", "category": "preferences"}` returns only preferences. With `app.state.memory_encoder` set to a fake, the fake is called and `maybe_build_encoder` is never called (monkeypatch it to raise).

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/unit/memory/test_memory_search.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `search.py`**

- **Tokenizing:** use `service._tokenize` (move it into `search.py` and re-import it in `service`), and drop `service._QUERY_STOPWORDS` from the query.
- **Lexical leg:** the documents with BM25 > 0, in descending order.
- **Dense leg:** cosine over normalized vectors. Keep the top `max(20, 4*max_results)` with sim > 0.
- **Fusion:** `rrf_rank([lex, dense], key_fn=lambda r: r.id)`. Return the fused score. Lexical-only returns the BM25 score.
- **Encoder failure:** if `ensure_embeddings` raises, log it and fall back to lexical-only.
- **Embedding storage:** vectors are stored with `np.asarray(v, dtype=np.float32).tobytes()`. Import numpy lazily inside the dense path.

- [ ] **Step 4: Update the router**

`/search` reads `getattr(request.app.state, "memory_encoder", None)` and accepts `category: str | None = None`.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/unit/memory tests/unit/servers/web/test_memory_router.py -q`
Expected: PASS. Existing `test_service_basic.py` lexical expectations may need a score update only. Ordering assertions must still hold, and if one changes, check it against the BM25 semantics before editing it.

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(memory): hybrid BM25 + cached e5 memory search fused with RRF"
```

---

### Task 3: Hierarchical preamble

**Files:**
- Modify: `src/internal/memory/service.py:39-128`
- Test: `tests/unit/memory/test_injection.py` (rewrite `:128-135`), new `tests/unit/memory/test_hierarchical_preamble.py`

**Interfaces:**
- Produces: `memory_preamble(store, user_id, *, max_items=20, query=None, encoder=None) -> str` (same signature).
  - It still begins with `_MEMORY_PREAMBLE_HEADER`.
  - Bullets stay `"- {text}"` lines, grouped under `"[{category}]"` heading lines (with `": {summary}"` appended when a non-empty node summary exists).
  - A root line, `"Memory index: {root summary or 'constraints 2 · preferences 5 · ...'}"`, follows the header.
  - Total output is capped at `MEMORY_PREAMBLE_CHAR_BUDGET = 4000` chars. Bullets are trimmed from the recency fill first, never from constraints.

- [ ] **Step 1: Write failing tests**

```python
def test_constraints_always_present_above_cap_without_query():
    store = AgenticSearchStore(":memory:"); _seed_above_cap(store)   # existing helper; peanut is oldest
    for q in (None, "", "quantum computing"):
        assert "User is allergic to peanuts" in memory_preamble(store, "u1", query=q)

def test_grouped_by_category_with_root_line():
    store = _store_with(["User is allergic to peanuts", "Prefers arXiv"])
    pre = memory_preamble(store, "u1")
    assert "Memory index:" in pre
    assert pre.index("[constraints]") < pre.index("- User is allergic to peanuts")
    assert "[preferences]" in pre

def test_node_summary_shown_in_heading():
    store.upsert_memory_index_node("u1", "preferences", summary="likes primary sources", memory_count=1)
    assert "[preferences]: likes primary sources" in memory_preamble(store, "u1")

def test_deterministic_and_bounded():
    store = _store_with([f"fact {i} " + "x" * 300 for i in range(40)])
    a = memory_preamble(store, "u1", query="fact"); b = memory_preamble(store, "u1", query="fact")
    assert a == b and len(a) <= 4000 + len(_MEMORY_PREAMBLE_HEADER)

def test_empty_user_is_empty_string():
    assert memory_preamble(AgenticSearchStore(":memory:"), "nobody") == ""
```

Rewrite `test_memory_preamble_above_cap_without_query_is_recency` so that the allergy **is** present, and the remaining `MEMORY_INJECTION_MAX - 1` bullets are the newest. Update the other `test_injection.py` assertions that count `"\n- "` or index bullets only where grouping changes their position. Keep each test's intent.

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/unit/memory/test_hierarchical_preamble.py tests/unit/memory/test_injection.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

Selection runs in this order, de-duplicated by text:
1. all `constraints`
2. `hybrid_search` hits (up to `MEMORY_RELEVANT_SLOTS`)
3. newest fill, up to `max_items`

At or below the cap, everything is selected. Render categories in `CATEGORIES` order, and memories chronologically within each. A search failure degrades to constraints + recency, as `_select_relevant` does today.

- [ ] **Step 4: Run the memory and web memory tests**

Run: `pytest tests/unit/memory tests/unit/servers/web/test_memory_injection.py tests/unit/access -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(memory): hierarchical, constraint-pinned memory preamble"
```

---

### Task 4: Lifecycle hook registry and `after_tool_call` from ToolAgentLoop

**Files:**
- Create: `src/internal/hooks/lifecycle.py`
- Modify: `src/agents/tool/tool_calling.py:675-683, 777-780` (new `on_tool_call` kwarg)
- Test: `tests/unit/hooks/test_lifecycle.py`, extend `tests/unit/test_on_turn_callback.py` or a new `tests/unit/agents/test_on_tool_call.py`

**Interfaces:**
- Produces:

```python
@dataclass
class TurnContext:
    user_id: str | None
    session_id: str | None
    query: str
    store: Any
    llm: Any = None
    encoder: Any = None
    route: str | None = None
    answer: str | None = None

@dataclass(frozen=True)
class ToolCallContext:
    tool_name: str
    arguments: dict
    result_summary: str      # truncated to 500 chars
    user_id: str | None
    session_id: str | None

BeforeTurnHandler = Callable[[TurnContext], Awaitable[str | None]]
AfterTurnHandler = Callable[[TurnContext], Awaitable[None]]
AfterToolCallHandler = Callable[[ToolCallContext], Awaitable[None]]

class LifecycleHooks:
    def __init__(self, *, timeout_seconds: float = 2.0) -> None: ...
    def on_before_turn(self, h: BeforeTurnHandler) -> None: ...
    def on_after_turn(self, h: AfterTurnHandler) -> None: ...
    def on_after_tool_call(self, h: AfterToolCallHandler) -> None: ...
    async def before_turn(self, ctx: TurnContext) -> str: ...        # joins non-empty fragments with "\n\n"; each handler under timeout
    def after_turn(self, ctx: TurnContext) -> list[asyncio.Task]: ... # schedules each handler as a task, held in a module task set
    async def after_tool_call(self, ctx: ToolCallContext) -> None: ...  # awaited, each under timeout, errors swallowed
```

- `ToolAgentLoop.run(..., on_tool_call: Callable[[str, dict, str], Awaitable[None]] | None = None)` is called once per tool execution result, right after the `on_turn` loop at `:778`, with `(tool_name, arguments, response_text)`. Exceptions it raises are logged and swallowed.

- [ ] **Step 1: Write failing tests**

```python
async def test_before_turn_joins_in_order_and_isolates_errors():
    h = LifecycleHooks()
    async def a(ctx): return "A"
    async def boom(ctx): raise RuntimeError("x")
    async def none(ctx): return None
    async def b(ctx): return "B"
    for f in (a, boom, none, b): h.on_before_turn(f)
    assert await h.before_turn(_ctx()) == "A\n\nB"

async def test_before_turn_timeout_skips_handler():
    h = LifecycleHooks(timeout_seconds=0.05)
    async def slow(ctx): await asyncio.sleep(1); return "late"
    h.on_before_turn(slow)
    assert await h.before_turn(_ctx()) == ""

async def test_after_turn_runs_in_background_and_swallows():
    seen = []
    h = LifecycleHooks()
    async def ok(ctx): seen.append(ctx.answer)
    async def boom(ctx): raise RuntimeError
    h.on_after_turn(ok); h.on_after_turn(boom)
    tasks = h.after_turn(_ctx(answer="done"))
    await asyncio.gather(*tasks, return_exceptions=True)
    assert seen == ["done"]

async def test_after_tool_call_receives_context(): ...  # handler records ctx; assert tool_name/args/summary truncated to 500
```

ToolAgentLoop test: reuse the scripted-tool fixture from `tests/unit/test_on_turn_callback.py`. Pass `on_tool_call` and assert it received `("<tool>", {...}, "<response text>")` once per executed tool.

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/unit/hooks/test_lifecycle.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the registry and the `on_tool_call` plumbing**

The response text is `self._tool_message_content(result)`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/hooks tests/unit/test_on_turn_callback.py tests/unit/test_no_blocking_calls_in_async.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(hooks): in-process lifecycle hook registry; ToolAgentLoop on_tool_call"
```

---

### Task 5: Recall via `before_turn` in `/api/agent`

**Files:**
- Create: `src/internal/memory/lifecycle.py` (recall handler only in this task)
- Modify: `src/internal/access/capabilities.py:42-72` (`include_memory: bool = True`), `src/internal/servers/web/app.py` (`create_web_app` builds `app.state.lifecycle_hooks`; `_run_agent_impl` `:1932-1965`)
- Test: `tests/unit/servers/web/test_memory_injection.py` (extend), `tests/unit/access/test_capabilities.py` (one new test)

**Interfaces:**
- Consumes: `LifecycleHooks`, `TurnContext` (Task 4); `memory_preamble` (Task 3).
- Produces:
  - `async def recall_handler(ctx: TurnContext) -> str | None`: returns `None` when `ctx.user_id is None`. Otherwise runs `asyncio.to_thread(memory_preamble, ctx.store, ctx.user_id, query=ctx.query, encoder=ctx.encoder)`.
  - `def register_memory_hooks(hooks: LifecycleHooks, settings) -> None`: registers recall here. Tasks 7 and 8 add their handlers to it.
  - `resolve_capabilities(..., include_memory=False)` returns `memory_preamble=""` without touching memory.

- [ ] **Step 1: Write failing tests**

- `test_capabilities.py`: `resolve_capabilities(AuthenticatedUser(id="u1"), store, include_memory=False).memory_preamble == ""`, and `memory_preamble` is never called (monkeypatched to raise).
- `test_memory_injection.py`:
  - **Rewritten query:** a QUERY_PROCESSING webhook rewrites the query, and the recall handler sees the rewritten query and a non-None `session_id` (register a recording before_turn hook on `app.state.lifecycle_hooks`).
  - **Signed-in caller:** a signed-in caller's stored memory still reaches `answer_with_retrieval`'s `user_memory` (an existing assertion, kept).
  - **Anonymous caller:** gets `user_memory=None`.
  - **Failing hook:** a raising before_turn hook doesn't fail the request (200, answer present).

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/unit/servers/web/test_memory_injection.py tests/unit/access/test_capabilities.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement the wiring**

In `_run_agent_impl`:
- call `resolve_capabilities(..., include_memory=False)` (drop `query`/`encoder` from that call)
- after `_ensure_session`, build `TurnContext(user_id, session_id, query, store=db, llm=llm, encoder=app.state.memory_encoder)`
- set `user_memory = (await hooks.before_turn(ctx)) or None`
- keep `ctx` in a local variable for Task 7's `after_turn`

`create_web_app` builds `LifecycleHooks()` and calls `register_memory_hooks` before the lifespan.

- [ ] **Step 4: Run the web and memory tests**

Run: `pytest tests/unit/servers/web tests/unit/access tests/unit/memory -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(memory): recall through the before_turn lifecycle hook"
```

---

### Task 6: Memory on the SEARCH and TOOL routes, plus the `search_memory` tool

**Files:**
- Create: `src/internal/memory/agent_tool.py`
- Modify:
  - `src/internal/servers/web/app.py`:
    - `_run_search_agent` `:790-830` gains `user_memory: str | None = None`
    - `_run_search_direct_or_escalate` threads it to `_escalate`
    - `_run_auto_routed` `:1412-1470` passes it to the TOOL and SEARCH branches
    - the explicit-mode call sites near `:2052`, `:2186`, `:2261` and `:2300`
    - remove the comment at `:1471`
  - `src/internal/servers/web/tool_agent_runner.py:153-235`
  - `docs/chat-engine.md`
- Test: `tests/unit/servers/web/test_memory_routes.py`, `tests/unit/memory/test_agent_tool.py`

**Interfaces:**
- Consumes: `hybrid_search` (Task 2).
- Produces:
  - `build_search_memory_tool(store, user_id: str, encoder=None) -> Tool`, a Tool named `"search_memory"`. Its properties:
    - `effect=ToolEffect.READ_ONLY` (or the repo's read-only equivalent), `citeable=False`
    - schema `{query: str (required), category: enum CATEGORIES (optional), max_results: int 1..10 default 5}`
    - returns newline-joined `"[{category}] {text}"`, or `"no matching memories"`
  - `_run_tool_agent(..., user_memory: str | None = None, memory_store=None, memory_user_id: str | None = None, memory_encoder=None)`. It appends `user_memory` to the system message after `TOOL_AGENT_SYSTEM_PROMPT` (and after any working-memory summary). When `memory_store` and `memory_user_id` are set, it prepends `build_search_memory_tool(...)` and drops any tool named `"search_memories"` from the offered list.
  - `_run_search_agent(..., user_memory=None)` builds `system_prompt` whenever `summary or user_memory`, as instruction + `"\n\n"` + summary + `"\n\n"` + user_memory (skipping empties).

- [ ] **Step 1: Write failing tests**

```python
async def test_search_memory_tool_is_user_scoped():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "Prefers arXiv"); store.add_user_memory("u2", "Prefers arXiv mirrors")
    tool = build_search_memory_tool(store, "u1")
    text, *_ = await tool.execute("i", {"query": "arXiv"})
    assert "Prefers arXiv" in text and "mirrors" not in text

async def test_search_memory_tool_category_filter(): ...  # category="constraints" -> "no matching memories"
```

Route tests (monkeypatch `ToolAgentLoop.run` / the search loop class to capture `messages` and `tools`):
- TOOL route: the system message contains a stored memory, the tools include `search_memory`, and the tools exclude a stub `search_memories`.
- SEARCH escalation: the loop's `search_config.system_prompt` contains the memory.
- Anonymous TOOL route: no `search_memory` tool and no memory text.

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/unit/memory/test_agent_tool.py tests/unit/servers/web/test_memory_routes.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the tool and the threading**

`user_id` and `user_memory` from `_run_agent_impl` reach `_run_auto_routed` and the explicit-mode branches. Update `docs/chat-engine.md:20-28` to say every route gets memory.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/servers/web tests/unit/memory -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(memory): inject memory into SEARCH and TOOL routes; request-bound search_memory tool"
```

---

### Task 7: Per-turn automatic capture via `after_turn`

**Files:**
- Create: `src/internal/memory/capture.py`
- Modify: `src/internal/memory/lifecycle.py` (capture handler + guard), `src/internal/servers/web/app.py` (`finally` block `:2394-2404`: call `hooks.after_turn(ctx)` and pass `auto_curate=False` to `schedule_compression` when per-turn capture is on)
- Test: `tests/unit/memory/test_capture.py`, extend `tests/unit/servers/web/test_memory_injection.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 5.
- Produces:
  - `has_durable_signal(user_message: str) -> bool`
  - `extract_memories(llm, user_message: str, answer: str) -> list[tuple[str, str]]`: (text, category). Calls `llm.complete(prompt, max_tokens=400, temperature=0.0)` and parses `{"memories": [...]}` leniently: first `{` to last `}`, unknown category → `general`, anything invalid → `[]`.
  - `NEAR_DUP_MIN_SCORE`: the near-duplicate threshold on the lexical-normalized score. Use token Jaccard ≥ `0.6` on content tokens, or cosine ≥ `0.9` when an encoder is present.
  - `reconcile(store, user_id, items, *, encoder=None) -> dict[str, int]`: `{"added": n, "updated": m}`, adding or updating per the spec and never deleting.
  - `async def capture_turn(store, user_id, session_id, llm, user_message, answer, *, encoder=None) -> dict | None`: checks session ownership via `service._readable(store.get_chat_session(session_id), user_id)`, then gate → `to_thread(extract)` → `to_thread(reconcile)`, then writes `add_memory_trajectory(..., trajectory={"source": "auto_turn", ...})`.
  - `user_guard(user_id) -> contextlib.AbstractContextManager[bool]` in `lifecycle.py`: yields False if that user is already busy.
  - `capture_handler(ctx)` registered by `register_memory_hooks` only when `settings.memory_auto_curate`. It runs only when `ctx.user_id` and `ctx.llm` are set.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("msg,expected", [
    ("I'm allergic to shellfish, find me restaurants", True),
    ("I prefer peer-reviewed sources", True),
    ("We decided to go with Postgres", True),
    ("What is FAISS?", False),
    ("compare dense and sparse retrieval", False),
])
def test_has_durable_signal(msg, expected): ...

async def test_gate_skips_llm():
    llm = _FakeLLM(raise_if_called=True)
    assert await capture_turn(store, "u1", sid, llm, "What is FAISS?", "...") is None

async def test_capture_adds_with_category():
    llm = _FakeLLM('{"memories":[{"text":"User is allergic to shellfish","category":"constraints"}]}')
    await capture_turn(store, "u1", sid, llm, "I'm allergic to shellfish", "ok")
    assert [(r.memory_text, r.category) for r in store.get_user_memory_records("u1")] == [("User is allergic to shellfish","constraints")]
    assert store.list_memory_trajectories("u1")[0].trajectory["source"] == "auto_turn"

async def test_capture_updates_near_duplicate():
    store.add_user_memory("u1", "User prefers peer reviewed sources", category="preferences")
    llm = _FakeLLM('{"memories":[{"text":"User prefers peer-reviewed sources only","category":"preferences"}]}')
    await capture_turn(store, "u1", sid, llm, "I prefer peer-reviewed sources only", "ok")
    recs = store.get_user_memory_records("u1")
    assert len(recs) == 1 and recs[0].memory_text.endswith("only")

async def test_capture_ignores_unparseable_llm_output(): ...   # "not json", '{"memories":[{"text":5}]}' -> no rows, no raise
async def test_no_capture_for_unowned_session(): ...           # session owned by u2 -> None, LLM not called
def test_user_guard_skips_when_busy():
    with user_guard("u1") as a:
        with user_guard("u1") as b:
            assert a is True and b is False
    with user_guard("u1") as c: assert c is True
```

Web test, with the flag on and a fake LLM:
- the after_turn handler runs once per turn
- `schedule_compression` receives `auto_curate=False`
- an anonymous request schedules no capture

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/unit/memory/test_capture.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

The extraction prompt reuses the curation prompt's rules (one contextual sentence, no secrets) and lists `CATEGORIES`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/memory tests/unit/servers/web tests/unit/test_no_blocking_calls_in_async.py tests/unit/test_no_live_llm_in_tests.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(memory): per-turn automatic capture via after_turn hook"
```

---

### Task 8: Consolidation, the auto trigger, index summaries, `/index`, and docs

**Files:**
- Create: `src/internal/memory/consolidation.py`
- Modify:
  - `src/internal/memory/service.py:198-252` (`consolidate_memories` delegates; keep the signature plus kwargs)
  - `src/internal/memory/lifecycle.py` (consolidation trigger)
  - `src/internal/memory/router.py`
  - `docs/api-reference.md`
- Test: `tests/unit/memory/test_consolidation.py`, extend `tests/unit/servers/web/test_memory_router.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 7 (`user_guard`, the near-dup similarity helper; reuse it, don't duplicate it).
- Produces:
  - `consolidate_memories(store, user_id, resolve_conflicts=True, *, llm=None, encoder=None) -> dict`. Report keys: `initial, duplicates_removed, conflicts_resolved, merged, reclassified, summaries_rebuilt, final`.
  - `rebuild_index_summaries(store, user_id, *, llm=None, only_dirty=True) -> int`.
    - Deterministic summary: `"{n} memories; key terms: {top 5 tokens}; latest: {newest 2, each ≤ 80 chars}"`.
    - Root deterministic summary: category counts in `CATEGORIES` order, e.g. `"constraints 2 · preferences 5"`.
  - `CONSOLIDATE_EVERY = int(os.getenv("AGENTIC_SEARCH_MEMORY_CONSOLIDATE_EVERY", "10"))`. The last-consolidated memory count is stored in the root node's `memory_count`.
  - `consolidation_handler(ctx)`: when `active_count - root.memory_count >= CONSOLIDATE_EVERY`, it runs consolidation under `user_guard` in a thread. Otherwise, if dirty nodes exist, it runs `rebuild_index_summaries` only.
  - Router:
    - `GET /api/memory/index` → `{"root": {"summary", "memory_count"}, "categories": [{"name","summary","memory_count","memories":[{"id","text","updated_at"}]}]}`, in `CATEGORIES` order, empty categories omitted.
    - `/save` accepts `category`.
    - `/list` items include `category` and accept `?category=`.
    - `/consolidate` passes `llm` and the app encoder.

- [ ] **Step 1: Write failing tests**

```python
def test_near_duplicates_merge_without_llm_keeps_newest():
    add("User prefers peer reviewed sources"); add("User prefers peer-reviewed sources")
    rep = consolidate_memories(store, "u1")
    assert rep["merged"] == 1 and len(store.get_user_memory_records("u1")) == 1

def test_constraints_never_superseded_or_merged_by_llm():
    add("User is allergic to peanuts"); add("User is allergic to peanuts and tree nuts")
    llm = _FakeLLM(supersede_all=True, merge_all=True)
    consolidate_memories(store, "u1", llm=llm)
    assert len(store.get_user_memory_records("u1", category="constraints")) == 2

def test_exact_duplicate_constraint_collapses():
    add("User is allergic to peanuts"); add("user is allergic to peanuts")
    assert consolidate_memories(store, "u1")["duplicates_removed"] == 1

def test_llm_contradiction_supersedes_and_records_metadata():
    a = add("Lives in Berlin"); b = add("Moved to Munich")       # both profile
    llm = _FakeLLM(json_reply={"superseded": [a.id]})
    rep = consolidate_memories(store, "u1", llm=llm)
    assert [r.memory_text for r in store.get_user_memory_records("u1")] == ["Moved to Munich"]
    assert rep["conflicts_resolved"]

def test_llm_consolidation_ignores_unknown_ids():  # {"superseded": ["mem_nope"]} and junk text -> no deletions, no raise

def test_reclassify_general_with_llm_but_not_out_of_constraints(): ...

def test_rebuild_summaries_deterministic_and_clears_dirty():
    add("Prefers arXiv"); n = rebuild_index_summaries(store, "u1")
    nodes = {x.node: x for x in store.get_memory_index_nodes("u1")}
    assert n >= 2 and not nodes["preferences"].dirty and nodes["__root__"].summary.startswith("preferences 1")

async def test_auto_trigger_threshold(monkeypatch):
    monkeypatch.setattr(consolidation, "CONSOLIDATE_EVERY", 3)
    # 2 memories -> handler rebuilds summaries only; 3 -> consolidate_memories called once
```

Router tests:
- `/index` shape and order
- `/save` with category
- `/list?category=`
- the existing `/consolidate` report keys still present

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/unit/memory/test_consolidation.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

The order is exact dedup → near-dup merge (non-constraints) → contradictions (non-constraints) → reclassify (`general` only) → summaries.

- **LLM contradiction prompt:** one `complete` call per category with ≥ 2 memories, capped at 30 memories per call (newest). It returns `{"superseded": [ids]}`. Only ids from that batch are honored.
- **LLM merge:** one call per cluster, returning `{"text": ...}`. Update the newest member and soft-delete the rest.
- **Superseded rows:** set `metadata.superseded_by` via `update_user_memory` metadata before `delete_user_memory`.
- **No LLM:** without an LLM, apply the existing `tags[0]` newest-wins rule, again skipping constraints.

Document all endpoint changes in `docs/api-reference.md` under the memory section.

- [ ] **Step 4: Run the full suite and lint**

Run: `ruff check . --fix && ruff format . && pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(memory): LLM-backed consolidation, index summaries, auto trigger and /api/memory/index"
```

---

## Final verification

- [ ] `pip install -r requirements-unit-test.txt` (pinned versions) then `pytest -q`. Everything passes.
- [ ] Smoke test:
  1. Start the backend with `AGENTIC_SEARCH_EMBEDDED_RETRIEVAL=demo AGENTIC_SEARCH_WEB_DB_PATH=/tmp/mem.db`.
  2. `POST /api/memory/save {"text":"I'm allergic to peanuts"}` as a signed-in user.
  3. `GET /api/memory/index` shows it under `constraints`.
