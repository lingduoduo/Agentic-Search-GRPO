# Follow-up Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve follow-up turns into standalone retrieval queries with a switch-aware gate, wire the resolved query into the default retrieval paths behind a flag, and measure it on a repaired multi-turn eval.

**Architecture:** A pure function `resolve_follow_up` in `src/internal/search/context.py` decides continuation vs switch from reference cues, message length, and an optional e5 cosine to the previous standalone topic. `_run_agent_impl` calls it once per request when `AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION` is on and threads the result as a new `retrieval_query` kwarg through the auto-routed SEARCH and CHAT paths, the explicit `chat_loop` and `chat_once` modes, `AgenticRAGLoop.run`, `answer_with_retrieval` and `SearchPipeline.run`; the answer prompt keeps the raw message. The eval harness gains `gated`/`gated_cues` conditions, per-corpus slices, a dev/test split with τ tuned on dev, and a success-criteria check that decides the flag default.

**Tech Stack:** Python 3.10+, FastAPI, pytest, numpy, sentence-transformers (e5-base-v2 via `gate_embedder()`, optional).

**Spec:** `docs/superpowers/specs/2026-09-24-follow-up-resolution-design.md`

## Global Constraints

- Routing is untouched: `recognize_intent` keeps receiving the raw message.
- The answer prompt keeps the raw message; only retrieval receives the resolved query.
- With `AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION` off, every call receives exactly what it does today and no `follow_up` metadata appears.
- Rules, in order: `no_history`, `reference` (whole-word it, its, they, them, their, this, that, these, those, one), `fragment` (≤ 4 words), `semantic` (cosine ≥ τ), `switch`. Cue words are not a rule.
- Topic = most recent earlier user message the gate classified standalone, looking back at most 3 user turns; a continuation searches `f"{topic}\n{message}"`.
- `build_retrieval_context` is not modified (it is the eval's frozen `regex` baseline).
- `SearchAgentLoop`, `ToolAgentLoop`, memory recall, hooks, the explicit `search_tool`, `hybrid_search`, `search_agent` and `tool_agent` modes keep the raw query.
- τ is chosen on the dev half only, grid 0.70–0.95 step 0.01, ties → highest τ; every reported number is test-half only.
- Success criteria (test half, all four to flip the default on): (1) `gated` beats `raw` on SciFact follow-up Hit@5 (tfidf) with CI lower bound > 0; (2) `gated` topic-switch carry-over ≤ 0.15; (3) `gated` SciFact follow-up tfidf hits ≥ `concat` hits − 1; (4) median resolve latency ≤ 30 ms.
- Unit tests must pass with torch unimportable; never load e5 in unit tests (patch `gate_embedder` to return `None`).
- `data/` is gitignored; tracked files there are force-added.
- Branch `feat/follow-up-resolution`; never commit to `main`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Run `ruff check --fix` and `ruff format` on touched files before each commit (the pre-commit hook rejects unformatted files and the commit silently does not happen — check `git log -1` after committing).

## Review Focus

1. **A follow-up to a follow-up** ("weather in Tokyo" → "and in Paris?" → "Berlin?"): the topic must stay "weather in Tokyo", never "and in Paris?" and never a chained string — test in Task 1.
2. **A switch that contains a pronoun used generically** ("Proofread this sentence: …") fires `reference`: expected and measured, but a switch without any cue and with low cosine must pass through unchanged even when the previous turn was a continuation — test in Task 1.
3. **The embedder is missing or the cosine returns `None`**: `semantic` is skipped, no crash, and the harness refuses to run `gated` (it would silently equal `gated_cues`) — tests in Tasks 1 and 5.
4. **Flag off**: no resolver call, no `follow_up` key, `retrieval_query=None` reaches every consumer — test in Task 4.
5. **The first turn of a session** (empty history) resolves to the raw message with reason `no_history` and adds `follow_up` metadata with `continuation: false` — test in Task 4.

---

### Task 1: The resolver

**Files:**
- Modify: `src/internal/search/context.py` (append below `build_retrieval_context`)
- Modify: `src/internal/utils/embedding_gate.py` (add `follow_up_cos_min`)
- Test: `tests/unit/search/test_follow_up_resolution.py`

**Interfaces:**
- Produces:
  - `Resolution(query: str, continuation: bool, reason: str)` frozen dataclass, in `src.internal.search.context`;
  - `resolve_follow_up(message: str, history: Iterable[ChatMessage], *, cosine: Callable[[str, str], float | None] | None, tau: float) -> Resolution`;
  - `follow_up_cos_min() -> float` in `src.internal.utils.embedding_gate`, reading `AGENTIC_SEARCH_FOLLOW_UP_COS_MIN`, default `DEFAULT_FOLLOW_UP_COS_MIN = 0.85` (provisional; Task 7 replaces it with the dev-chosen τ).

- [ ] **Step 1: Write the failing tests**

```python
from src.context import ChatMessage
from src.internal.search.context import Resolution, resolve_follow_up
from src.internal.utils.embedding_gate import follow_up_cos_min


def _history(*user_texts: str) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for text in user_texts:
        messages.append(ChatMessage(role="user", content=text))
        messages.append(ChatMessage(role="assistant", content="Here is what I found."))
    return messages


def _no_cosine(message, topic):
    raise AssertionError("cosine must not be consulted")


def test_first_turn_is_standalone():
    assert resolve_follow_up("What is FAISS?", [], cosine=_no_cosine, tau=0.8) == Resolution(
        "What is FAISS?", False, "no_history"
    )


def test_reference_word_makes_a_continuation():
    got = resolve_follow_up(
        "Which index types does it support?",
        _history("What is FAISS?"),
        cosine=_no_cosine,
        tau=0.8,
    )
    assert got == Resolution(
        "What is FAISS?\nWhich index types does it support?", True, "reference"
    )


def test_reference_words_match_whole_words_only():
    # "item" and "Italy" contain "it" but are not references.
    got = resolve_follow_up(
        "Which item ships fastest to Italy by default?",
        _history("What is FAISS?"),
        cosine=None,
        tau=0.8,
    )
    assert got.reason == "switch"


def test_short_fragment_makes_a_continuation():
    got = resolve_follow_up(
        "And cardiac muscle?", _history("Tirasemtiv targets fast-twitch muscle."), cosine=_no_cosine, tau=0.8
    )
    assert got.continuation and got.reason == "fragment"


def test_cue_word_alone_does_not_make_a_continuation():
    got = resolve_follow_up(
        "Also, what's the weather in Oslo right now?",
        _history("How does dense retrieval with FAISS work?"),
        cosine=lambda message, topic: 0.1,
        tau=0.8,
    )
    assert got == Resolution("Also, what's the weather in Oslo right now?", False, "switch")


def test_semantic_similarity_makes_a_continuation():
    got = resolve_follow_up(
        "Which compounds activate the dephosphorylated form?",
        _history("AMPK activation reduces fibrosis in the lungs."),
        cosine=lambda message, topic: 0.9,
        tau=0.8,
    )
    assert got.continuation and got.reason == "semantic"


def test_below_threshold_is_a_switch():
    got = resolve_follow_up(
        "Which compounds activate the dephosphorylated form?",
        _history("AMPK activation reduces fibrosis in the lungs."),
        cosine=lambda message, topic: 0.79,
        tau=0.8,
    )
    assert got.reason == "switch"


def test_missing_embedder_or_failed_cosine_skips_semantic():
    for cosine in (None, lambda message, topic: None):
        got = resolve_follow_up(
            "Which compounds activate the dephosphorylated form?",
            _history("AMPK activation reduces fibrosis in the lungs."),
            cosine=cosine,
            tau=0.8,
        )
        assert got.reason == "switch"


def test_topic_skips_earlier_continuations():
    got = resolve_follow_up(
        "Berlin?",
        _history("What's the weather in Tokyo?", "And in Paris?"),
        cosine=None,
        tau=0.8,
    )
    assert got.query == "What's the weather in Tokyo?\nBerlin?"


def test_topic_moves_to_the_latest_switch():
    got = resolve_follow_up(
        "Is it open late?",
        _history("What is FAISS?", "Find coffee shops near Union Square, San Francisco."),
        cosine=lambda message, topic: 0.1,
        tau=0.8,
    )
    assert got.query == "Find coffee shops near Union Square, San Francisco.\nIs it open late?"


def test_switch_after_a_continuation_passes_through():
    got = resolve_follow_up(
        "Write a haiku about autumn leaves falling slowly.",
        _history("What is RAG?", "Where does reranking fit in that pipeline?"),
        cosine=lambda message, topic: 0.2,
        tau=0.8,
    )
    assert got == Resolution("Write a haiku about autumn leaves falling slowly.", False, "switch")


def test_topic_looks_back_at_most_three_user_turns():
    history = _history(
        "What is FAISS?",
        "Does it use GPUs?",
        "Is it fast?",
        "What about its memory use?",
    )
    got = resolve_follow_up("Is it open source?", history, cosine=None, tau=0.8)
    # Every later turn continues "What is FAISS?", but it has left the 3-turn
    # window, so the oldest turn inside the window becomes the topic.
    assert got.query == "Does it use GPUs?\nIs it open source?"


def test_assistant_and_tool_markup_messages_are_ignored():
    history = [
        ChatMessage(role="user", content="What is FAISS?"),
        ChatMessage(role="assistant", content="<tool_call>search</tool_call> Also, how about BM25?"),
    ]
    got = resolve_follow_up("Does it use GPUs?", history, cosine=None, tau=0.8)
    assert got.query == "What is FAISS?\nDoes it use GPUs?"


def test_follow_up_cos_min_reads_the_environment(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_FOLLOW_UP_COS_MIN", "0.9")
    assert follow_up_cos_min() == 0.9
    monkeypatch.delenv("AGENTIC_SEARCH_FOLLOW_UP_COS_MIN")
    assert 0.7 <= follow_up_cos_min() <= 0.95
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/search/test_follow_up_resolution.py -q`
Expected: collection error, `ImportError: cannot import name 'Resolution'`.

- [ ] **Step 3: Implement**

In `src/internal/utils/embedding_gate.py`, below `search_direct_cos_min`:

```python
# Provisional; replaced by the τ the multi-turn eval chooses on its dev half.
DEFAULT_FOLLOW_UP_COS_MIN = 0.85


def follow_up_cos_min() -> float:
    return float(
        os.environ.get(
            "AGENTIC_SEARCH_FOLLOW_UP_COS_MIN", str(DEFAULT_FOLLOW_UP_COS_MIN)
        )
    )
```

In `src/internal/search/context.py`, add `from typing import Callable` to the `typing` import and append:

```python
_REFERENCE = re.compile(
    r"\b(?:it|its|they|them|their|this|that|these|those|one)\b", re.IGNORECASE
)
FRAGMENT_MAX_WORDS = 4
TOPIC_LOOKBACK = 3

Cosine = Callable[[str, str], "float | None"]


@dataclass(frozen=True)
class Resolution:
    """What retrieval searches for, and why."""

    query: str
    continuation: bool
    reason: str  # no_history | reference | fragment | semantic | switch


def _continues(message: str, topic: str, cosine: Cosine | None, tau: float) -> tuple[bool, str]:
    if _REFERENCE.search(message):
        return True, "reference"
    if len(message.split()) <= FRAGMENT_MAX_WORDS:
        return True, "fragment"
    if cosine is not None:
        score = cosine(message, topic)
        if score is not None and score >= tau:
            return True, "semantic"
    return False, "switch"


def resolve_follow_up(
    message: str,
    history: Iterable[ChatMessage],
    *,
    cosine: Cosine | None,
    tau: float,
) -> Resolution:
    """Resolve a follow-up into a standalone retrieval query, without an LLM.

    Cue words ("also", "and", "how about") are deliberately not a signal: alone
    they misfire on topic switches. The topic is the latest earlier user message
    this gate itself judged standalone, within the last TOPIC_LOOKBACK user turns,
    so a chain of follow-ups keeps the original topic and never grows.
    """
    user_texts = [
        m.content
        for m in _safe_history(history)
        if m.role.lower() == "user" and m.content.strip()
    ][-TOPIC_LOOKBACK:]
    topic: str | None = None
    for text in user_texts:
        if topic is None or not _continues(text, topic, cosine, tau)[0]:
            topic = text
    if topic is None:
        return Resolution(message, False, "no_history")
    continues, reason = _continues(message, topic, cosine, tau)
    if continues:
        return Resolution(f"{topic}\n{message}", True, reason)
    return Resolution(message, False, reason)
```

Note `test_assistant_and_tool_markup_messages_are_ignored` passes because only user messages are read; `_safe_history` also drops tool-markup assistant messages.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/search/test_follow_up_resolution.py tests/unit/search/test_context.py -q`
Expected: all PASS (the existing `build_retrieval_context` tests are untouched).

- [ ] **Step 5: Mutation-check**

(a) Remove the `\b` word boundaries from `_REFERENCE` → `test_reference_words_match_whole_words_only` must FAIL. (b) Change the topic loop to `topic = text` unconditionally → `test_topic_skips_earlier_continuations` must FAIL. (c) Delete `[-TOPIC_LOOKBACK:]` → `test_topic_looks_back_at_most_three_user_turns` must FAIL. (d) Change `score >= tau` to `score > tau - 0.1` → `test_below_threshold_is_a_switch` must FAIL. Revert each, delete `__pycache__` for the module, re-run, PASS.

- [ ] **Step 6: Commit**

```bash
git add src/internal/search/context.py src/internal/utils/embedding_gate.py tests/unit/search/test_follow_up_resolution.py docs/superpowers/plans/2026-09-24-follow-up-resolution.md
git commit -m "search: switch-aware follow-up resolver for retrieval queries"
```

---

### Task 2: `retrieval_query` on AgenticRAGLoop and answer_with_retrieval

**Files:**
- Modify: `src/agents/search/agentic_rag.py` (`run`, lines ~197–360)
- Modify: `src/context/pipeline.py` (`answer_with_retrieval`, lines ~422–452)
- Test: `tests/unit/test_follow_up_retrieval_query.py`

**Interfaces:**
- Produces: `AgenticRAGLoop.run(question, *, retrieval_query: str | None = None, ...)` and `answer_with_retrieval(question, *, retrieval_query: str | None = None, ...)`. `None` means "use `question`".

- [ ] **Step 1: Write the failing tests**

```python
import asyncio
from unittest.mock import MagicMock

from src.agents.search.agentic_rag import AgenticRAGConfig, AgenticRAGLoop
from src.context import SearchContextBundle


def _loop(monkeypatch, seen):
    async def fake_retrieve(queries, **kwargs):
        seen["retrieved"].extend(queries)
        return [SearchContextBundle(query=q, documents=[]) for q in queries]

    monkeypatch.setattr("src.agents.search.agentic_rag.retrieve_contexts", fake_retrieve)
    llm = MagicMock()
    llm.complete.side_effect = lambda *args, **kwargs: "yes"
    return AgenticRAGLoop(AgenticRAGConfig(max_rounds=1), llm=llm)


def test_agentic_rag_retrieves_on_retrieval_query(monkeypatch):
    seen = {"retrieved": []}
    loop = _loop(monkeypatch, seen)
    asyncio.run(
        loop.run(
            "Does it support GPUs?",
            retrieval_query="What is FAISS?\nDoes it support GPUs?",
        )
    )
    assert "What is FAISS?\nDoes it support GPUs?" in seen["retrieved"]
    assert "Does it support GPUs?" not in seen["retrieved"]


def test_agentic_rag_without_retrieval_query_is_unchanged(monkeypatch):
    seen = {"retrieved": []}
    loop = _loop(monkeypatch, seen)
    asyncio.run(loop.run("Does it support GPUs?"))
    assert "Does it support GPUs?" in seen["retrieved"]
    assert not any("What is FAISS?" in q for q in seen["retrieved"])


def test_answer_with_retrieval_retrieves_on_retrieval_query(monkeypatch):
    from src.context import pipeline

    seen = {}

    async def fake_retrieve_context(question, **kwargs):
        seen["retrieved"] = question
        return SearchContextBundle(query=question, documents=[])

    monkeypatch.setattr(pipeline, "retrieve_context", fake_retrieve_context)
    result = asyncio.run(
        pipeline.answer_with_retrieval(
            "Does it support GPUs?",
            retrieval_query="What is FAISS?\nDoes it support GPUs?",
        )
    )
    assert seen["retrieved"] == "What is FAISS?\nDoes it support GPUs?"
    assert result is not None


def test_answer_with_retrieval_defaults_to_question(monkeypatch):
    from src.context import pipeline

    seen = {}

    async def fake_retrieve_context(question, **kwargs):
        seen["retrieved"] = question
        return SearchContextBundle(query=question, documents=[])

    monkeypatch.setattr(pipeline, "retrieve_context", fake_retrieve_context)
    asyncio.run(pipeline.answer_with_retrieval("Does it support GPUs?"))
    assert seen["retrieved"] == "Does it support GPUs?"
```

The AgenticRAG tests use a permissive LLM stub (every completion returns "yes"), since the enhancer, sufficiency judge and synthesis all call `llm.complete`; enhancement may add "yes"-derived queries, so the assertions only require that the (retrieval) question itself was retrieved and the other string was not. Confirm by reading `QueryEnhancer.enhance_async` → `bundle.all_queries()` that the original query is always included before relying on it.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_follow_up_retrieval_query.py -q`
Expected: the two `retrieval_query` tests FAIL with `TypeError: ... unexpected keyword argument 'retrieval_query'`; the two default tests PASS.

- [ ] **Step 3: Implement**

`src/agents/search/agentic_rag.py`, `run`: add the keyword-only parameter `retrieval_query: str | None = None` after `question`, and as the first statement after the `_emit` helper:

```python
        # Retrieval (enhancement, sufficiency, gap queries) runs on the resolved
        # standalone query; synthesis still answers the question as asked.
        search_question = retrieval_query or question
```

Then change exactly these three calls: `self._enhancer.enhance_async(question)` → `(search_question)`; `self._is_sufficient(question, merged)` → `(search_question, merged)`; `self._generate_followup(question, merged)` → `(search_question, merged)`. Leave both `SearchContextBundle(query=question, ...)` lines and the `AnswerGenerationRequest(question=question, ...)` unchanged.

`src/context/pipeline.py`, `answer_with_retrieval`: add `retrieval_query: str | None = None` as the last keyword parameter, and change the `retrieve_context(question, ...)` call's first argument to `retrieval_query or question`. Leave `collect_tool_evidence(question, ...)` and generation unchanged.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_follow_up_retrieval_query.py tests/unit/test_agentic_rag.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

Revert `enhance_async(search_question)` to `enhance_async(question)` → `test_agentic_rag_retrieves_on_retrieval_query` must FAIL. Revert `retrieval_query or question` in `answer_with_retrieval` to `question` → `test_answer_with_retrieval_retrieves_on_retrieval_query` must FAIL. Restore, re-run, PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agents/search/agentic_rag.py src/context/pipeline.py tests/unit/test_follow_up_retrieval_query.py
git commit -m "agents: retrieval_query kwarg separates what is searched from what is answered"
```

---

### Task 3: Thread `retrieval_query` through the web routing functions

**Files:**
- Modify: `src/internal/search/pipeline.py` (`SearchPipeline.run`)
- Modify: `src/internal/servers/web/app.py`: `_auto_search_pipeline` (~598), `_run_agentic_rag` (~748), `_run_search_direct_or_escalate` (~952), `_run_auto_routed` (~1199)
- Test: `tests/unit/servers/web/test_follow_up_wiring.py`

**Interfaces:**
- Consumes: `AgenticRAGLoop.run(retrieval_query=)` (Task 2).
- Produces: a keyword `retrieval_query: str | None = None` on `SearchPipeline.run`, `_auto_search_pipeline`, `_run_agentic_rag`, `_run_search_direct_or_escalate` and `_run_auto_routed`. `None` everywhere means today's behaviour.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio

import src.internal.servers.web.app as web_app
from src.shared_configs.intent import RouteStrategy

RESOLVED = "What is FAISS?\nDoes it support GPUs?"


def _search_call(monkeypatch, **kwargs):
    seen = {}

    async def fake_direct(query, **kw):
        seen["direct"] = query
        return []

    def fake_gate(query, docs, **kw):
        seen["gate"] = query
        return False, "weak", 0.0, None

    async def fake_agent(query, **kw):
        seen["agent"] = query
        return "agent answer", [], [], "search", {}

    async def fake_pipeline(query, **kw):
        seen["pipeline"] = (query, kw.get("retrieval_query"))
        return "pipeline answer", [], [], "search", kw.get("extra", {})

    monkeypatch.setattr(web_app, "_run_direct_search", fake_direct)
    monkeypatch.setattr(web_app, "_direct_gate_decision", fake_gate)
    monkeypatch.setattr(web_app, "_run_search_agent", fake_agent)
    monkeypatch.setattr(web_app, "_auto_search_pipeline", fake_pipeline)
    monkeypatch.setattr(web_app, "gate_embedder", lambda: None)
    asyncio.run(
        web_app._run_search_direct_or_escalate(
            "Does it support GPUs?",
            manager=kwargs.get("manager", object()),
            tokenizer=kwargs.get("tokenizer", object()),
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            source_provider="retrieval",
            on_turn=None,
            retrieval_query=kwargs.get("retrieval_query"),
        )
    )
    return seen


def test_direct_search_and_gate_use_the_resolved_query(monkeypatch):
    seen = _search_call(monkeypatch, retrieval_query=RESOLVED)
    assert seen["direct"] == RESOLVED
    assert seen["gate"] == RESOLVED


def test_escalation_to_the_search_agent_keeps_the_raw_message(monkeypatch):
    seen = _search_call(monkeypatch, retrieval_query=RESOLVED)
    assert seen["agent"] == "Does it support GPUs?"


def test_degraded_pipeline_gets_raw_query_and_resolved_retrieval_query(monkeypatch):
    seen = _search_call(monkeypatch, retrieval_query=RESOLVED, manager=None, tokenizer=None)
    assert seen["pipeline"] == ("Does it support GPUs?", RESOLVED)


def test_search_without_retrieval_query_is_unchanged(monkeypatch):
    seen = _search_call(monkeypatch)
    assert seen["direct"] == seen["gate"] == "Does it support GPUs?"


def test_auto_routed_chat_passes_retrieval_query_to_agentic_rag(monkeypatch):
    seen = {}

    async def fake_rag(query, **kw):
        seen["rag"] = (query, kw.get("retrieval_query"))
        return "answer", [], [], "chat", {}

    monkeypatch.setattr(web_app, "_run_agentic_rag", fake_rag)
    asyncio.run(
        web_app._run_auto_routed(
            "Does it support GPUs?",
            llm=object(),
            manager=None,
            tokenizer=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            resolved=None,
            forced_route=RouteStrategy.CHAT,
            retrieval_query=RESOLVED,
        )
    )
    assert seen["rag"] == ("Does it support GPUs?", RESOLVED)


def test_auto_routed_search_passes_retrieval_query(monkeypatch):
    seen = {}

    async def fake_search(query, **kw):
        seen["search"] = (query, kw.get("retrieval_query"))
        return "answer", [], [], "search", {}

    monkeypatch.setattr(web_app, "_run_search_direct_or_escalate", fake_search)
    asyncio.run(
        web_app._run_auto_routed(
            "Does it support GPUs?",
            llm=None,
            manager=None,
            tokenizer=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            resolved=None,
            forced_route=RouteStrategy.SEARCH,
            retrieval_query=RESOLVED,
        )
    )
    assert seen["search"] == ("Does it support GPUs?", RESOLVED)


def test_search_pipeline_uses_the_retrieval_query_override():
    from src.internal.search.pipeline import SearchPipeline

    seen = {}

    class Retrieval:
        async def retrieve(self, query, history, filters, top_k):
            seen["retrieve"] = query
            return []

    class Ranking:
        async def rank(self, query, candidates, top_k):
            seen["rank"] = query
            return []

    class Inference:
        async def generate(self, query, history, evidence):
            seen["generate"] = query
            return None

    pipeline = SearchPipeline(Retrieval(), Ranking(), Inference())
    try:
        asyncio.run(pipeline.run("Does it support GPUs?", [], None, 5, "retrieval", retrieval_query=RESOLVED))
    except Exception:
        pass  # the stub stages return nothing useful; only the queries matter
    assert seen["retrieve"] == RESOLVED
```

Before writing `test_search_pipeline_uses_the_retrieval_query_override`, read `SearchPipeline.__init__` and `run` (`src/internal/search/pipeline.py`) and shape the three stub stages to the exact method names and return types it calls, so the run reaches `retrieve` without the `try/except`; drop the `try/except` if it does. If `_run_auto_routed` with `resolved=None` fails before dispatch (it reads `app_settings`/`resolved`), pass `app_settings=load_app_settings()` and `resolved=load_app_settings()` from `src.internal.configs` instead.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/servers/web/test_follow_up_wiring.py -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'retrieval_query'` (the unchanged-behaviour test also errors until the kwarg exists).

- [ ] **Step 3: Implement**

`src/internal/search/pipeline.py`, `SearchPipeline.run`: add keyword `retrieval_query: str | None = None`; after `context = build_retrieval_context(query, history)` add `search_query = retrieval_query or context.retrieval_query` and replace every other use of `context.retrieval_query` in `run` (the `extra["retrieval_query"]` entry, the `retrieve(...)` call and the `rank(...)` call) with `search_query`. `context.history` and `generate(query, ...)` stay.

`src/internal/servers/web/app.py`:

- `_auto_search_pipeline`: add `retrieval_query: str | None = None`; call `pipeline.run(query, history, filters, top_k, source_provider, retrieval_query=retrieval_query)`.
- `_run_agentic_rag`: add `retrieval_query: str | None = None`; pass `retrieval_query=retrieval_query` to `rag_loop.run(...)`.
- `_run_search_direct_or_escalate`: add `retrieval_query: str | None = None` after `on_turn`; first statement of the body: `search_query = retrieval_query or query`. Use `search_query` for: the direct `_run_direct_search(...)` call, `_direct_gate_decision(...)`, the `"query"` in the `direct_retrieval` capture, `queries=[...]` and `"retrieval_query"` of the direct-hit return, and the external-fallback `_run_direct_search(...)`, `queries=[...]` and `"retrieval_query"`. Keep `query` for `_run_search_agent(...)`, the `logger.warning` text and the `"No results found for: ..."` message. Pass `retrieval_query=retrieval_query` to both `_auto_search_pipeline(...)` calls.
- `_run_auto_routed`: add `retrieval_query: str | None = None`; pass it to `_run_search_direct_or_escalate(...)`, `_run_agentic_rag(...)` and the CHAT-degraded `_auto_search_pipeline(...)`. The TOOL branch is unchanged.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/servers/web/test_follow_up_wiring.py tests/unit/test_search_route_access_filters.py tests/unit/test_execution_fallbacks.py tests/unit/search -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

(a) Pass `query` instead of `search_query` to the direct `_run_direct_search` → `test_direct_search_and_gate_use_the_resolved_query` must FAIL. (b) Pass `search_query` to `_run_search_agent` → `test_escalation_to_the_search_agent_keeps_the_raw_message` must FAIL. (c) Drop `retrieval_query=` from `_run_auto_routed`'s `_run_agentic_rag` call → `test_auto_routed_chat_passes_retrieval_query_to_agentic_rag` must FAIL. Restore, re-run, PASS.

- [ ] **Step 6: Commit**

```bash
git add src/internal/search/pipeline.py src/internal/servers/web/app.py tests/unit/servers/web/test_follow_up_wiring.py
git commit -m "web: thread retrieval_query through the search and chat routes"
```

---

### Task 4: Resolve once per request behind the flag

**Files:**
- Modify: `src/internal/servers/web/app.py`: `SearchExperienceSettings` (~170–215), `_run_agent_impl` (after `history = working.messages`, ~1691; the auto-route call ~1733; `chat_loop` ~1865; the `chat_once` fallthrough ~1977–2016)
- Test: `tests/unit/servers/web/test_follow_up_wiring.py` (append)

**Interfaces:**
- Consumes: `resolve_follow_up`, `Resolution` (Task 1), `follow_up_cos_min` (Task 1), the `retrieval_query` kwargs (Tasks 2–3).
- Produces: `SearchExperienceSettings.follow_up_resolution: bool` (env `AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION`, default off); response `hook_metadata["follow_up"] = {"continuation": bool, "reason": str, "query": str}` on the auto, `chat_loop` and `chat_once` paths when the flag is on.

- [ ] **Step 1: Write the failing tests** (append)

```python
from fastapi.testclient import TestClient

from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


class _ChatLLM:
    def complete(self, messages, **_):
        return "chat"


def _two_turns(monkeypatch, tmp_path, *, flag, second="Does it support GPUs?", mode=None):
    calls = []

    async def fake_rag(query, **kw):
        calls.append((query, kw.get("retrieval_query")))
        return "an answer", [], [], "chat", {}

    monkeypatch.setattr(web_app, "_run_agentic_rag", fake_rag)
    monkeypatch.setattr(web_app, "gate_embedder", lambda: None)
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3", follow_up_resolution=flag),
        llm=_ChatLLM(),
    )
    client = TestClient(app)
    body = {"query": "What is FAISS?"}
    if mode:
        body["mode"] = mode
    first = client.post("/api/agent", json=body).json()
    second_body = {**body, "query": second, "session_id": first["session_id"]}
    second_response = client.post("/api/agent", json=second_body).json()
    return calls, first, second_response


def test_flag_on_resolves_the_follow_up_for_retrieval(monkeypatch, tmp_path):
    calls, _first, second = _two_turns(monkeypatch, tmp_path, flag=True)
    assert calls[-1] == ("Does it support GPUs?", "What is FAISS?\nDoes it support GPUs?")
    assert second["hook_metadata"]["follow_up"] == {
        "continuation": True,
        "reason": "reference",
        "query": "What is FAISS?\nDoes it support GPUs?",
    }


def test_flag_on_first_turn_reports_no_history(monkeypatch, tmp_path):
    calls, first, _second = _two_turns(monkeypatch, tmp_path, flag=True)
    assert calls[0] == ("What is FAISS?", "What is FAISS?")
    assert first["hook_metadata"]["follow_up"] == {
        "continuation": False,
        "reason": "no_history",
        "query": "What is FAISS?",
    }


def test_flag_off_changes_nothing(monkeypatch, tmp_path):
    calls, first, second = _two_turns(monkeypatch, tmp_path, flag=False)
    assert calls == [("What is FAISS?", None), ("Does it support GPUs?", None)]
    assert "follow_up" not in first["hook_metadata"]
    assert "follow_up" not in second["hook_metadata"]


def test_explicit_chat_loop_mode_is_resolved_too(monkeypatch, tmp_path):
    calls, _first, _second = _two_turns(monkeypatch, tmp_path, flag=True, mode="chat_loop")
    assert calls[-1][1] == "What is FAISS?\nDoes it support GPUs?"


def test_chat_once_mode_passes_retrieval_query(monkeypatch, tmp_path):
    seen = []

    async def fake_awr(question, **kw):
        seen.append((question, kw.get("retrieval_query")))
        from src.context.models import AnswerGenerationResult
        from src.context import SearchContextBundle

        return AnswerGenerationResult(
            answer="an answer", citations=[], context=SearchContextBundle(query=question, documents=[])
        )

    monkeypatch.setattr(web_app, "answer_with_retrieval", fake_awr)
    monkeypatch.setattr(web_app, "gate_embedder", lambda: None)
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3", follow_up_resolution=True),
        llm=_ChatLLM(),
    )
    client = TestClient(app)
    first = client.post("/api/agent", json={"query": "What is FAISS?", "mode": "chat_once"}).json()
    client.post(
        "/api/agent",
        json={"query": "Does it support GPUs?", "mode": "chat_once", "session_id": first["session_id"]},
    )
    assert seen[-1] == ("Does it support GPUs?", "What is FAISS?\nDoes it support GPUs?")
```

Before relying on `AnswerGenerationResult(...)` in `test_chat_once_mode_passes_retrieval_query`, read its definition (`grep -n "class AnswerGenerationResult" -A15 src/context/models.py`) and construct it with its real required fields. Both "What is FAISS?" and "Does it support GPUs?" end in "?", which the regex stage routes to `chat`, so the auto path reaches `_run_agentic_rag` without the LLM classifier.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/servers/web/test_follow_up_wiring.py -q -k "flag or explicit or chat_once"`
Expected: FAIL, `TypeError: SearchExperienceSettings.__init__() got an unexpected keyword argument 'follow_up_resolution'`.

- [ ] **Step 3: Implement**

`SearchExperienceSettings`: add the field after `memory_auto_curate`:

```python
    # Resolve follow-up turns into standalone retrieval queries (the answer
    # prompt still gets the raw message). Off by default until the multi-turn
    # eval's success criteria hold.
    follow_up_resolution: bool = False
```

and in `from_app_settings`: `follow_up_resolution=_flag("AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION"),`.

Imports at the top of `app.py`: add `follow_up_cos_min` to the existing `from src.internal.utils.embedding_gate import (...)` block, and `from src.internal.search.context import resolve_follow_up`.

`_run_agent_impl`, immediately after `history = working.messages` and before `db.add_chat_message(...)`:

```python
        # Resolved once, for retrieval only: routing and the answer prompt keep
        # the message as typed.
        follow_up_meta: dict = {}
        retrieval_query: str | None = None
        if settings.follow_up_resolution:
            resolution = await asyncio.to_thread(
                resolve_follow_up,
                query,
                history,
                cosine=make_cosine_fn(gate_embedder()),
                tau=follow_up_cos_min(),
            )
            retrieval_query = resolution.query
            follow_up_meta = {
                "follow_up": {
                    "continuation": resolution.continuation,
                    "reason": resolution.reason,
                    "query": resolution.query,
                }
            }
```

`make_cosine_fn(None)` returns a function that always yields `None`, so a missing embedder skips the `semantic` rule.

Then:
- auto path: pass `retrieval_query=retrieval_query` to `_run_auto_routed(...)`, and before its `_finalize_response(...)` add `extra.update(follow_up_meta)`;
- `chat_loop`: pass `retrieval_query=retrieval_query` to `_run_agentic_rag(...)` and `extra.update(follow_up_meta)` before its `_finalize_response(...)`;
- `chat_once` fallthrough: pass `retrieval_query=retrieval_query` to `answer_with_retrieval(...)` and change its `_finalize_response(..., extra={}, ...)` to `extra=dict(follow_up_meta)`.

No other mode is changed.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/servers/web/test_follow_up_wiring.py tests/unit/servers/web/test_web_experience_app.py -q`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

(a) Replace `if settings.follow_up_resolution:` with `if True:` → `test_flag_off_changes_nothing` must FAIL. (b) Drop `extra.update(follow_up_meta)` on the auto path → `test_flag_on_resolves_the_follow_up_for_retrieval` must FAIL. (c) Pass `history=[]` to `resolve_follow_up` → the same test must FAIL. Restore, re-run, PASS.

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_follow_up_wiring.py
git commit -m "web: resolve follow-ups once per request behind AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION"
```

---

### Task 5: Harness — gated conditions, per-corpus slices, dev/test split, τ tuning, criteria

**Files:**
- Modify: `examples/measure_multi_turn_continuity.py`
- Test: `tests/unit/test_measure_multi_turn_continuity.py` (append; existing tests must keep passing)

**Interfaces:**
- Consumes: `resolve_follow_up`, `Resolution` (Task 1); `gate_embedder`, `make_cosine_fn` (`src.internal.utils.embedding_gate`).
- Produces:
  - `Resolver = Callable[[str, list[ChatMessage]], Resolution]`;
  - `evaluate(conversations, routers, retrievers, resolvers: dict[str, Resolver] | None = None)`; rows gain `corpus` (`turn.corpus`), `continuation` (`bool | None`) and `gate_reason` (`str | None`); resolver conditions follow the four baseline conditions;
  - `summarize` derives its conditions from the rows (first-seen order; `raw` first) and adds `<relation>/<corpus>` slices; `format_table` prints whatever conditions the summary holds and adds the `follow_up/scifact` and `topic_switch/scifact` slices;
  - `split_of(conversation_id: str) -> str` (`"dev" | "test"`, `sha256(id)` first byte even → dev);
  - `gate_accuracy(rows, condition) -> dict` (`{"mean", "n"}` over non-opening rows of that condition);
  - `choose_tau(conversations, cosine, grid=TAU_GRID) -> float`;
  - `median_latency_ms(conversations, cosine, tau) -> float`;
  - `check_criteria(summary, rows, latency_ms) -> dict[str, bool]`;
  - `TAU_GRID = tuple(round(0.70 + 0.01 * i, 2) for i in range(26))`.

- [ ] **Step 1: Write the failing tests** (append; add `choose_tau, check_criteria, gate_accuracy, median_latency_ms, split_of` to the import block and `from src.internal.search.context import resolve_follow_up`)

```python
from functools import partial


def _resolvers(cosine=None, tau=0.8):
    return {
        "gated": partial(resolve_follow_up, cosine=cosine, tau=tau),
        "gated_cues": partial(resolve_follow_up, cosine=None, tau=tau),
    }


OSLO_SWITCH = Turn(
    "Tell me the Oslo weather forecast for the coming weekend please.",
    "tool",
    "topic_switch",
    "Tell me the Oslo weather forecast for the coming weekend please.",
)


def test_evaluate_adds_gated_conditions_with_continuation():
    conv = Conversation("c", (TOKYO, PARIS, OSLO_SWITCH))
    rows = evaluate([conv], {"rules": lambda q: "tool"}, {}, _resolvers())
    by = {(r["turn_index"], r["condition"]): r for r in rows}
    assert by[(1, "gated")]["query"] == "weather in Tokyo\nand in Paris?"
    assert by[(1, "gated")]["continuation"] is True
    assert by[(2, "gated")]["query"] == OSLO_SWITCH.text
    assert by[(2, "gated")]["carried"] is False
    assert by[(1, "raw")]["continuation"] is None
    assert len(rows) == 3 * 6


def test_gate_accuracy_scores_non_opening_turns():
    conv = Conversation("c", (TOKYO, PARIS, OSLO_SWITCH))
    rows = evaluate([conv], {"rules": lambda q: "tool"}, {}, _resolvers())
    assert gate_accuracy(rows, "gated") == {"mean": 1.0, "n": 2}


def test_summarize_reports_resolver_conditions_and_corpus_slices():
    conv = Conversation("c", (FAISS, FAISS_TYPES))
    rows = evaluate(
        [conv],
        {"rules": lambda q: "search"},
        {"tfidf": lambda corpus, q: ["d2"] if "FAISS" in q and "variants" in q else ["d1"]},
        _resolvers(),
    )
    summary = summarize(rows, ["rules"], ["tfidf"], resamples=20, seed=0)
    hit = summary["follow_up/demo"]["hit5:tfidf"]
    assert hit["raw"]["mean"] == 0.0
    assert hit["gated"]["mean"] == 1.0
    assert "gated" in format_table(summary)


def test_split_is_stable_and_roughly_balanced():
    ids = [f"conv-{i}" for i in range(200)]
    splits = [split_of(i) for i in ids]
    assert splits == [split_of(i) for i in ids]
    assert 70 <= splits.count("dev") <= 130


def test_choose_tau_picks_the_best_threshold_and_breaks_ties_high():
    follow = Turn(
        "Which compounds activate the dephosphorylated form?",
        "search", "follow_up", "Which compounds activate dephosphorylated AMPK?",
        kind="pronoun", corpus="demo", relevant_doc_ids=("d1",),
    )
    switch = Turn(
        "Explain the causes of the French revolution in detail.",
        "chat", "topic_switch", "Explain the causes of the French revolution in detail.",
    )
    opening = Turn("AMPK activation reduces lung fibrosis.", "search", "opening",
                   "AMPK activation reduces lung fibrosis.", corpus="demo", relevant_doc_ids=("d1",))
    convs = [Conversation("a", (opening, follow)), Conversation("b", (opening, switch))]
    scores = {follow.text: 0.9, switch.text: 0.75}
    tau = choose_tau(convs, lambda message, topic: scores.get(message, 0.0))
    # Any τ in (0.75, 0.90] classifies both correctly; ties resolve to the highest.
    assert tau == 0.9


def test_median_latency_is_measured_in_milliseconds():
    conv = Conversation("c", (TOKYO, PARIS, OSLO_SWITCH))
    assert median_latency_ms([conv], lambda m, t: 0.5, tau=0.8) >= 0.0


def test_check_criteria_reads_the_four_thresholds():
    summary = {
        "follow_up/scifact": {"hit5:tfidf": {
            "gated": {"mean": 0.8, "n": 10, "delta_vs_raw": {"point": 0.3, "low": 0.1, "high": 0.5}},
            "concat": {"mean": 0.9, "n": 10, "delta_vs_raw": None},
        }},
        "topic_switch": {"carry_over": {"gated": {"mean": 0.1, "n": 10, "delta_vs_raw": None}}},
    }
    got = check_criteria(summary, [], latency_ms=12.0)
    assert got == {
        "gated_beats_raw": True,
        "carry_over_at_most_0.15": True,
        "within_one_hit_of_concat": True,
        "latency_at_most_30ms": True,
    }
    summary["topic_switch"]["carry_over"]["gated"]["mean"] = 0.2
    assert check_criteria(summary, [], latency_ms=40.0)["carry_over_at_most_0.15"] is False
```

`within_one_hit_of_concat` compares hit **counts**: `round(mean * n)`; with n=10, gated 8 vs concat 9 is within one.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -q`
Expected: collection `ImportError` for the new names.

- [ ] **Step 3: Implement**

Add to the top-level imports: `import hashlib`, `import statistics`, `import time`, `from functools import lru_cache, partial`, and `from src.internal.search.context import Resolution, resolve_follow_up`.

Constants and helpers (below `MRR_K`):

```python
TAU_GRID = tuple(round(0.70 + 0.01 * i, 2) for i in range(26))
CARRY_OVER_MAX = 0.15
LATENCY_MAX_MS = 30.0

Resolver = Callable[[str, list[ChatMessage]], Resolution]


def split_of(conversation_id: str) -> str:
    return "dev" if hashlib.sha256(conversation_id.encode()).digest()[0] % 2 == 0 else "test"
```

`evaluate`: add `resolvers: dict[str, Resolver] | None = None`. Inside the per-turn loop, iterate `(*CONDITIONS, *(resolvers or {}))`; for a resolver condition compute `resolution = resolvers[condition](turn.text, _history([t.text for t in conversation.turns[:index]]))`, `query = resolution.query`, and set `continuation = resolution.continuation`, `gate_reason = resolution.reason`; for baseline conditions `query = build_query(...)`, `continuation = gate_reason = None`. Add `"corpus": turn.corpus`, `"continuation": continuation`, `"gate_reason": gate_reason` to the row dict.

`_slices(corpora: Sequence[str])`: after the relation and kind slices add

```python
    for relation in RELATIONS:
        for corpus in corpora:
            slices[f"{relation}/{corpus}"] = (
                lambda row, relation=relation, corpus=corpus: row["relation"] == relation
                and row.get("corpus") == corpus
            )
```

`summarize`: compute `conditions = list(dict.fromkeys(row["condition"] for row in rows))` and `corpora = sorted({row["corpus"] for row in rows if row.get("corpus")})`; iterate `conditions` instead of `CONDITIONS` and call `_slices(corpora)`.

`format_table`: iterate `per_condition` (its keys) instead of `CONDITIONS`, and print slices `("all", "follow_up", "topic_switch", "follow_up/scifact", "topic_switch/scifact")`.

New functions:

```python
def gate_accuracy(rows: Sequence[dict], condition: str) -> dict:
    scored = [
        float(row["continuation"] == (row["relation"] == "follow_up"))
        for row in rows
        if row["condition"] == condition and row["relation"] != "opening"
    ]
    return {"mean": _mean(scored) if scored else None, "n": len(scored)}


def _resolve_all(conversations, cosine, tau):
    for conversation in conversations:
        for index, turn in enumerate(conversation.turns):
            if turn.relation == "opening":
                continue
            history = _history([t.text for t in conversation.turns[:index]])
            yield turn, history, partial(resolve_follow_up, cosine=cosine, tau=tau)


def choose_tau(conversations, cosine, grid=TAU_GRID) -> float:
    best_tau, best_acc = grid[0], -1.0
    for tau in grid:
        results = [
            resolve(turn.text, history).continuation == (turn.relation == "follow_up")
            for turn, history, resolve in _resolve_all(conversations, cosine, tau)
        ]
        accuracy = _mean([float(r) for r in results])
        if accuracy >= best_acc:  # ties resolve to the highest τ
            best_tau, best_acc = tau, accuracy
    return best_tau


def median_latency_ms(conversations, cosine, tau) -> float:
    timings = []
    for turn, history, resolve in _resolve_all(conversations, cosine, tau):
        start = time.perf_counter()
        resolve(turn.text, history)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def check_criteria(summary: dict, rows: Sequence[dict], latency_ms: float) -> dict[str, bool]:
    hit = summary["follow_up/scifact"]["hit5:tfidf"]
    gated, concat = hit["gated"], hit["concat"]
    return {
        "gated_beats_raw": gated["delta_vs_raw"]["low"] > 0,
        "carry_over_at_most_0.15": summary["topic_switch"]["carry_over"]["gated"]["mean"]
        <= CARRY_OVER_MAX,
        "within_one_hit_of_concat": round(gated["mean"] * gated["n"])
        >= round(concat["mean"] * concat["n"]) - 1,
        "latency_at_most_30ms": latency_ms <= LATENCY_MAX_MS,
    }
```

`main`: add `--tau` (`type=float`, default `None`). After loading corpora and routers:

```python
    from src.internal.utils.embedding_gate import gate_embedder, make_cosine_fn

    embedder = gate_embedder()
    if embedder is None:
        raise SystemExit("gated condition needs the e5 gate embedder; it did not load")
    cosine = make_cosine_fn(embedder)
    cached = lru_cache(maxsize=None)(cosine)
    dev = [c for c in conversations if split_of(c.id) == "dev"]
    test = [c for c in conversations if split_of(c.id) == "test"]
    tau = args.tau if args.tau is not None else choose_tau(dev, cached)
    resolvers = {
        "gated": partial(resolve_follow_up, cosine=cached, tau=tau),
        "gated_cues": partial(resolve_follow_up, cosine=None, tau=tau),
    }
    rows = evaluate(test, routers, retrievers, resolvers)
    summary = summarize(rows, args.routers, args.retrievers, resamples=args.resamples, seed=args.seed)
    latency = median_latency_ms(test, cosine, tau)
    gates = {name: gate_accuracy(rows, name) for name in resolvers}
    criteria = check_criteria(summary, rows, latency)
```

Print `format_table(summary)`, then `tau`, the dev/test conversation counts, `gates`, `latency` and `criteria`; write all of them plus `config`, `summary` and `rows` to `--out`. `check_criteria` raises `KeyError` when the dataset has no SciFact follow-ups or no switches; that is intended (the criteria cannot be judged).

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -q`
Expected: all PASS, including every pre-existing test.

- [ ] **Step 5: Torch-free check**

```bash
python - <<'EOF'
import sys, pytest
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith(("torch.", "sentence_transformers", "transformers")):
            raise ImportError(f"blocked {name}")
sys.meta_path.insert(0, _Block())
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/unit/test_measure_multi_turn_continuity.py", "tests/unit/search/test_follow_up_resolution.py", "tests/unit/test_follow_up_retrieval_query.py"]))
EOF
```
Expected: all pass, none skipped.

- [ ] **Step 6: Mutation-check**

(a) In `choose_tau` change `>=` to `>` → `test_choose_tau_picks_the_best_threshold_and_breaks_ties_high` must FAIL. (b) In `gate_accuracy` drop the `relation != "opening"` filter → `test_gate_accuracy_scores_non_opening_turns` must FAIL. (c) In `check_criteria` compare means instead of hit counts for criterion 3 (`gated["mean"] >= concat["mean"]`) → `test_check_criteria_reads_the_four_thresholds` must FAIL. Restore, re-run, PASS.

- [ ] **Step 7: Commit**

```bash
git add examples/measure_multi_turn_continuity.py tests/unit/test_measure_multi_turn_continuity.py
git commit -m "examples: gated resolver conditions, per-corpus slices, dev/test split and criteria"
```

---

### Task 6: Dataset v2 (before any gated run)

**Files:**
- Modify: `data/eval/multi_turn_conversations.jsonl` (force-added)
- Modify: `tests/unit/test_measure_multi_turn_continuity.py` (quota and qrels tests)

Do not run `main` until this task's commit exists.

- [ ] **Step 1: Update the tests first**

Replace the SciFact part of `test_committed_dataset_meets_the_spec_composition`'s opening quota with `openings["scifact"] >= 30` and add:

```python
    scifact_switches = [
        t for c in conversations for t in c.turns
        if t.relation == "topic_switch" and t.corpus == "scifact"
    ]
    assert len(scifact_switches) >= 10
```

In `test_scifact_gold_labels_match_beir_qrels`, replace the text assertion with:

```python
                if turn.relation == "follow_up":
                    # Resolver-realistic: the question with its referent filled
                    # in, never the BEIR claim (claims can contain the answer).
                    assert turn.gold_rewrite != queries[turn.beir_query_id], conv.id
                else:
                    assert turn.gold_rewrite == queries[turn.beir_query_id], conv.id
```

Run `pytest tests/unit/test_measure_multi_turn_continuity.py -q -k "committed_dataset or scifact_gold"`. Expected: both FAIL (12 SciFact openings; follow-up golds equal BEIR text).

- [ ] **Step 2: Rewrite the 12 existing SciFact follow-up gold rewrites**

Keep each follow-up's `text`, `beir_query_id` and `relevant_doc_ids`; set `gold_rewrite` to:

| id | gold_rewrite |
|---|---|
| sci-01 | Do vitamin B12 levels, together with folate, influence the association between homocysteine and preeclampsia? |
| sci-02 | Are PI3K plus MEK inhibitors effective against KRAS mutant tumors? |
| sci-03 | Which compounds activate dephosphorylated AMPK? |
| sci-04 (turn 1) | Which mutation makes HIV resistant to AZT (zidovudine)? |
| sci-04 (turn 2, chat) | Summarize in plain language which mutation makes HIV resistant to AZT. |
| sci-05 | Can GATA3 replace OCT4 and SOX2 to reprogram human cells? |
| sci-06 | What does PTEN do to PtdIns(3,4)P2? |
| sci-07 | What is the role of vitamin D in the relationship between calcium and parathyroid hormone? |
| sci-08 | Does CRP predict exacerbations in COPD? |
| sci-09 | What phosphorylates the ATM protein after DNA damage? |
| sci-10 | Does tirasemtiv target cardiac muscle? |
| sci-11 | Does lack of FGF21 in mice cause atherosclerotic plaque formation? |
| sci-12 | Does the DESMOND program affect biochemical outcomes? |

- [ ] **Step 3: Add 18 SciFact conversations (sci-13 … sci-30)**

Each pair is (opening BEIR id → follow-up BEIR id). The opening is the BEIR claim verbatim with its qrels and `beir_query_id`. The follow-up's `text` refers back instead of naming the entity (pronoun, "that protein", a fragment, or "what about …"), its `gold_rewrite` is that question with the entity filled in and no answer terms from the BEIR claim, and its `relevant_doc_ids` are the follow-up id's qrels.

| id | opening → follow-up |
|---|---|
| sci-13 | 216 → 218 (CX3CR1) |
| sci-14 | 1174 → 1180 (MDA5) |
| sci-15 | 612 → 614 (LRRK2) |
| sci-16 | 413 → 416 (APOE4) |
| sci-17 | 909 → 910 (PKG-la) |
| sci-18 | 745 → 747 (MafA) |
| sci-19 | 76 → 723 (Ly49Q) |
| sci-20 | 1140 → 1141 (α-tocopheryl acetate) |
| sci-21 | 24 → 576 (PD-1) |
| sci-22 | 189 → 43 (HIV-1) |
| sci-23 | 401 → 82 (BMP4) |
| sci-24 | 1028 → 1334 (Type 1 Diabetes Tregs) |
| sci-25 | 651 → 1352 (West Nile virus) |
| sci-26 | 441 → 323 (G-CSF) |
| sci-27 | 700 → 702 (PIN1) |
| sci-28 | 484 → 485 (H4 G94P) |
| sci-29 | 1234 → 547 (IL-10) |
| sci-30 | 336 → 890 (DENV-1) |

Give sci-13 … sci-22 a third turn: `relation: topic_switch`, `route: search`, `corpus: scifact`, the BEIR claim verbatim with its qrels and `beir_query_id`, from these unrelated ids in order: 793, 1139, 253, 396, 186, 1142, 340, 1018, 529, 1349.

Before writing, confirm each id has qrels (`data/beir/scifact/qrels/{train,test}.tsv`); replace any that does not with another same-entity pair from the list the #641 run used (`sci_pairs.txt` procedure in `docs/superpowers/plans/2026-09-24-multi-turn-continuity-eval.md` Task 6 Step 3) and record the swap in the commit message. Aim for about half the new follow-ups without a cue word (`_is_follow_up(text)` false), matching v1.

Generate the file with a scratchpad script that reads text and qrels from BEIR (as v1 did), never retyping claims.

- [ ] **Step 4: Validate**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -q` — expected all PASS.
Then: `python -c "from examples.measure_multi_turn_continuity import *; c=load_conversations(DEFAULT_DATA); check_relevant_ids(c, load_corpora({t.corpus for x in c for t in x.turns if t.corpus})); print(len(c), 'ok', sum(split_of(x.id)=='test' for x in c), 'test')"` — expected `58 ok <n> test`. (Loads data only; runs no resolver, router or retriever.)

- [ ] **Step 5: Commit before running anything**

```bash
git add -f data/eval/multi_turn_conversations.jsonl
git add tests/unit/test_measure_multi_turn_continuity.py
git commit -m "data: multi-turn eval v2 — resolver-realistic SciFact golds, 30 SciFact conversations, 10 SciFact switches"
```

---

### Task 7: Run, set τ, decide the default, record results, open the PR

**Files:**
- Modify: `src/internal/utils/embedding_gate.py` (`DEFAULT_FOLLOW_UP_COS_MIN`)
- Modify (only if all criteria hold): `src/internal/servers/web/app.py` (flag default)
- Create: `data/eval/multi_turn_continuity.json` (force-added; overwrites v1 output)
- Modify: `docs/superpowers/specs/2026-09-24-follow-up-resolution-design.md` (append `## Results`)

- [ ] **Step 1: Run**

```bash
caffeinate -i python3 -m examples.measure_multi_turn_continuity 2>&1 | tee "$SCRATCH/follow_up_run.log"
```
(`$SCRATCH` = the session scratchpad directory.) Expected: tables, the chosen τ, gate accuracies, latency, the four criteria, `wrote data/eval/multi_turn_continuity.json`.

- [ ] **Step 2: Sanity checks before believing anything**

- `gated_cues` gate accuracy ≤ `gated` (semantic can only add correct continuations if τ was tuned sensibly); if not, read the dev/test gate reasons.
- `gated` rows on `topic_switch` with `carried: true` — read each one's `gate_reason`; they should be `reference` or `fragment` (expected weak spots), never `no_history`.
- `concat` SciFact follow-up tfidf hits on the test half, and `raw`, both computed by hand from the rows, match the summary.
- Latency printed is the uncached cosine (`median_latency_ms(test, cosine, tau)`), not `cached`.

If any fails, stop and debug; do not report numbers.

- [ ] **Step 3: Set τ**

Set `DEFAULT_FOLLOW_UP_COS_MIN` in `src/internal/utils/embedding_gate.py` to the printed dev-chosen τ and update its comment to `# Chosen on the multi-turn eval's dev half (<date>, τ grid 0.70–0.95).`. Run `pytest tests/unit/search/test_follow_up_resolution.py -q` (expected PASS).

- [ ] **Step 4: Decide the default**

If all four criteria are `True`: in `SearchExperienceSettings.from_app_settings` replace `follow_up_resolution=_flag("AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION")` with

```python
            follow_up_resolution=os.environ.get(
                "AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION", "1"
            ).strip().lower()
            in {"1", "true", "yes"},
```

change the field default to `True` and its comment to say it is on by default because the eval's criteria held (link the spec). Add a test to `test_follow_up_wiring.py`:

```python
def test_flag_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION", raising=False)
    assert SearchExperienceSettings.from_app_settings().follow_up_resolution is True
    monkeypatch.setenv("AGENTIC_SEARCH_FOLLOW_UP_RESOLUTION", "0")
    assert SearchExperienceSettings.from_app_settings().follow_up_resolution is False
```

Write it first, watch it fail, then make the change. If any criterion is `False`, leave the default off and add nothing.

- [ ] **Step 5: Full suite**

Run: `pytest -q` (redirect to a file; read the tail). Expected: all pass. If a run stalls in one file, run that file alone before concluding anything (a memory-pressure stall happened in #641).

- [ ] **Step 6: Append `## Results` to the spec**

Test-half only: τ and how it was chosen; gate accuracy for `gated` and `gated_cues` with the reason breakdown; SciFact follow-up Hit@5 (tfidf and hybrid) for `raw`, `regex`, `concat`, `gated`, `gated_cues`, `gold_rewrite` as counts with CIs vs raw; topic-switch carry-over and SciFact switch Hit@5 per condition; latency; each criterion with its value; the default decision. Every number must be recomputed from `data/eval/multi_turn_continuity.json` before it is written.

- [ ] **Step 7: Commit and open the PR**

```bash
git add -f data/eval/multi_turn_continuity.json
git add src/internal/utils/embedding_gate.py src/internal/servers/web/app.py tests/unit/servers/web/test_follow_up_wiring.py docs/superpowers/specs/2026-09-24-follow-up-resolution-design.md
git commit -m "search: follow-up resolution results and τ from the dev half"
git push -u origin feat/follow-up-resolution
gh pr create --title "Resolve follow-up turns before retrieval with a switch-aware gate" --body-file "$SCRATCH/pr_body.md"
```

Write `$SCRATCH/pr_body.md` first: a What section (resolver, wiring, flag), the Step 6 results table and criteria with the default decision stated in words, the test plan (suites run with counts, torch-free check, mutation checks, full suite), and the attribution line `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

- [ ] **Step 8: After merge**, confirm `git diff feat/follow-up-resolution origin/main -- src examples tests data/eval docs` is empty (squash has dropped late commits here before).
