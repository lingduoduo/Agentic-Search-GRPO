# Multi-turn Continuity Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic, LLM-free harness that measures how routing and retrieval handle follow-up turns and topic switches, under four query conditions, with conversation-clustered CIs.

**Architecture:** One module, `examples/measure_multi_turn_continuity.py`, in five layers: dataset loading/validation → per-condition query construction → per-turn scoring against injected router/retriever callables → aggregation with `cluster_bootstrap_ci` → CLI that builds the real routers (`recognize_intent`) and retrievers (`TfidfRetriever`, e5 hybrid). The dataset `data/eval/multi_turn_conversations.jsonl` is hand-written and committed before any real run.

**Tech Stack:** Python 3.10+, pytest, numpy, scikit-learn (TF-IDF), sentence-transformers (only `knn` router and `hybrid` retriever).

**Spec:** `docs/superpowers/specs/2026-09-24-multi-turn-continuity-eval-design.md`

## Global Constraints

- No LLM anywhere: `recognize_intent(..., llm=None, explicit_source=False)`.
- Conditions are exactly `raw`, `regex`, `concat`, `gold_rewrite` (never "oracle").
- Routers are exactly `rules` (`intent_index_path=None`) and `knn` (`intent_index_path=data/intent_index`); `knn` fails the run if the index does not load.
- Retrievers are exactly `tfidf` and `hybrid`; `hybrid` never degrades to TF-IDF (build the encoder directly, let it raise).
- Hit@5 and MRR@10; a clarify decision is a routing miss.
- CIs: `cluster_bootstrap_ci` from `src/model/post_training/eval/stats.py`, resampling conversations.
- The unit tests must pass with torch unimportable (CI installs `requirements-unit-test.txt`, no torch): import `recognize_intent`, retrievers and sentence-transformers lazily inside the factory functions only.
- `data/` is gitignored; tracked files there are force-added (`git add -f`).
- Work on branch `feat/multi-turn-continuity-eval`; never commit to `main`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **The kNN index fails to load** (wrong path, missing model): `knn` would silently equal `rules`. Expect a `SystemExit` naming the path — test in Task 5.
2. **A typo in `relevant_doc_ids`**: the turn would score as a permanent miss for every condition. Expect a `ValueError` listing the unknown ids — test in Task 5.
3. **`corpus_scifact.jsonl` absent** (it is untracked): expect a `SystemExit` naming the corpus and file, not a run over fewer turns — test in Task 5.
4. **A follow-up whose previous turn is itself a follow-up** ("weather in Tokyo" → "and in Paris?" → "and Berlin?"): `concat` must use the immediately previous user turn, `regex` the most recent non-follow-up — test in Task 2.
5. **A retriever returning fewer than K results or none** (TF-IDF drops zero-score docs): expect `hit5=False`, `rr10=0.0`, no crash — test in Task 3.

---

### Task 1: Dataset model, loader and validation

**Files:**
- Create: `examples/measure_multi_turn_continuity.py`
- Test: `tests/unit/test_measure_multi_turn_continuity.py`

**Interfaces:**
- Produces: `Turn`, `Conversation` (frozen dataclasses below), `load_conversations(path: Path) -> list[Conversation]` raising `ValueError(f"{where}: ...")`; constants `ROUTES`, `RELATIONS`, `KINDS`, `CONDITIONS`, `HIT_K = 5`, `MRR_K = 10`, `DEFAULT_DATA`, `DEFAULT_OUT`, `INTENT_INDEX_DIR`.

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

import pytest

from examples.measure_multi_turn_continuity import (
    Conversation,
    Turn,
    load_conversations,
)


def _write(tmp_path: Path, conversations: list[dict]) -> Path:
    path = tmp_path / "conversations.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in conversations) + "\n")
    return path


def _search(text, relation="opening", kind=None, gold=None, ids=("d1",)):
    turn = {
        "text": text,
        "route": "search",
        "relation": relation,
        "gold_rewrite": gold or text,
        "corpus": "demo",
        "relevant_doc_ids": list(ids),
    }
    if kind:
        turn["kind"] = kind
    return turn


def _tool(text, relation="opening", kind=None, gold=None):
    turn = {"text": text, "route": "tool", "relation": relation, "gold_rewrite": gold or text}
    if kind:
        turn["kind"] = kind
    return turn


def test_loads_a_valid_conversation(tmp_path):
    path = _write(
        tmp_path,
        [{"id": "c1", "turns": [
            _tool("weather in Tokyo"),
            _tool("and in Paris?", "follow_up", "ellipsis", "weather in Paris"),
        ]}],
    )
    [conv] = load_conversations(path)
    assert conv == Conversation(
        "c1",
        (
            Turn("weather in Tokyo", "tool", "opening", "weather in Tokyo"),
            Turn("and in Paris?", "tool", "follow_up", "weather in Paris", kind="ellipsis"),
        ),
    )


@pytest.mark.parametrize(
    ("turns", "message"),
    [
        ([_tool("and in Paris?", "follow_up", "ellipsis", "weather in Paris"), _tool("x")], "first turn must be an opening"),
        ([_tool("weather in Tokyo"), _tool("weather in Oslo", "tangent")], "unknown relation"),
        ([_tool("weather in Tokyo"), {**_search("faiss", "topic_switch"), "relevant_doc_ids": []}], "relevant_doc_ids"),
        ([_tool("weather in Tokyo"), {**_tool("weather in Oslo", "topic_switch"), "corpus": "demo"}], "relevant_doc_ids"),
        ([_tool("weather in Tokyo"), _tool("and Paris?", "follow_up", None, "weather in Paris")], "kind"),
        ([_tool("weather in Tokyo"), _tool("and Paris?", "follow_up", "sarcasm", "weather in Paris")], "unknown kind"),
        ([_tool("weather in Tokyo"), _tool("weather in Oslo", "topic_switch", gold="Oslo weather")], "gold_rewrite must equal"),
        ([_tool("weather in Tokyo"), {**_tool("weather in Oslo", "topic_switch"), "note": "x"}], "unknown keys"),
        ([_tool("weather in Tokyo")], "at least 2 turns"),
        ([_tool("weather in Tokyo"), _tool("weather in Oslo", "opening")], "only the first turn"),
    ],
)
def test_rejects_invalid_conversations(tmp_path, turns, message):
    path = _write(tmp_path, [{"id": "c1", "turns": turns}])
    with pytest.raises(ValueError, match=message):
        load_conversations(path)


def test_rejects_duplicate_ids(tmp_path):
    conv = {"id": "c1", "turns": [_tool("weather in Tokyo"), _tool("weather in Oslo", "topic_switch")]}
    with pytest.raises(ValueError, match="duplicate id"):
        load_conversations(_write(tmp_path, [conv, conv]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v`
Expected: collection error, `ModuleNotFoundError: examples.measure_multi_turn_continuity`.

- [ ] **Step 3: Implement**

```python
"""Measure how routing and retrieval handle follow-up turns and topic switches.

History reaches every chat surface but only the answer prompt reads it: intent
routing and the default retrieval paths see the latest user message alone. This
harness replays hand-written conversations and scores each user turn under four
query conditions:

- ``raw``: the message as typed — today's behaviour;
- ``regex``: ``build_retrieval_context``, the existing deterministic rewriter
  that only the degraded no-LLM paths use;
- ``concat``: the previous user message glued to this one;
- ``gold_rewrite``: the hand-written standalone query — the ceiling a perfect
  resolver reaches.

It reports route accuracy (``recognize_intent`` with no LLM, clarify = miss),
retrieval Hit@5 / MRR@10, and the topic-switch carry-over rate, each difference
against ``raw`` with a 95% CI that resamples conversations.

Run:

    python -m examples.measure_multi_turn_continuity

``--routers rules --retrievers tfidf`` runs without sentence-transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA = Path("data/eval/multi_turn_conversations.jsonl")
DEFAULT_OUT = Path("data/eval/multi_turn_continuity.json")
INTENT_INDEX_DIR = Path("data/intent_index")

ROUTES = ("chat", "search", "tool")
RELATIONS = ("opening", "follow_up", "topic_switch")
KINDS = ("pronoun", "ellipsis", "comparison", "elaboration")
CONDITIONS = ("raw", "regex", "concat", "gold_rewrite")
HIT_K = 5
MRR_K = 10

_TURN_KEYS = {
    "text",
    "route",
    "relation",
    "kind",
    "gold_rewrite",
    "corpus",
    "relevant_doc_ids",
    "beir_query_id",
}


@dataclass(frozen=True)
class Turn:
    text: str
    route: str
    relation: str
    gold_rewrite: str
    kind: str | None = None
    corpus: str | None = None
    relevant_doc_ids: tuple[str, ...] = ()
    beir_query_id: str | None = None


@dataclass(frozen=True)
class Conversation:
    id: str
    turns: tuple[Turn, ...]


def _parse_turn(where: str, raw: dict) -> Turn:
    unknown = set(raw) - _TURN_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("text", "route", "relation", "gold_rewrite"):
        if not str(raw.get(key, "")).strip():
            raise ValueError(f"{where}: missing {key}")
    turn = Turn(
        text=raw["text"],
        route=raw["route"],
        relation=raw["relation"],
        gold_rewrite=raw["gold_rewrite"],
        kind=raw.get("kind"),
        corpus=raw.get("corpus"),
        relevant_doc_ids=tuple(str(d) for d in raw.get("relevant_doc_ids", ())),
        beir_query_id=raw.get("beir_query_id"),
    )
    if turn.route not in ROUTES:
        raise ValueError(f"{where}: unknown route {turn.route!r}")
    if turn.relation not in RELATIONS:
        raise ValueError(f"{where}: unknown relation {turn.relation!r}")
    if (turn.relation == "follow_up") != (turn.kind is not None):
        raise ValueError(f"{where}: kind is required on follow-ups and only there")
    if turn.kind is not None and turn.kind not in KINDS:
        raise ValueError(f"{where}: unknown kind {turn.kind!r}")
    if turn.relation != "follow_up" and turn.gold_rewrite != turn.text:
        raise ValueError(f"{where}: a {turn.relation} turn's gold_rewrite must equal its text")
    if (turn.route == "search") != bool(turn.corpus and turn.relevant_doc_ids):
        raise ValueError(
            f"{where}: corpus and relevant_doc_ids are required on search turns and only there"
        )
    return turn


def load_conversations(path: Path) -> list[Conversation]:
    conversations: list[Conversation] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        conv_id = str(raw.get("id", "")).strip()
        if not conv_id:
            raise ValueError(f"line {line_no}: missing id")
        if conv_id in seen:
            raise ValueError(f"{conv_id}: duplicate id")
        seen.add(conv_id)
        raw_turns = raw.get("turns") or []
        if len(raw_turns) < 2:
            raise ValueError(f"{conv_id}: needs at least 2 turns")
        turns = tuple(
            _parse_turn(f"{conv_id} turn {i}", t) for i, t in enumerate(raw_turns)
        )
        if turns[0].relation != "opening":
            raise ValueError(f"{conv_id} turn 0: first turn must be an opening")
        if any(t.relation == "opening" for t in turns[1:]):
            raise ValueError(f"{conv_id}: only the first turn may be an opening")
        conversations.append(Conversation(conv_id, turns))
    if not conversations:
        raise ValueError(f"{path}: no conversations")
    return conversations
```

Note the "first turn must be an opening" case: turn 0 is a follow-up, which passes `_parse_turn`, so the opening check fires. The `"relevant_doc_ids"` cases hit the search/corpus check.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/measure_multi_turn_continuity.py tests/unit/test_measure_multi_turn_continuity.py docs/superpowers/specs/2026-09-24-multi-turn-continuity-eval-design.md docs/superpowers/plans/2026-09-24-multi-turn-continuity-eval.md
git commit -m "examples: multi-turn continuity eval — dataset model and validation"
```

---

### Task 2: Query conditions

**Files:**
- Modify: `examples/measure_multi_turn_continuity.py`
- Test: `tests/unit/test_measure_multi_turn_continuity.py`

**Interfaces:**
- Consumes: `Conversation`, `Turn`, `CONDITIONS`.
- Produces: `build_query(condition: str, conversation: Conversation, index: int) -> str`; `PLACEHOLDER_REPLY: str`.

- [ ] **Step 1: Write the failing tests**

```python
from examples.measure_multi_turn_continuity import build_query


def _conv(*turns: Turn) -> Conversation:
    return Conversation("c", turns)


TOKYO = Turn("weather in Tokyo", "tool", "opening", "weather in Tokyo")
PARIS = Turn("and in Paris?", "tool", "follow_up", "weather in Paris", kind="ellipsis")
BERLIN = Turn("and Berlin?", "tool", "follow_up", "weather in Berlin", kind="ellipsis")
PRICE = Turn("its price in 2020?", "tool", "follow_up", "Tesla stock price in 2020", kind="pronoun")
CUE_SWITCH = Turn("also, explain BM25", "chat", "topic_switch", "also, explain BM25")


def test_raw_and_gold_rewrite():
    conv = _conv(TOKYO, PARIS)
    assert build_query("raw", conv, 1) == "and in Paris?"
    assert build_query("gold_rewrite", conv, 1) == "weather in Paris"


def test_first_turn_is_unchanged_by_every_condition():
    conv = _conv(TOKYO, PARIS)
    for condition in ("raw", "regex", "concat", "gold_rewrite"):
        assert build_query(condition, conv, 0) == "weather in Tokyo"


def test_concat_uses_the_immediately_previous_user_turn():
    conv = _conv(TOKYO, PARIS, BERLIN)
    assert build_query("concat", conv, 2) == "and in Paris?\nand Berlin?"


def test_regex_prepends_the_most_recent_non_follow_up_turn():
    conv = _conv(TOKYO, PARIS, BERLIN)
    assert build_query("regex", conv, 2) == "weather in Tokyo\nand Berlin?"


def test_regex_leaves_an_uncued_follow_up_alone():
    conv = _conv(TOKYO, PRICE)
    assert build_query("regex", conv, 1) == "its price in 2020?"


def test_regex_false_positive_on_a_cue_word_switch():
    conv = _conv(TOKYO, CUE_SWITCH)
    assert build_query("regex", conv, 1) == "weather in Tokyo\nalso, explain BM25"


def test_unknown_condition_raises():
    with pytest.raises(ValueError, match="unknown condition"):
        build_query("oracle", _conv(TOKYO, PARIS), 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v -k "raw or regex or concat or condition or first_turn"`
Expected: FAIL, `ImportError: cannot import name 'build_query'`.

- [ ] **Step 3: Implement** (add below `load_conversations`; add the two imports at the top)

```python
from src.context import ChatMessage
from src.internal.search.context import build_retrieval_context

# No condition reads assistant text; the placeholder only keeps the history
# shaped like a real session (user/assistant alternating).
PLACEHOLDER_REPLY = "Here is what I found."


def _history(prior_texts: list[str]) -> list[ChatMessage]:
    history: list[ChatMessage] = []
    for text in prior_texts:
        history.append(ChatMessage(role="user", content=text))
        history.append(ChatMessage(role="assistant", content=PLACEHOLDER_REPLY))
    return history


def build_query(condition: str, conversation: Conversation, index: int) -> str:
    turn = conversation.turns[index]
    prior = [t.text for t in conversation.turns[:index]]
    if condition == "raw":
        return turn.text
    if condition == "gold_rewrite":
        return turn.gold_rewrite
    if condition == "concat":
        return f"{prior[-1]}\n{turn.text}" if prior else turn.text
    if condition == "regex":
        return build_retrieval_context(turn.text, _history(prior)).retrieval_query
    raise ValueError(f"unknown condition {condition!r}")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

Change `prior[-1]` to `prior[0]` → `test_concat_uses_the_immediately_previous_user_turn` must FAIL. Change the `regex` branch to `return turn.text` → both `test_regex_prepends...` and `test_regex_false_positive...` must FAIL. Revert both; confirm `git diff` shows only intended code, then re-run and see PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/measure_multi_turn_continuity.py tests/unit/test_measure_multi_turn_continuity.py
git commit -m "examples: multi-turn eval query conditions (raw/regex/concat/gold_rewrite)"
```

---

### Task 3: Per-turn scoring

**Files:**
- Modify: `examples/measure_multi_turn_continuity.py`
- Test: `tests/unit/test_measure_multi_turn_continuity.py`

**Interfaces:**
- Consumes: `build_query`, `CONDITIONS`, `HIT_K`, `MRR_K`.
- Produces:
  - `Router = Callable[[str], str]` — returns `"chat" | "search" | "tool" | "clarify"`;
  - `Retriever = Callable[[str, str], list[str]]` — `(corpus, query) -> ranked doc ids`;
  - `score_retrieval(ranked: list[str], relevant: tuple[str, ...]) -> tuple[bool, float]`;
  - `evaluate(conversations, routers: dict[str, Router], retrievers: dict[str, Retriever]) -> list[dict]`, one row per (turn, condition) with keys `conversation_id, turn_index, relation, kind, gold_route, condition, query, carried, route` (router → predicted) and `retrieval` (retriever → `{"hit5": bool, "rr10": float}`, `{}` on non-search turns).

- [ ] **Step 1: Write the failing tests**

```python
from examples.measure_multi_turn_continuity import evaluate, score_retrieval


def test_score_retrieval_hit_and_reciprocal_rank():
    assert score_retrieval(["a", "b", "c"], ("c",)) == (True, pytest.approx(1 / 3))
    assert score_retrieval(["a", "b", "c", "d", "e", "f"], ("f",)) == (False, pytest.approx(1 / 6))


def test_score_retrieval_handles_short_and_empty_rankings():
    assert score_retrieval([], ("a",)) == (False, 0.0)
    assert score_retrieval(["x"], ("a",)) == (False, 0.0)


def test_score_retrieval_ignores_ranks_past_ten():
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    assert score_retrieval(ranked, ("a",)) == (False, 0.0)


FAISS = Turn("what is FAISS", "search", "opening", "what is FAISS", corpus="demo", relevant_doc_ids=("d1",))
FAISS_TYPES = Turn(
    "which variants does it offer",
    "search",
    "follow_up",
    "FAISS index types",
    kind="pronoun",
    corpus="demo",
    relevant_doc_ids=("d2",),
)


def test_evaluate_rows_route_and_retrieval():
    conv = Conversation("c", (FAISS, FAISS_TYPES))
    routes = {"what is FAISS": "search", "FAISS index types": "search"}
    router = lambda q: routes.get(q, "clarify")  # noqa: E731
    retriever = lambda corpus, q: ["d2"] if "index types" in q else ["d1"]  # noqa: E731

    rows = evaluate([conv], {"rules": router}, {"tfidf": retriever})

    assert len(rows) == 2 * 4
    raw = next(r for r in rows if r["turn_index"] == 1 and r["condition"] == "raw")
    assert raw["route"] == {"rules": "clarify"}
    assert raw["retrieval"]["tfidf"] == {"hit5": False, "rr10": 0.0}
    assert raw["carried"] is False
    gold = next(r for r in rows if r["turn_index"] == 1 and r["condition"] == "gold_rewrite")
    assert gold["route"] == {"rules": "search"}
    assert gold["retrieval"]["tfidf"] == {"hit5": True, "rr10": 1.0}
    concat = next(r for r in rows if r["turn_index"] == 1 and r["condition"] == "concat")
    assert concat["carried"] is True


def test_evaluate_skips_retrieval_on_non_search_turns():
    conv = Conversation("c", (TOKYO, PARIS))
    calls = []
    rows = evaluate(
        [conv],
        {"rules": lambda q: "tool"},
        {"tfidf": lambda corpus, q: calls.append(q) or []},
    )
    assert calls == []
    assert all(r["retrieval"] == {} for r in rows)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v -k "score_retrieval or evaluate"`
Expected: FAIL, `ImportError`.

- [ ] **Step 3: Implement** (add `from collections.abc import Callable, Sequence` at the top)

```python
Router = Callable[[str], str]
Retriever = Callable[[str, str], list[str]]


def score_retrieval(ranked: list[str], relevant: tuple[str, ...]) -> tuple[bool, float]:
    wanted = set(relevant)
    hit = any(doc_id in wanted for doc_id in ranked[:HIT_K])
    rr = next(
        (1.0 / rank for rank, doc_id in enumerate(ranked[:MRR_K], 1) if doc_id in wanted),
        0.0,
    )
    return hit, rr


def evaluate(
    conversations: Sequence[Conversation],
    routers: dict[str, Router],
    retrievers: dict[str, Retriever],
) -> list[dict]:
    rows: list[dict] = []
    for conversation in conversations:
        for index, turn in enumerate(conversation.turns):
            for condition in CONDITIONS:
                query = build_query(condition, conversation, index)
                retrieval: dict[str, dict] = {}
                if turn.route == "search":
                    for name, retrieve in retrievers.items():
                        hit, rr = score_retrieval(
                            retrieve(turn.corpus, query), turn.relevant_doc_ids
                        )
                        retrieval[name] = {"hit5": hit, "rr10": rr}
                rows.append(
                    {
                        "conversation_id": conversation.id,
                        "turn_index": index,
                        "relation": turn.relation,
                        "kind": turn.kind,
                        "gold_route": turn.route,
                        "condition": condition,
                        "query": query,
                        # Read on topic-switch turns only, where the gold query is
                        # the text itself: any difference is prior-turn text.
                        "carried": query != turn.text,
                        "route": {name: route(query) for name, route in routers.items()},
                        "retrieval": retrieval,
                    }
                )
    return rows
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

Change `ranked[:HIT_K]` to `ranked` → `test_score_retrieval_hit_and_reciprocal_rank` must FAIL. Drop the `if turn.route == "search":` guard (retrieve always, `turn.relevant_doc_ids` empty) → `test_evaluate_skips_retrieval_on_non_search_turns` must FAIL. Revert, re-run, PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/measure_multi_turn_continuity.py tests/unit/test_measure_multi_turn_continuity.py
git commit -m "examples: multi-turn eval per-turn routing and retrieval scoring"
```

---

### Task 4: Aggregation with conversation-clustered CIs

**Files:**
- Modify: `examples/measure_multi_turn_continuity.py`
- Test: `tests/unit/test_measure_multi_turn_continuity.py`

**Interfaces:**
- Consumes: rows from `evaluate`.
- Produces: `summarize(rows, routers: Sequence[str], retrievers: Sequence[str], *, resamples: int, seed: int) -> dict` shaped `summary[slice][metric][condition] = {"mean": float, "n": int, "delta_vs_raw": {"point","low","high"} | None}`. Slices: `all`, each relation, `follow_up/<kind>`. Metrics: `route_acc:<router>`, `hit5:<retriever>`, `mrr10:<retriever>`, `carry_over` (topic-switch turns only). A slice/metric with no applicable rows is omitted.

- [ ] **Step 1: Write the failing tests**

```python
from examples.measure_multi_turn_continuity import summarize


def _row(conv, index, condition, *, relation="follow_up", kind="pronoun", predicted="search", carried=False, hit=None):
    return {
        "conversation_id": conv,
        "turn_index": index,
        "relation": relation,
        "kind": kind if relation == "follow_up" else None,
        "gold_route": "search",
        "condition": condition,
        "query": "q",
        "carried": carried,
        "route": {"rules": predicted},
        "retrieval": {} if hit is None else {"tfidf": {"hit5": hit, "rr10": 1.0 if hit else 0.0}},
    }


def _four_conditions(conv, index, gold_pred, other_pred="clarify", **kwargs):
    rows = []
    for condition in ("raw", "regex", "concat", "gold_rewrite"):
        predicted = gold_pred if condition == "gold_rewrite" else other_pred
        rows.append(_row(conv, index, condition, predicted=predicted, **kwargs))
    return rows


def test_route_accuracy_counts_clarify_as_a_miss():
    rows = _four_conditions("c1", 1, "search")
    summary = summarize(rows, ["rules"], [], resamples=50, seed=0)
    acc = summary["follow_up"]["route_acc:rules"]
    assert acc["raw"]["mean"] == 0.0
    assert acc["gold_rewrite"]["mean"] == 1.0
    assert acc["gold_rewrite"]["delta_vs_raw"]["point"] == 1.0
    assert acc["raw"]["delta_vs_raw"] is None


def test_bootstrap_resamples_conversations_not_turns():
    # One conversation, turns alternating correct/incorrect under gold_rewrite.
    # Resampling the single conversation always reproduces it, so the CI is a
    # point; resampling turns would spread it.
    rows = []
    for index in range(1, 9):
        rows += _four_conditions("c1", index, "search" if index % 2 else "chat")
    summary = summarize(rows, ["rules"], [], resamples=200, seed=0)
    delta = summary["follow_up"]["route_acc:rules"]["gold_rewrite"]["delta_vs_raw"]
    assert delta == {"point": 0.5, "low": 0.5, "high": 0.5}


def test_carry_over_is_scored_on_topic_switches_only():
    rows = []
    rows += [_row("c1", 1, c, relation="topic_switch", carried=(c == "concat")) for c in ("raw", "regex", "concat", "gold_rewrite")]
    rows += [_row("c1", 2, c, carried=True) for c in ("raw", "regex", "concat", "gold_rewrite")]
    summary = summarize(rows, ["rules"], [], resamples=50, seed=0)
    assert summary["topic_switch"]["carry_over"]["concat"]["mean"] == 1.0
    assert summary["topic_switch"]["carry_over"]["raw"]["mean"] == 0.0
    assert "carry_over" not in summary["follow_up"]
    assert summary["all"]["carry_over"]["concat"]["n"] == 1


def test_retrieval_metrics_skip_non_search_turns():
    rows = _four_conditions("c1", 1, "search", hit=True) + _four_conditions("c2", 1, "tool")
    summary = summarize(rows, ["rules"], ["tfidf"], resamples=50, seed=0)
    assert summary["all"]["hit5:tfidf"]["raw"]["n"] == 1
    assert summary["all"]["route_acc:rules"]["raw"]["n"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v -k "summar or bootstrap or carry_over or skip_non_search"`
Expected: FAIL, `ImportError`.

- [ ] **Step 3: Implement** (add the import at the top)

```python
from src.model.post_training.eval.stats import cluster_bootstrap_ci

ValueOf = Callable[[dict], "float | None"]


def _slices() -> dict[str, Callable[[dict], bool]]:
    slices: dict[str, Callable[[dict], bool]] = {"all": lambda row: True}
    for relation in RELATIONS:
        slices[relation] = lambda row, relation=relation: row["relation"] == relation
    for kind in KINDS:
        slices[f"follow_up/{kind}"] = lambda row, kind=kind: row["kind"] == kind
    return slices


def _metrics(routers: Sequence[str], retrievers: Sequence[str]) -> dict[str, ValueOf]:
    metrics: dict[str, ValueOf] = {}
    for name in routers:
        metrics[f"route_acc:{name}"] = lambda row, name=name: float(
            row["route"][name] == row["gold_route"]
        )
    for name in retrievers:
        metrics[f"hit5:{name}"] = lambda row, name=name: (
            float(row["retrieval"][name]["hit5"]) if row["retrieval"] else None
        )
        metrics[f"mrr10:{name}"] = lambda row, name=name: (
            row["retrieval"][name]["rr10"] if row["retrieval"] else None
        )
    metrics["carry_over"] = lambda row: (
        float(row["carried"]) if row["relation"] == "topic_switch" else None
    )
    return metrics


def _by_conversation(
    rows: Sequence[dict], condition: str, in_slice: Callable[[dict], bool], value_of: ValueOf
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["condition"] != condition or not in_slice(row):
            continue
        value = value_of(row)
        if value is not None:
            grouped.setdefault(row["conversation_id"], []).append(value)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _paired_difference(units: Sequence[tuple[list[float], list[float]]]) -> float:
    cond_values = [v for unit in units for v in unit[0]]
    raw_values = [v for unit in units for v in unit[1]]
    return _mean(cond_values) - _mean(raw_values)


def summarize(
    rows: Sequence[dict],
    routers: Sequence[str],
    retrievers: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> dict:
    summary: dict = {}
    for slice_name, in_slice in _slices().items():
        for metric_name, value_of in _metrics(routers, retrievers).items():
            raw = _by_conversation(rows, "raw", in_slice, value_of)
            if not raw:
                continue
            per_condition: dict = {}
            for condition in CONDITIONS:
                grouped = _by_conversation(rows, condition, in_slice, value_of)
                values = [v for conv_values in grouped.values() for v in conv_values]
                delta = None
                if condition != "raw":
                    # Units are conversations: each carries its (condition, raw)
                    # values for the same turns, so the difference stays paired.
                    units = [(grouped[conv], raw[conv]) for conv in raw]
                    point, low, high = cluster_bootstrap_ci(
                        units, _paired_difference, resamples=resamples, seed=seed
                    )
                    delta = {"point": point, "low": low, "high": high}
                per_condition[condition] = {
                    "mean": _mean(values),
                    "n": len(values),
                    "delta_vs_raw": delta,
                }
            summary.setdefault(slice_name, {})[metric_name] = per_condition
    return summary
```

Applicability is identical across conditions (it depends only on the turn), so `grouped` has the same conversations and value counts as `raw`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v`
Expected: all PASS.

- [ ] **Step 5: Mutation-check**

Make the bootstrap resample turns: replace `units = [(grouped[conv], raw[conv]) for conv in raw]` with `units = [([c], [r]) for conv in raw for c, r in zip(grouped[conv], raw[conv])]` → `test_bootstrap_resamples_conversations_not_turns` must FAIL. Make `carry_over` apply to every relation → `test_carry_over_is_scored_on_topic_switches_only` must FAIL. Revert, re-run, PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/measure_multi_turn_continuity.py tests/unit/test_measure_multi_turn_continuity.py
git commit -m "examples: multi-turn eval aggregation with conversation-clustered CIs"
```

---

### Task 5: Real routers, retrievers, corpus checks and CLI

**Files:**
- Modify: `examples/measure_multi_turn_continuity.py`
- Test: `tests/unit/test_measure_multi_turn_continuity.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `make_router(name: str, index_dir: Path = INTENT_INDEX_DIR) -> Router` (`"rules"` / `"knn"`; `SystemExit` if `knn` index does not load);
  - `load_corpora(names: set[str]) -> dict[str, list[dict]]` (`SystemExit` naming corpus and path on a missing file);
  - `check_relevant_ids(conversations, corpora: dict[str, list[dict]]) -> None` (`ValueError` listing unknown ids);
  - `make_retriever(name: str, corpora, device: str) -> Retriever` (`"tfidf"` / `"hybrid"`);
  - `format_table(summary: dict) -> str`;
  - `main(argv: Sequence[str] | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
from examples.measure_multi_turn_continuity import (
    check_relevant_ids,
    format_table,
    load_corpora,
    make_retriever,
    make_router,
)


def test_rules_router_returns_a_route_or_clarify():
    route = make_router("rules")
    assert route("hello") == "chat"
    assert route("tell me a joke") in {"chat", "search", "tool", "clarify"}


def test_knn_router_refuses_to_run_without_its_index(tmp_path):
    with pytest.raises(SystemExit, match="intent index did not load"):
        make_router("knn", index_dir=tmp_path / "missing")


def test_missing_corpus_file_exits_naming_it(monkeypatch, tmp_path):
    import src.internal.servers.retrieval.corpus_registry as registry

    manifest = {"ghost": {"path": str(tmp_path / "ghost.jsonl")}}
    monkeypatch.setattr(registry, "load_manifest", lambda: manifest)
    with pytest.raises(SystemExit, match="ghost"):
        load_corpora({"ghost"})


def test_check_relevant_ids_lists_unknown_ids():
    conv = Conversation("c", (FAISS, FAISS_TYPES))
    with pytest.raises(ValueError, match="d2"):
        check_relevant_ids([conv], {"demo": [{"id": "d1"}]})
    check_relevant_ids([conv], {"demo": [{"id": "d1"}, {"id": "d2"}]})


def test_tfidf_retriever_ranks_the_matching_doc_first():
    corpora = {
        "demo": [
            {"id": "d1", "title": "FAISS", "contents": "vector index library"},
            {"id": "d2", "title": "BM25", "contents": "sparse keyword ranking"},
        ]
    }
    retrieve = make_retriever("tfidf", corpora, device="cpu")
    assert retrieve("demo", "sparse keyword ranking")[0] == "d2"


def test_format_table_mentions_each_condition():
    rows = _four_conditions("c1", 1, "search")
    table = format_table(summarize(rows, ["rules"], [], resamples=20, seed=0))
    for condition in ("raw", "regex", "concat", "gold_rewrite"):
        assert condition in table
```

`test_rules_router_returns_a_route_or_clarify` imports `recognize_intent` but with `intent_index_path=None` never touches torch; confirm in Step 5.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v -k "router or corpus or relevant or tfidf or format_table"`
Expected: FAIL, `ImportError`.

- [ ] **Step 3: Implement** (add `import argparse` and `from dataclasses import dataclass, replace` at the top)

```python
def make_router(name: str, index_dir: Path = INTENT_INDEX_DIR) -> Router:
    from src.internal.configs import load_app_settings
    from src.internal.servers.web.intent import similarity
    from src.internal.servers.web.intent.recognizer import recognize_intent

    if name not in ("rules", "knn"):
        raise ValueError(f"unknown router {name!r}")
    settings = replace(
        load_app_settings(),
        intent_index_path=index_dir if name == "knn" else None,
        intent_shadow_mode=False,
        route_clarification=True,
    )
    if name == "knn" and similarity.load_intent_index(settings) is None:
        raise SystemExit(f"knn router: intent index did not load from {index_dir}")

    def route(query: str) -> str:
        decision = recognize_intent(query, llm=None, explicit_source=False, settings=settings)
        return "clarify" if decision.clarification is not None else decision.strategy.value

    return route


def load_corpora(names: set[str]) -> dict[str, list[dict]]:
    from src.internal.servers.retrieval.corpus_registry import (
        load_manifest,
        resolve_corpus_docs,
    )

    manifest = load_manifest()
    corpora: dict[str, list[dict]] = {}
    for name in sorted(names):
        entry = manifest.get(name)
        if entry is None:
            raise SystemExit(f"corpus {name!r} is not registered in data/corpora.json")
        path = Path(entry["path"] if isinstance(entry, dict) else entry)
        # resolve_corpus_docs only warns and skips a missing file, which would
        # score fewer turns instead of failing.
        if not path.exists():
            raise SystemExit(f"corpus {name!r} missing: {path}")
        corpora[name] = resolve_corpus_docs(name, manifest)
    return corpora


def check_relevant_ids(
    conversations: Sequence[Conversation], corpora: dict[str, list[dict]]
) -> None:
    known = {name: {str(d.get("id")) for d in docs} for name, docs in corpora.items()}
    missing = sorted(
        f"{conv.id} turn {i}: {doc_id}"
        for conv in conversations
        for i, turn in enumerate(conv.turns)
        if turn.corpus
        for doc_id in turn.relevant_doc_ids
        if doc_id not in known.get(turn.corpus, set())
    )
    if missing:
        raise ValueError("relevant_doc_ids not in corpus: " + "; ".join(missing))


def make_retriever(name: str, corpora: dict[str, list[dict]], device: str) -> Retriever:
    from src.internal.servers.retrieval.demo import TfidfRetriever

    if name not in ("tfidf", "hybrid"):
        raise ValueError(f"unknown retriever {name!r}")
    sparse = {corpus: TfidfRetriever.from_docs(docs) for corpus, docs in corpora.items()}
    dense = {}
    if name == "hybrid":
        # Built directly, not via hybrid._build_dense: that one degrades to
        # TF-IDF on failure, which would make `hybrid` silently equal `tfidf`.
        from src.internal.servers.retrieval.hybrid import (
            DenseEmbeddingRetriever,
            build_e5_encoder,
        )

        encoder = build_e5_encoder(device=device)
        dense = {c: DenseEmbeddingRetriever(docs, encoder=encoder) for c, docs in corpora.items()}

    def retrieve(corpus: str, query: str) -> list[str]:
        if name == "hybrid":
            from src.internal.servers.retrieval.hybrid import _fuse_rows

            # Mirrors the hybrid server: each leg fetches 2x, RRF keeps MRR_K.
            fetch_k = MRR_K * 2
            rows = _fuse_rows(
                dense[corpus].retrieve([query], topk=fetch_k),
                sparse[corpus].retrieve([query], topk=fetch_k),
                MRR_K,
            )[0]
        else:
            rows = sparse[corpus].retrieve([query], topk=MRR_K)[0]
        return [str(item["document"]["id"]) for item in rows]

    return retrieve


def format_table(summary: dict) -> str:
    lines: list[str] = []
    for slice_name in ("all", "follow_up", "topic_switch"):
        for metric_name, per_condition in summary.get(slice_name, {}).items():
            cells = []
            for condition in CONDITIONS:
                entry = per_condition[condition]
                cell = f"{condition} {entry['mean']:.2f}"
                delta = entry["delta_vs_raw"]
                if delta is not None:
                    cell += f" ({delta['point']:+.2f} [{delta['low']:+.2f},{delta['high']:+.2f}])"
                cells.append(cell)
            n = per_condition["raw"]["n"]
            lines.append(f"{slice_name:<13} {metric_name:<16} n={n:<4} " + " | ".join(cells))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--routers", nargs="+", default=["rules", "knn"], choices=["rules", "knn"])
    parser.add_argument("--retrievers", nargs="+", default=["tfidf", "hybrid"], choices=["tfidf", "hybrid"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    conversations = load_conversations(args.data)
    corpora = load_corpora({t.corpus for c in conversations for t in c.turns if t.corpus})
    check_relevant_ids(conversations, corpora)
    routers = {name: make_router(name) for name in args.routers}
    retrievers = {name: make_retriever(name, corpora, args.device) for name in args.retrievers}

    rows = evaluate(conversations, routers, retrievers)
    summary = summarize(rows, args.routers, args.retrievers, resamples=args.resamples, seed=args.seed)
    print(format_table(summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "config": {**{k: str(v) for k, v in vars(args).items()}, "conversations": len(conversations)},
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v`
Expected: all PASS.

- [ ] **Step 5: Torch-free check** (the CI unit job has no torch)

Run:
```bash
python - <<'EOF'
import sys, pytest
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith(("torch.", "sentence_transformers", "transformers")):
            raise ImportError(f"blocked {name}")
sys.meta_path.insert(0, _Block())
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "tests/unit/test_measure_multi_turn_continuity.py"]))
EOF
```
Expected: all pass (none skipped). If an import fails, move it inside the function that needs it.

- [ ] **Step 6: Mutation-check**

Delete the `if name == "knn" and similarity.load_intent_index(...) is None` guard → `test_knn_router_refuses_to_run_without_its_index` must FAIL. Make `check_relevant_ids` return early → `test_check_relevant_ids_lists_unknown_ids` must FAIL. Delete the `if not path.exists()` check → `test_missing_corpus_file_exits_naming_it` must FAIL (the registry only warns). Revert; delete any `__pycache__` for the module; re-run, PASS.

- [ ] **Step 7: Lint and commit**

```bash
ruff check examples/measure_multi_turn_continuity.py tests/unit/test_measure_multi_turn_continuity.py --fix && ruff format examples/measure_multi_turn_continuity.py tests/unit/test_measure_multi_turn_continuity.py
git add examples/measure_multi_turn_continuity.py tests/unit/test_measure_multi_turn_continuity.py
git commit -m "examples: multi-turn eval routers, retrievers, corpus checks and CLI"
```

---

### Task 6: Author the dataset (before any real run)

**Files:**
- Create: `data/eval/multi_turn_conversations.jsonl` (force-added)
- Test: `tests/unit/test_measure_multi_turn_continuity.py` (quota test on the real file)

**Interfaces:**
- Consumes: `load_conversations`; `src.internal.search.context._is_follow_up`.

Do **not** run `main` or any router/retriever on the dataset until it is committed.

- [ ] **Step 1: Write the failing quota test**

```python
from collections import Counter

from examples.measure_multi_turn_continuity import DEFAULT_DATA
from src.internal.search.context import _is_follow_up


def test_committed_dataset_meets_the_spec_composition():
    conversations = load_conversations(DEFAULT_DATA)
    turns = [t for c in conversations for t in c.turns]
    follow_ups = [t for t in turns if t.relation == "follow_up"]
    switches = [t for t in turns if t.relation == "topic_switch"]

    assert len(conversations) >= 40
    assert all(2 <= len(c.turns) <= 4 for c in conversations)
    openings = Counter(
        c.turns[0].corpus if c.turns[0].route == "search" else c.turns[0].route
        for c in conversations
    )
    assert openings["demo"] >= 8 and openings["scifact"] >= 8
    assert openings["tool"] >= 8 and openings["chat"] >= 6
    assert len(follow_ups) >= 40
    assert all(n >= 6 for n in Counter(t.kind for t in follow_ups).values())
    assert set(Counter(t.kind for t in follow_ups)) == {"pronoun", "ellipsis", "comparison", "elaboration"}
    uncued = [t for t in follow_ups if not _is_follow_up(t.text)]
    assert len(uncued) * 2 >= len(follow_ups)
    assert len(switches) >= 12
    assert sum(_is_follow_up(t.text) for t in switches) >= 4
    route_changes = sum(
        1
        for c in conversations
        for prev, turn in zip(c.turns, c.turns[1:])
        if turn.relation == "follow_up" and turn.route != prev.route
    )
    assert route_changes >= 6


def test_scifact_gold_labels_match_beir_qrels():
    qrels_dir = Path("data/beir/scifact/qrels")
    if not qrels_dir.exists():
        pytest.skip("BEIR SciFact not downloaded")
    qrels: dict[str, set[str]] = {}
    for split in ("train.tsv", "test.tsv"):
        for line in (qrels_dir / split).read_text().splitlines()[1:]:
            query_id, doc_id, _ = line.split("\t")
            qrels.setdefault(query_id, set()).add(doc_id)
    queries = {
        q["_id"]: q["text"]
        for q in map(json.loads, Path("data/beir/scifact/queries.jsonl").read_text().splitlines())
    }
    for conv in load_conversations(DEFAULT_DATA):
        for turn in conv.turns:
            if turn.corpus == "scifact":
                assert turn.beir_query_id, f"{conv.id}: scifact turn without beir_query_id"
                assert turn.gold_rewrite == queries[turn.beir_query_id], conv.id
                assert set(turn.relevant_doc_ids) == qrels[turn.beir_query_id], conv.id
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v -k "committed_dataset or scifact_gold"`
Expected: FAIL, `FileNotFoundError` for `data/eval/multi_turn_conversations.jsonl`.

- [ ] **Step 3: Find SciFact query pairs that share an entity**

```bash
python - <<'EOF'
import json, re
from collections import defaultdict
queries = {q["_id"]: q["text"] for q in map(json.loads, open("data/beir/scifact/queries.jsonl"))}
labelled = set()
for split in ("train", "test"):
    for line in open(f"data/beir/scifact/qrels/{split}.tsv").read().splitlines()[1:]:
        labelled.add(line.split("\t")[0])
by_term = defaultdict(list)
for qid in labelled:
    for term in set(re.findall(r"\b[A-Z][A-Za-z0-9-]{2,}\b", queries[qid])):
        by_term[term].append(qid)
for term, ids in sorted(by_term.items(), key=lambda kv: -len(kv[1])):
    if 2 <= len(ids) <= 6:
        print(term, "|", " || ".join(f"{i}: {queries[i]}" for i in ids))
EOF
```

Pick at least 8 pairs whose two queries share an entity (a gene, drug, condition). Turn 1 = query A verbatim (`opening`); turn 2 = query B with the entity replaced by a reference ("it", "this protein", or dropped), `relation: follow_up`, `gold_rewrite` = query B verbatim, `beir_query_id` = B's id, `relevant_doc_ids` = B's qrels. Put the pair's `beir_query_id` and qrels on turn 1 too.

- [ ] **Step 4: Write the dataset**

One JSON object per line: `{"id": "...", "turns": [...]}` using the Task 1 schema. Id prefixes by opening family: `demo-NN`, `sci-NN`, `tool-NN`, `chat-NN`. Meet every quota in Step 1, and follow the spec's composition rules:

- **demo search** (≥8): topics from `data/corpus.jsonl` titles (FAISS, BM25, RAG, reranking, e5, BGE, DPR, MS MARCO…); `relevant_doc_ids` = the doc whose title the gold rewrite is about. Include comparisons ("how does it compare with BM25?" → gold "compare dense retrieval with FAISS and BM25 sparse retrieval", relevant both docs).
- **SciFact** (≥8): Step 3 pairs.
- **tool** (≥8): weather, stocks, currency, geocoding. Ellipsis chains ("weather in Tokyo" → "and in Paris?" → "Berlin?"), pronoun ("what's its market cap?").
- **chat** (≥6): explanations/writing, with elaboration follow-ups ("why?", "shorter please", "give an example").
- **route changes** (≥6 follow-ups): e.g. demo search → "summarise that in one line" (chat); stock price → "why did it drop?" (search? no corpus doc exists → label `chat`).
- **topic switches** (≥12), of which ≥4 start with a cue word ("also, …", "and what's …", "how about we switch to …").
- ≥ half of follow-ups must be uncued: `_is_follow_up(text)` false.

Label each turn's `route` by what a person would want, never by what the router says.

- [ ] **Step 5: Validate**

Run: `pytest tests/unit/test_measure_multi_turn_continuity.py -v`
Then: `python -c "from examples.measure_multi_turn_continuity import *; from pathlib import Path; c=load_conversations(DEFAULT_DATA); check_relevant_ids(c, load_corpora({t.corpus for x in c for t in x.turns if t.corpus})); print(len(c), 'ok')"`
Expected: all PASS; prints `<N> ok`. (These load data only; no router or retriever runs.)

- [ ] **Step 6: Commit the dataset before running anything**

```bash
git add -f data/eval/multi_turn_conversations.jsonl
git add tests/unit/test_measure_multi_turn_continuity.py
git commit -m "data: 40+ hand-written multi-turn conversations for the continuity eval"
```

---

### Task 7: Run the eval and record results

**Files:**
- Create: `data/eval/multi_turn_continuity.json` (force-added)
- Modify: `docs/superpowers/specs/2026-09-24-multi-turn-continuity-eval-design.md` (append `## Results`)

- [ ] **Step 1: Run**

```bash
caffeinate -i python -m examples.measure_multi_turn_continuity 2>&1 | tee /tmp/multi_turn_run.log
```
Expected: tables for `all`, `follow_up`, `topic_switch`; `wrote data/eval/multi_turn_continuity.json`. Hybrid over SciFact embeds 5,183 docs once (a few minutes on MPS).

- [ ] **Step 2: Sanity checks before believing anything**

- `gold_rewrite` route accuracy on `opening` rows equals `raw` (they are the same string).
- `concat` carry-over on `topic_switch` is 1.00.
- `knn` and `rules` rows differ somewhere (else the index did not engage).
- Spot-read 5 `raw` follow-up rows with `hit5: false` and confirm the query really lacks the entity.

If any fails, stop and debug (superpowers:systematic-debugging), do not report numbers.

- [ ] **Step 3: Append `## Results` to the spec**

Record, with CIs: follow-up route accuracy per router per condition; follow-up Hit@5 per retriever per condition; topic-switch Hit@5 and carry-over; the per-kind breakdown's largest gaps; and the decision the numbers imply for the fix spec (does `regex`/`concat` close most of `raw`→`gold_rewrite`, or is an LLM rewriter needed; does routing need continuity). State the caveat: one author wrote conversations and gold rewrites.

- [ ] **Step 4: Commit and open the PR**

```bash
git add -f data/eval/multi_turn_continuity.json
git add docs/superpowers/specs/2026-09-24-multi-turn-continuity-eval-design.md
git commit -m "examples: multi-turn continuity eval results"
git push -u origin feat/multi-turn-continuity-eval
gh pr create --title "Measure routing and retrieval on follow-up turns and topic switches" --body "<summary of results + test plan>"
```

- [ ] **Step 5: Verify the merge carries every commit** (squash has dropped late commits here before): after merge, `git diff <branch> origin/main -- examples tests data/eval docs` must be empty.
