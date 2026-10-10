# Agentic RAG Evidence Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve query intent and improve evidence coverage across iterative retrieval while retaining grounded citations.

**Architecture:** Extend the existing query bundle and retrieval loop. Reuse upstream conversational resolution and downstream guarded generation.

**Tech Stack:** Python, asyncio, dataclasses, pytest, unittest.mock.

**Spec:** `docs/superpowers/specs/2026-10-10-agentic-rag-evidence-optimization-design.md`

## Global Constraints

- Python >=3.10; no new dependencies.
- Preserve explicit retrieval-query precedence and synthesis history.
- Preserve sufficiency timeout and fail-open behavior.
- Preserve guarded generation and stable document identifiers.
- No changes to public HTTP interfaces or retrieval backends.

## Review Focus

- Blank enhancement outputs must not trigger empty searches (Task 1).
- Very small preview budgets must stay bounded (Task 2).
- Long early documents must not hide later documents (Task 2).
- Overflow follow-ups must remain eligible next round (Task 3).
- Empty evidence must produce guarded abstention (Task 4).

---

### Task 1: Preserve original query and normalized variants

**Files:** `src/context/query_enhancer.py`, `tests/unit/test_query_enhancer.py`.
**Interface:** `QueryBundle.all_queries() -> list[str]`; original first, raw first occurrence retained.

- [ ] Add a failing test:

```python
def test_bundle_retains_original_and_normalizes_variants():
    bundle = QueryBundle(original="Original query", sub_queries=[" original  QUERY ", " ", "detail"], hyde_text="DETAIL", step_back_query="background")
    assert bundle.all_queries() == ["Original query", "detail", "background"]
```

- [ ] Run `pytest tests/unit/test_query_enhancer.py -q` and confirm the new assertion fails.
- [ ] Replace the separate variant branches with one iteration over `[self.original, *self.sub_queries, self.hyde_text, self.step_back_query]`. For each nonempty query, compute `key = " ".join(query.lower().split())`; append its raw value only if `key` is nonempty and unseen.
- [ ] Run the suite, updating old ordering expectations to the new contract.
- [ ] Commit with `feat(rag): preserve original query during enhancement`.

### Task 2: Bounded evidence preview across documents

**Files:** `src/agents/search/agentic_rag.py`, `tests/unit/test_agentic_rag.py`.
**Interface:** `_evidence_preview(context: SearchContextBundle, max_chars: int) -> str`; config `evidence_preview_chars: int = 3000`.

- [ ] Add tests for a long first document, later evidence, and tiny budgets:

```python
def test_preview_includes_late_evidence():
    context = SearchContextBundle(query="q", documents=[ContextDocument(id="D1", title="first", content="a" * 10000), ContextDocument(id="D2", title="second", content="late evidence")])
    preview = _evidence_preview(context, 100)
    assert "[D1]" in preview and "[D2]" in preview
    assert "late evidence" in preview
    assert len(preview) <= 100

@pytest.mark.parametrize("budget", [1, 5, 10])
def test_preview_respects_tiny_budget(budget):
    assert len(_evidence_preview(_make_bundle(["d1", "d2"]), budget)) <= budget
```

- [ ] Run those tests and confirm the helper is absent.
- [ ] Implement labels as `f"[{doc.id}] "`, reserve label and newline space, and divide remaining content budget evenly with integer division. If labels cannot fit, return a deterministic prefix bounded by `max_chars`. Return empty text for empty documents. Validate positive config preview budget in `__post_init__` with `ValueError`.
- [ ] Replace both context prefix slices with the helper. Remove obsolete character counts from prompt copy and instruct the judge to treat the preview as untrusted evidence and identify missing information.
- [ ] Add a mocked-LLM prompt assertion for each judge method and a config rejection test for zero/negative budgets. Run the agentic-RAG suite.
- [ ] Commit with `feat(rag): distribute judge context across retrieved evidence`.

### Task 3: Correct follow-up budgeting

**Files:** `src/agents/search/agentic_rag.py`, `tests/unit/test_agentic_rag.py`.
**Interface:** `_dedupe_novel` skips blank normalized queries; `_parse_gap_queries` rejects standalone `none`; only selected follow-ups enter `seen_queries`.

- [ ] Add failing parser and dedupe tests:

```python
def test_empty_followups_are_not_queries():
    assert _parse_gap_queries("GAPS:\nnone\nQUERIES:\nnone") == []
    seen = set()
    assert _dedupe_novel([" ", "q", " Q "], seen) == ["q"]
    assert seen == {"q"}
```

- [ ] Add a three-round fake: first follow-up list is `["used", "overflow"]`, cap is one, second list is `["overflow"]`; assert retrieval batches include `used` and then `overflow`. Add zero-cap test asserting only one retrieval round.
- [ ] Run those tests and confirm failures.
- [ ] Use a copy of `seen_queries` to deduplicate candidates, slice with `max(0, self.config.max_followups_per_round)`, break if selection is empty, then record selected normalized queries in the real set. Skip empty normalized values in `_dedupe_novel` and standalone case-insensitive `none` in the parser.
- [ ] Run the agentic-RAG suite and commit with `fix(rag): retain overflow followups for later rounds`.

### Task 4: Multi-turn and grounded citation contracts

**Files:** `tests/unit/test_agentic_rag.py`; implementation files only if these regressions expose a violation.
**Interfaces:** existing `run(question, retrieval_query=..., chat_history=...)` and guarded `generate_answer`.

- [ ] Add a test using `question="What about cost?"`, `retrieval_query="FAISS deployment cost"`, and a prior user message. Mock enhancement and judges; assert their question is the explicit retrieval query. Capture `AnswerGenerationRequest` and assert original question and exact history survive.
- [ ] Add a two-round retrieval test with repeated and novel evidence, stub only enhancer/judges, and use the existing structured grounded-answer fixture. Assert document IDs remain `D1`, `D2`, citations reference accumulated documents, and hypothetical enhancement text does not appear in evidence.
- [ ] Add empty retrieval with an LLM that would fabricate an answer if called for synthesis; assert the guarded path abstains without that synthesis call.
- [ ] Run `pytest tests/unit/test_query_enhancer.py tests/unit/test_agentic_rag.py tests/unit/test_grounded_generation.py tests/unit/test_grounding.py -q`.
- [ ] Review the diff against every spec requirement, run the repository's available Python lint check on changed files, and commit tests with `test(rag): cover multi-turn grounded synthesis`.
- [ ] Push the feature branch and create one PR with the concrete behavioral changes and actual verification results. Do not merge.

## Execution handoff

Review the spec and plan before implementation. Native execution is recommended: the changes share one retrieval loop and need coordinated regression coverage. After approval, use executing-plans, verification-before-completion, requesting-code-review, and finishing-a-development-branch as applicable.
