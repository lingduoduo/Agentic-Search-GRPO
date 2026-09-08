# Intent Recognition Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate serving-time intent recognition behind one result-bearing entry point and open a tested PR.

**Architecture:** A web intent package separates dependency-free types, rules, similarity adaptation and cascade orchestration. One RouteDecision carries strategy, clarification and metadata. Offline index construction and evaluation retain their current package.

**Tech Stack:** Python >=3.10, dataclasses, pytest, FastAPI, NumPy, optional sentence-transformers, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-08-intent-recognition-consolidation-design.md`

## Global Constraints

- Preserve external API fields, route values, explicit overrides, capability degradation and authorization.
- Defaults remain top_k=8, min_route_margin=0.010, min_module_score=0.8215; no encoder, index-format or data changes.
- Exactly one deciding intent capture stage; at most one model evaluation stage; no query/arbitrary completion in routing telemetry.
- No torch/sentence-transformers requirement on package import, and no encoder load until prediction.
- Modules and composite remain diagnostics only; no multi-intent execution planner.
- Work only in `/Users/linghuang/Git/Agentic-Search-intent-consolidation`, branch `refactor/consolidate-intent-recognition`.
- Use `/Users/linghuang/miniconda3/bin/python` for local tests; bare `python` is unavailable in the worktree shell.

---

### Task 1: Consolidate the runtime and migrate consumers with regression coverage

**Files:**
- Create: `src/internal/servers/web/intent/{__init__,types,rules,similarity,recognizer}.py`
- Delete: `src/internal/servers/web/intent_routing.py`, `src/internal/servers/web/ml_intent.py`
- Modify: `src/internal/servers/web/app.py`, `src/internal/servers/web/tool_agent_runner.py`
- Modify: all Python callers/imports in `src/`, `examples/`, and `tests/` referring to the removed modules or web route entry points.
- Test: `tests/unit/test_intent_routing.py`, `tests/unit/test_ml_intent.py`, `tests/unit/servers/web/test_agent_router.py`, `tests/unit/servers/web/test_stage_emits_intent.py`, and affected web dispatcher tests.

**Interfaces:**
- Consumes: existing AppSettings, IntentIndex, request_capture.record_stage, LLMClient.complete.
- Produces: `recognize_intent(query: str, *, llm: LLMClient | None, explicit_source: bool, settings: AppSettings | None = None) -> RouteDecision` in the public `web.intent` package.
- Produces: `RouteDecision(strategy: RouteStrategy, clarification: Clarification | None = None, metadata: dict = field(default_factory=dict))`.
- Internal functions retain useful names (`classify_route`, `predict_route`, `_regex_route`, `_rule_based_route_or_none`) in their owning submodule; there is no compatibility shim for old modules or route_query.

- [ ] **Step 1: Add failing behavioral regressions to the existing router tests.** Reuse the existing `_FakeLLM` and call `ir.route_request` before migration. These tests catch wrong rule branches and label-order guesses:

```python
@pytest.mark.parametrize("query", ["hi", "HI!", "hello", "hi there", "thanks", "thank you."])
def test_standalone_greeting_is_chat_without_clarification(query):
    decision = ir.route_request(query, llm=None, explicit_source=False)
    assert decision.strategy is RouteStrategy.CHAT
    assert decision.clarification is None

@pytest.mark.parametrize("reply", ["not chat; search", "search or tool", "chat, search, tool"])
def test_classifier_rejects_conflicting_labels(reply):
    strategy, detail = classify_route("query", _FakeLLM(reply))
    assert strategy is None
    assert detail == {"raw_label": "unexpected"}
```

Run `/Users/linghuang/miniconda3/bin/python -m pytest -q -o addopts='' tests/unit/servers/web/test_agent_router.py -k 'standalone_greeting or conflicting_labels'`; record the expected assertion failures.

- [ ] **Step 2: Extract types, rules and similarity; implement regression fixes.** Move existing code into the specified owning modules, remove circular imports and retain lazy dependency behavior. Use whole-utterance greeting matching before bare lookup, and exclude greeting terms from bare lookups. Parse distinct labels rather than enum-order matches:

```python
labels = {value for value in _LABEL_BY_VALUE if re.search(rf"\b{value}\b", content)}
if len(labels) == 1:
    captured_label = labels.pop()
    strategy = _LABEL_BY_VALUE[captured_label]
else:
    captured_label = "unexpected"
```

Add negative greeting examples (`hello world tutorial`, `hi, find the report`) and retain single-label explanatory completion/redaction tests.

- [ ] **Step 3: Replace the cascade interface and migrate callers.** Resolve settings once, accumulate metadata locally, and return it from every decision path; capture stages stay in the recognizer. Preserve model/shadow/abstention diagnostics and error fallback behavior. Dispatcher integration is:

```python
decision = recognize_intent(query, llm=llm, explicit_source=explicit_source, settings=app_settings)
extra.update(decision.metadata)
```

Use `decision.strategy` for consumers previously using `route_query`. Update monkeypatch targets to defining submodules and `app.recognize_intent`. Keep existing test assertions on actual routes, capture counts and metadata. Move `_infer_intent_from_output` to `tool_agent_runner.py` and update its tests. Remove unused strategy-only helpers if they have no production caller, migrating their tests to the result-bearing interface.

- [ ] **Step 4: Verify import and default-settings boundaries.** Add fresh-subprocess import checks in both public-package-first and similarity-first order with a meta-path blocker for torch and sentence_transformers; import success plus a real greeting recognition must be asserted. Add no-explicit-settings tests that patch configuration loading and exercise shadow mode and disabled clarification through the public entry point. Assert returned model metadata agrees with the real capture stage on served, abstained and shadow cases, reusing existing fixtures where possible.

- [ ] **Step 5: Run tests, lint and commit the runtime change.** Run:

```bash
/Users/linghuang/miniconda3/bin/python -m pytest -q -o addopts='' tests/unit/test_intent_model.py tests/unit/test_ml_intent.py tests/unit/test_intent_routing.py tests/unit/servers/web/test_agent_router.py tests/unit/servers/web/test_stage_emits_intent.py tests/unit/test_execution_fallbacks.py
/Users/linghuang/miniconda3/bin/python -m ruff check src/internal/servers/web/intent tests/unit/servers/web/test_agent_router.py
/Users/linghuang/miniconda3/bin/python -m ruff format --check src/internal/servers/web/intent tests/unit/servers/web/test_agent_router.py
```

Run the affected web-server suite once after integration, record results and environmental failures precisely, and self-review the diff. Commit only task-owned Python files with `refactor(intent): consolidate serving recognition and decision metadata`.

### Task 2: Align live documentation and prepare the reviewed PR

**Files:**
- Modify: `docs/request-routing.md`, `docs/training-and-evaluation.md`, `docs/configuration.md` where current guidance is stale.
- Modify: other live documentation references to removed runtime modules as needed; historical specs/plans remain historical.

**Interfaces:**
- Consumes: Task 1's public `web.intent.recognize_intent`, returned RouteDecision.metadata, unchanged API response fields and serving settings.
- Produces: consistent current routing/ownership/configuration documentation. PR creation is performed by the controller after review and validation.

- [ ] **Step 1: Update routing and metadata documentation.** Describe `recognize_intent`, explicit-source semantics (`!= auto` forces search), the margin-only model gate, optional index path, shadow mode, and clarification behavior. Replace the old confidence-gate example with:

```text
AGENTIC_SEARCH_INTENT_INDEX_PATH=data/intent_index
AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN=0.010
AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE=0.8215
AGENTIC_SEARCH_INTENT_TOP_K=8
```

List existing model/shadow metadata accurately; remove claims about `route_threshold`, `model_below_threshold`, and confidence-floor settings in current serving guidance. Update source-ownership table to the new package. Clarify that selected route and executed intent can differ after degradation.

- [ ] **Step 2: Reconcile current training/evaluation guidance.** Correct present-tense top_k=15 claims to 8 while keeping historical comparisons clearly historical. State that offline CLI/index/data paths remain unchanged and module/composite diagnostics do not invoke a planner. Do not claim any new accuracy measurements. Update live links/references to the relocated adapter/rules.

- [ ] **Step 3: Check links and commit.** Inspect `git diff --check` and all changed Markdown paths/links; no prose-mirroring tests. Commit documentation with `docs(intent): align routing guide with consolidated recognition`.

- [ ] **Step 4: Independent review, verification and PR (controller).** Review Task 1 and Task 2 for spec compliance/code quality, then review the entire branch. Resolve material findings, run appropriate final checks and update this plan with actual results. Push only `refactor/consolidate-intent-recognition` and create a PR against main, using an exact body file containing problem, behavior, spec/plan links and validation limitations. Do not merge.
