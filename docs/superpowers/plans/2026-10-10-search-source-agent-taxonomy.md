# Search-source Agent Taxonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace the sixteen topic domains plus general with the twenty approved source categories while preserving legacy input compatibility.

**Architecture:** Keep one ordered DOMAIN_REGISTRY in the existing search module. Normalize selector aliases and capability tags at entry points; discovery and responses use canonical names. Expose typical source metadata without claiming new provider integrations.

**Tech Stack:** Python, pytest, FastAPI/Pydantic, React/TypeScript, Vitest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-10-search-source-agent-taxonomy-design.md`

## Global Constraints

- Exactly twenty canonical categories in the spec's order; default web has an empty hint.
- Retain domain field and endpoint names; advertise only canonical categories.
- Accept legacy selectors and capability tags. Preserve provider selection, filters, limits, lifecycle cleanup, errors, and timeout policies.
- Typical sources are descriptive metadata, not newly integrated connectors.
- Do not add autonomous agents, fan-out, or automatic multi-source routing.
- Preserve historical specs/plans. No live-source quality or latency claims. Open a separate PR against main and verify its final CI; do not merge.

## Review Focus

- Omitted selectors in HTTP, MCP, tools and UI must all mean web; Task 1/2 tests cover defaults.
- Alias input must survive schema validation before normalization; Task 1 exercises execution through real tool schemas.
- environment.weather migrates to local.weather, while environment.web migrates to web.web; Task 1 tests both paths and conflicting selectors.
- Existing public-data routes must invoke the same tools/arguments and release instances on failure; Task 1 retains lifecycle tests.
- Discovery failure must leave a working web selector; Task 2 tests UI fallback and canonical request payloads.
- Old evaluation fixtures must not silently acquire new relevance meanings; Task 3 distinguishes historical topic evaluation from canonical source discovery.

### Task 1: Canonical registry and compatible tool dispatch

**Files:** Modify `src/internal/tools/search.py`, `src/internal/mcp_server/tools/search.py`; test `tests/unit/test_search_domains.py`, `tests/unit/test_domain_taxonomy.py`, `tests/unit/test_domain_search.py`, `tests/unit/test_search_domain_tools.py`, `tests/unit/test_domain_search_tools.py`, `tests/unit/test_domain_search_mcp.py`.

**Interfaces:** Preserve SearchDomain(description, query_hint, capabilities); append typical_sources: tuple[str, ...] = (). Keep normalize_search_domain(value: str = "web") -> str and prepare_domain_query(query: str, domain: str = "web") -> str. Add normalize_search_tag(tag: str) -> str for dispatch. search_domain_parameter() advertises canonical enum/default; tool execution normalizes aliases before enum validation wherever validation occurs.

- [x] Write failing tests for exact ordered canonical IDs and spec metadata, all selector aliases, default web query unchanged, and legacy capability migrations. Use parameterized tests with explicit mappings, including:

```python
@pytest.mark.parametrize("value,expected", [
    ("general", "web"), ("finance", "financial"), ("ip", "patent"),
    ("environment", "web"), ("film", "media"),
])
def test_legacy_selector_alias(value, expected):
    assert normalize_search_domain(value) == expected

@pytest.mark.parametrize("value,expected", [
    ("general.wikipedia", "web.wikipedia"),
    ("finance.quote", "financial.quote"),
    ("environment.weather", "local.weather"),
    ("environment.web", "web.web"),
])
def test_legacy_tag_alias(value, expected):
    assert normalize_search_tag(value) == expected
```

Add tests invoking actual tool.execute with legacy selectors, not just the normalization helper. Assert unknown selectors fail before provider invocation and supplied canonical domain agrees with normalized tag. Keep real stock-tool mocks and cancellation/cleanup coverage.

- [x] Run `pytest tests/unit/test_search_domains.py tests/unit/test_domain_taxonomy.py tests/unit/test_domain_search.py tests/unit/test_search_domain_tools.py tests/unit/test_domain_search_tools.py tests/unit/test_domain_search_mcp.py -q`; confirm failures reflect missing taxonomy/normalization.
- [x] Implement the twenty registry entries from the spec table, including exact topics and source metadata. Use aliases below; do not duplicate legacy keys in DOMAIN_REGISTRY:

```python
DOMAIN_ALIASES = {
    "general": "web", "finance": "financial", "ip": "patent",
    "health": "web", "security": "web", "energy": "web",
    "environment": "web", "agriculture": "web", "gaming": "web",
    "film": "media",
}

def normalize_search_tag(tag: str) -> str:
    if not isinstance(tag, str) or "." not in tag:
        raise InvalidToolInput("tag must be a capability returned by get_sub_domains")
    prefix, capability = tag.split(".", 1)
    normalized_prefix = prefix.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized_prefix == "environment" and capability == "weather":
        return "local.weather"
    return f"{normalize_search_domain(prefix)}.{capability}"
```

Keep tool-less web capability in every entry. Move wikipedia to web, quote/crypto/currency to financial, weather to local; retain travel location/nearby and add these routes under local. Preserve academic arxiv and resource wayback. Normalize tags before deriving omitted domain, and compare canonical domain against canonical tag. Serialize typical_sources as a list in capability directories. Update tool descriptions/defaults and MCP defaults. If a wrapper validates enum inputs before execute, normalize domain arguments at that boundary; test the real path instead of weakening unknown-input checks.
- [x] Re-run the Task 1 suite plus `tests/unit/test_search_query.py`; expected all pass. MCP coverage is included in `tests/unit/test_domain_search_mcp.py` above.
- [x] Commit `feat: migrate search tools to source-agent taxonomy`.

### Task 2: HTTP discovery and Assist selection

**Files:** Modify `src/internal/servers/web/app.py`, `web/src/types.ts`, `web/src/pages/AssistPage.tsx`, `web/src/components/SearchComposer.tsx`; test `tests/unit/servers/web/test_search_domain_http.py`, `web/src/pages/__tests__/AssistPage.test.tsx`, `web/src/components/__tests__/SearchComposer.test.tsx`, `web/src/__tests__/api.test.ts`.

**Interfaces:** GET /api/search-domains remains {domains: [{name: str, description: str, typical_sources: list[str]}]}; TypeScript SearchDomainOption adds typical_sources?: string[] to preserve existing client fixtures. Request domain defaults to web. Existing supported-mode policy remains, with canonical web treated as neutral.

- [x] Write failing HTTP tests for default web, twenty-category order, source arrays, legacy inputs accepted with canonical responses, and rejection of non-web categories in unsupported modes. Assert web/general keep corpus queries unchanged. Add UI tests for web fallback when discovery fails and financial selection reaching request payloads.

```python
def test_request_default_is_web():
    assert AgentExperienceRequest(query="q").domain == "web"

def test_source_metadata_is_exposed(client):
    domains = client.get("/api/search-domains").json()["domains"]
    assert len(domains) == 20
    assert domains[0]["name"] == "web"
    assert domains[0]["typical_sources"] == ["Web search engines"]
```

- [x] Run `pytest tests/unit/servers/web/test_search_domain_http.py -q` and from web run `npm run test:unit -- src/pages/__tests__/AssistPage.test.tsx src/components/__tests__/SearchComposer.test.tsx src/__tests__/api.test.ts`; verify intended failures.
- [x] Replace domain defaults and neutral checks in the HTTP search paths with canonical web. Normalize legacy inputs before mode validation. Update discovery return annotation for arrays and emit metadata:

```python
{"name": name, "description": entry.description,
 "typical_sources": list(entry.typical_sources)}
```

Update TS interface, Assist initial state and fallback, SearchComposer default and label to "Search source". Option descriptions can include typical sources in the existing title text without implying all providers are integrated:

```tsx
title={[opt.description, opt.typical_sources?.join(", ")].filter(Boolean).join(" — ")}
```

- [x] Run the Task 2 suites and `npm run typecheck` from web. Preserve streaming and mixed-mode query handling. Commit `feat: expose source taxonomy in HTTP and Assist`.

### Task 3: Active documentation and evaluation compatibility

**Files:** Modify `README.md`, `docs/search-engine.md`; inspect `src/internal/retrieval/domain_eval.py`, `examples/run_domain_relevance_eval.py`, `data/eval/domain_relevance_queries.json`, `tests/unit/test_domain_eval.py` and active callers discovered with rg.

**Interfaces:** Current discovery is twenty source categories; archived topic relevance labels retain their historical subject interpretation explicitly rather than being relabeled as measured source-agent relevance.

- [x] Search active references with `rg -n 'general|finance|environment|AVAILABLE_DOMAINS|DOMAIN_REGISTRY' src tests examples data web/src README.md docs/search-engine.md`. Identify tests that intentionally exercise aliases versus stale expectations. Do not blanket-replace unrelated financial or environmental content.
- [x] Add a regression test in test_domain_eval.py before changing any evaluator behavior if canonical normalization would alter fixture queries. Pin the existing topic-hint semantics for historical evaluation, or make the workflow reject historical data with a clear incompatibility message; choose based on current evaluator architecture and ledger the decision. Do not invent new labeled data.
- [x] Run `pytest tests/unit/test_domain_eval.py -q`; inspect failure before implementation if evaluator behavior changes.
- [x] Update active documentation with the exact source taxonomy table, web default, alias mappings, source-metadata versus integrated-provider distinction, and preserved request examples. Document historical evaluation limits and prevent silent topic-to-source reinterpretation.
- [x] Re-run domain evaluation tests and relevant tests found in the reference scan; commit `docs: align search documentation and evaluation with source taxonomy`.

### Task 4: Whole-branch review and remote verification

**Files:** Plan validation record and PR description; no unrelated source edits.

- [x] Run all touched domain suites, HTTP suite, evaluator suite, search-query consumers and timeout drift guard together; run Ruff on changed Python files, frontend typecheck and unit tests, and git diff --check.
- [x] Request one fresh whole-branch reviewer against origin/main, the spec, this plan and Review Focus. Fix material findings with reproducing RED-to-GREEN tests; record minor findings/rulings.
- [x] Update plan checkboxes and local validation results; push feature branch and open a focused PR against main. Watch CI for the final commit. Reproduce and fix attributable failures, then verify new final-commit checks.
- [ ] Record CI results in PR body, mark ready after all checks pass, and report PR URL. Do not merge.

## Validation record

Design approved by the user.

Local validation (2026-10-10, final commit before PR):
- `pytest`: 5329 passed, 1 skipped (includes the timeout drift guard).
- `cd web && npm run test:unit`: 246 passed; `npm run typecheck`: clean.
- Ruff check/format on all changed Python files: clean; `git diff --check`: clean.
- `python -m examples.run_domain_relevance_eval --dry-run`: 48 labelled queries, 0 live calls.

Task 3 decision: the evaluator pins the archived topic hints (`HISTORICAL_TOPIC_HINTS`)
rather than reinterpreting labels as source categories. Read through the new aliases,
45 of 48 hinted queries changed and 18 collapsed into identical arms. Label files keyed
by a source category are rejected; no new labels were invented.

Whole-branch review: 0 critical, 1 important, 4 minor.
- Important, fixed RED->GREEN: `domain=environment` + `tag=environment.weather` (the
  pair old discovery emitted) was rejected; a domain spelled like the tag's legacy
  prefix now defers to the tag migration. Genuine conflicts are still rejected.
- Minor, fixed: duplicate discovery entries for aliases of one category; stale
  "topic hint" wording; README implied per-source routing; internal-helper tests
  passed legacy values.
- Note, accepted per spec: a legacy web-aliased topic (e.g. `health`) with
  `mode=chat_once` is now accepted, since it becomes `web` with no hint.
