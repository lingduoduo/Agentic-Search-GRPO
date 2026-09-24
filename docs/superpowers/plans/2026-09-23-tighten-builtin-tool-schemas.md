# Tighten the built-in tool schemas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every seeded tool schema declares its real constraints as JSON Schema keywords, guarded by a test.

**Architecture:** Schema dict edits only, in the modules that own each tool; one guard test walks every seeded schema. No executor changes.

**Tech Stack:** Python, `jsonschema` Draft 2020-12 (already installed), pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-tighten-builtin-tool-schemas-design.md`

## Global Constraints

- Numeric ranges equal the existing in-code clamps (see the spec's Values table).
- `web_search.queries` maxItems 5.
- Patterns are ECMA-262-portable (no inline flags).
- Only open map: `params` on `search_domain`/`batch_search`.
- Executors and their clamps unchanged.

## Review Focus

- A value a real caller sends today that the tightened schema now rejects — pinned by running the existing public-data, domain-search and memory test files unchanged in Task 2.
- A default that violates its own schema (e.g. `limit` default outside its range) — pinned by `test_defaults_satisfy_their_schema`.
- A nested object inside `batch_search.queries.items` left open — pinned by the guard's recursive walk.
- Schema dicts shared between tools and mutated in place (e.g. `_search_properties()` reused) — pinned by the guard running over every tool independently.
- Pattern rejecting a value the executor accepts (e.g. `HTTP://` for extract_page, `zh-yue` for Wikipedia) — pinned by table rows.

---

### Task 1: Guard + table tests (RED)

**Files:** Create `tests/unit/test_tool_schema_strictness.py`.

- [ ] **Step 1:** Write the guard (rules 1–6 from the spec) over `tool_knowledge_base()` + `build_rag_routing_tool(...)` + the memory tools, a `test_defaults_satisfy_their_schema`, and a parametrised table `(tool, args, ok)` covering every row of the spec's Values table with one passing and one failing case each.
- [ ] **Step 2:** Run `pytest tests/unit/test_tool_schema_strictness.py -q`. Expected: guard and failing-case rows FAIL.

### Task 2: Tighten the schemas (GREEN)

**Files:** Modify `src/internal/tools/public_data/{knowledge,geo,market}.py`, `src/internal/tools/search.py` (`MultiQueryWebSearchTool.schema`, `_search_properties`, `build_domain_search_tools`), `src/internal/tools/routing_tools.py`, `src/internal/memory/tools.py`.

- [ ] **Step 1:** Apply the spec's Values table and rules; add `additionalProperties: False` to every object schema; `minLength: 1` on required strings.
- [ ] **Step 2:** Run `pytest tests/unit/test_tool_schema_strictness.py -q`. Expected: PASS.
- [ ] **Step 3:** Mutation check: remove one `maximum` and one `additionalProperties`; the guard goes red; restore; clear `__pycache__`.
- [ ] **Step 4:** Run `ruff check . && ruff format --check . && pytest -q`. Expected: only the pre-existing live-OpenAI `test_no_tool_calls_on_chat_path` failure.
- [ ] **Step 5:** Commit `tools: declare the built-in tools' constraints as schema keywords`.
