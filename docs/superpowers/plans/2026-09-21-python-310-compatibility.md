# Python 3.10 Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tree run on Python 3.10, the floor `pyproject.toml` declares, and leave a guard that notices when it stops.

**Architecture:** Three independent fixes, each a place where 3.11 or 3.12 changed something, plus an `ast` guard tied to `requires-python`.

**Tech Stack:** Python 3.10 and 3.12, pytest, `ast`, `tomllib`/`tomli`.

**Spec:** `docs/superpowers/specs/2026-09-21-python-310-compatibility-design.md`

## Global Constraints

- **Verify on both interpreters.** A fix that only passes on 3.12 is the bug being fixed. Run the 3.10 interpreter explicitly; `python3` here is 3.12.
- **Do not raise `requires-python` to dodge this.** The repo ships a `tomli` backport and a test asserting it, so 3.10 support is intended.
- **Do not touch `src/internal/hooks/executor.py:227`.** Its bare `except TimeoutError` follows `urllib`, where the builtin is what gets raised. It is correct.
- **Do not weaken a contract to make a test pass.** Where a float assertion must loosen, use a tolerance tight enough that a real change in the math still fails.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Replace `datetime.UTC` (3.11+)

**Files:**
- Modify: `src/internal/servers/web/tool_approval.py`, `src/agents/tool/tool_calling.py`, `tests/unit/test_tool_approval.py`, `tests/unit/servers/web/test_sse_streaming.py`, `tests/unit/servers/web/test_tool_approval_broker.py`

- [x] **Step 1: Import `timezone`, use `timezone.utc`**

14 references across 5 files. `timezone.utc` is the same object `UTC` aliases in 3.11+, so `tzinfo is timezone.utc` identity assertions keep holding.

**Verify:** the 3.10 interpreter collects the suite — 187 collection errors drop to the handful caused by missing packages.

---

### Task 2: Catch the timeout `asyncio.wait_for` actually raises

**Files:**
- Modify: `src/internal/servers/web/tool_approval.py`

- [x] **Step 1: `except asyncio.TimeoutError`**

3.11 merged `asyncio.TimeoutError` into the builtin; on 3.10 they are unrelated, so the builtin never catches it and the broker's expiry path falls through. `asyncio.TimeoutError` is correct on both and is what the other thirteen handlers in `src/` already use. Leave a comment saying why, or someone will "simplify" it back.

**Verify:** `test_expiry_returns_expired_and_cleans_up` passes on 3.10.

---

### Task 3: Stop two assertions encoding 3.12-only arithmetic

**Files:**
- Modify: `tests/unit/test_reward_performance_contracts.py`

- [x] **Step 1: Gate the Neumaier premise**

`assert naive != sum(values)` holds only where `sum()` compensates. The docstring already said "on CPython 3.12+"; make the code agree with it.

- [x] **Step 2: Compare the breakdown with `approx`**

Production and `reward_baseline()` sum in different orders and differ in the last ulp; 3.12's compensation was hiding it. Use `pytest.approx`, matching `test_every_preset_keeps_its_scalar_total` two tests below, and say in a comment why `==` is wrong here.

**Verify:** 67 passed on both interpreters.

---

### Task 4: Guard the floor

**Files:**
- Create: `tests/unit/test_python_version_floor.py`

- [x] **Step 1: Parse the declared floor, then walk the tree**

Read `requires-python`, `ast`-walk `src/`, `tests/` and `examples/`, fail on any imported name newer than the floor. Exempt `src/context/enums.py`, which already guards `StrEnum` behind `try/except ImportError`.

- [x] **Step 2: Guard the guard**

Pin the floor at 3.10 in its own test, so raising `requires-python` relaxes the check deliberately rather than silently.

**Verify:** `pytest -q` → 4327 passed on 3.12; 4155 passed on 3.10 with only missing-package failures left.

---

### Follow-up, not done here

Add 3.10 to the CI matrix. The `ast` guard catches names that do not exist; it cannot catch behaviour that differs, which is what Tasks 2 and 3 were. Only running the suite on the floor closes that. Left open because it widens CI cost and may surface dependency work on 3.10 — a call about the project's support commitment, not a bug fix.
