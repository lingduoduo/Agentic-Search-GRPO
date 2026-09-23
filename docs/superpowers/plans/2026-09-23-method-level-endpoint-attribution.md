# Method-Level Endpoint Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute endpoints to the helper method a test calls rather than the module it imports, follow derived URL constants, and delete what the added precision reveals.

**Architecture:** Two extractor additions — per-callable endpoint indexing plus module-level URL-prefix resolution — feeding the existing per-file buckets.

**Tech Stack:** Python 3.10+ (`ast`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-method-level-endpoint-attribution-design.md`

## Global Constraints

- **Read the bucket, do not infer it.** "In the old set but not the new" is not "now mixed"; that misreading nearly produced a wrong correction to #634. Print the bucket.
- **Confirm any "gone" family by prefix probe** against a real `create_web_app()` before deleting on it.
- **Measure both attributions on the same file set** when comparing, or the counts are not comparable.
- **Check `origin/main`'s collection errors** before attributing any to the change.
- **Avoid `git stash` here**; used twice for main comparisons and it left deletions unstaged both times. Verify `git diff --cached`.

---

### Task 1: Resolve to the callable

- [x] **Step 1: Index endpoints per `Class.method` and per module-level function.**
- [x] **Step 2: Detect which callables a test invokes**, by attribute and bare name.
- [x] **Step 3: Replace the coarse helper mode** with this, keeping `--direct-only` as the floor.

---

### Task 2: Follow derived URL constants

- [x] **Step 1: Record module-level assignments** of a server URL plus a prefix.
- [x] **Step 2: Prepend the prefix** when an f-string references such a name.
- [x] **Step 3: Verify against the motivating case** — `/manage/admin/discord-bot/...` and `/build/...`, 31 endpoints previously invisible.

---

### Task 3: Reconcile against #634

- [x] **Step 1: Compare wholly-dead sets** on the same 119 files.
- [x] **Step 2: Investigate the difference properly** — two files had moved to unattributable, not mixed; a direct probe confirmed their endpoint families return zero routes.
- [x] **Step 3: Conclude #634 is a safe subset.**

---

### Task 4: Delete what the precision reveals

- [x] **Step 1: Probe `/cli`, `/projects`, `/build` by prefix** — zero routes each.
- [x] **Step 2: Remove the three files**, and the two directories left with no tests.
- [x] **Step 3: Confirm the census reports zero wholly-dead files afterwards.**

---

### Task 5: Verify

- [x] **Step 1: Collection errors unchanged** at 14, checked against `origin/main`.
- [x] `pytest tests/unit` — 4362 passed, 1 skipped, unaffected.
- [x] **Step 2: Confirm deletions staged** after the stash round-trip.
- [x] Spec and plan committed on the branch.
