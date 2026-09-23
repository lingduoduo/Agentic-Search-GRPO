# Retire Dead Integration Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the 22 integration test files that reference no surviving endpoint, plus the directories and fixtures they leave orphaned.

**Architecture:** Deletion driven by the census tool from #633, using its stricter (helper-aware) attribution so the count is conservative.

**Tech Stack:** Python 3.10+, pytest, git.

**Spec:** `docs/superpowers/specs/2026-09-23-retire-dead-integration-tests-design.md`

## Global Constraints

- **Use the helper-aware count, not direct-only.** 22 is the conservative figure: it credits each test with every endpoint of every helper it imports. Direct-only says 36 and is the looser claim.
- **Verify "gone" by prefix against the live app**, not by exact match, before deleting anything that depends on it.
- **Check `main`'s collection errors first.** A red integration tree here is usually pre-existing; blaming the change in hand has wasted time before.
- **Leave the ~80 "mixed" files alone.** The tool cannot decide them, and guessing at scale is what made the census wrong twice.
- **Avoid `git stash` in this checkout.** It was used once here to compare against `main` and left the deletions unstaged — verify staging before committing.

---

### Task 1: Establish the set

- [x] **Step 1: Run the census** in helper-aware mode; capture the 22 files.
- [x] **Step 2: Spot-check the suspicious names** — `mcp`, `web_search`, `tags` — since those describe features the repo still has.
- [x] **Step 3: Confirm the features moved rather than vanished** (`mcp_server/` 5 modules, `web_search/` 6) and that their endpoint families return zero prefix matches. These are rewrites, not repairs.

---

### Task 2: Delete, including what is left orphaned

- [x] **Step 1: Compute which directories lose their last test** — eleven, including `pruning/` with its 70-file website fixture.
- [x] **Step 2: Remove those directories whole**, conftest fixtures included.
- [x] **Step 3: Remove the eleven remaining files individually** from directories that keep other tests.
- [x] **Step 4: Confirm all 22 are gone** and that nothing outside references the deleted paths.

---

### Task 3: Verify

- [x] **Step 1: Compare collection errors against `main`** — 16 before, 14 after; two removed, none introduced.
- [x] `pytest tests/unit` — 4362 passed, 1 skipped, unaffected.
- [x] **Step 2: Confirm the deletions are staged** after the stash round-trip.
- [x] Spec and plan committed on the branch.
