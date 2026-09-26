# Multi-Arch Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one multi-arch (amd64 and arm64) image per tag, built
natively on per-platform runners.

**Architecture:** A matrix `build` job pushes each platform by digest, and a
`merge` job creates the tagged manifest list with `buildx imagetools create`.

**Tech Stack:** GitHub Actions (`ubuntu-24.04-arm`), docker buildx, the
`docker/*` actions, `actions/upload-artifact` and `download-artifact` v4,
pytest, actionlint.

**Spec:** `docs/superpowers/specs/2026-09-25-multi-arch-images-design.md`

## Global Constraints

- The triggers, permissions, tags, CI-success gate and never-from-a-PR rule
  are unchanged from #663.
- No QEMU.
- A missing platform never publishes: `merge` needs every leg.

## Review Focus

- Each matrix leg checks out the tested `head_sha`.
- Digest artifact names must be unique per platform. The platform contains a
  `/`, so derive a safe pair.
- `imagetools create` must get the tags from the metadata JSON and every
  digest.

---

### Task 1: Tests (red)

- [ ] Rewrite `tests/unit/test_publish_workflow.py` for the `build` and
  `merge` jobs, following the spec's Testing section. Run it and expect red,
  because the workflow is still a single job.

### Task 2: Workflow (green)

- [ ] Rewrite the jobs in `.github/workflows/publish-image.yml`. Run the
  tests and actionlint, and expect green.
- [ ] Run the mutation checks from the spec.
- [ ] Commit.

### Task 3: Docs and verification

- [ ] In `docs/deploy.md`, list the two platforms and drop the
  amd64-only line.
- [ ] Run the full unit suite, then ruff, then `git diff --check`, then
  actionlint. Open the PR.
