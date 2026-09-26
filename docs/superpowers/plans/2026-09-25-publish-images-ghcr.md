# Publish Images To GHCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every `main` commit that passes CI, and every `v*` tag, publishes an
immutable `sha-` tagged image to GHCR, and compose can run any published tag.
Rollback then means redeploying the previous tag.

**Architecture:** A `workflow_run`-gated publish workflow (buildx, login,
metadata, build-push), a compose `image:` override variable, and a runbook.

**Tech Stack:** GitHub Actions, the docker/* actions, docker compose, pytest,
actionlint.

**Spec:** `docs/superpowers/specs/2026-09-25-publish-images-ghcr-design.md`

## Global Constraints

- Never publish from a pull request.
- Permissions are `contents: read` and `packages: write` only.
- Use `GITHUB_TOKEN`; no new secrets.
- Rollback always uses `sha-` tags, never `main`.
- `up --build` in the CI compose job stays unchanged.

## Review Focus

- A `workflow_run` build must check out `head_sha`, not the default branch
  tip.
- A tag push has no `workflow_run` payload, so the checkout falls back to
  `github.sha`.
- The documented-env-vars test and the compose contract test stay green.

---

### Task 1: Tests (red)

- [x] **Workflow test.** Write `tests/unit/test_publish_workflow.py` per the
  spec's Testing section. PyYAML parses `on:` as `True`, so read
  `doc.get("on", doc.get(True))`.
- [x] **Compose contract test.** Add a test to
  `tests/unit/test_docker_stack_contract.py`: `retrieval` and `web` have
  `image == "${AGENTIC_SEARCH_IMAGE:-agentic-search:local}"` and a `build`.
- [x] **Run them.** Expect failures: the workflow is missing and compose has
  no `image:`.

### Task 2: Workflow and compose (green)

- [x] **Workflow.** Write `.github/workflows/publish-image.yml`.
- [x] **Compose.** Add the `image:` lines to `docker/docker-compose.yml`.
- [x] **Run.** Run the tests and `actionlint`. Expect them to pass.
- [x] **Mutation checks.** These are in the spec.

### Task 3: Runbook and verification

- [x] **Runbook.** Write `docs/deploy.md`, and add the `AGENTIC_SEARCH_IMAGE`
  row to `docs/configuration.md` if the documented-env-vars test requires it.
- [x] **Verify.** Run the full unit suite, then ruff, then `git diff --check`.
  Open the PR.
