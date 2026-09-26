# GHCR Package Stays Public: Docs Plan

**Goal:** Record the decision to keep the GHCR package public, and drop the
"make it private" step from the runbook.

**Spec:** `docs/superpowers/specs/2026-09-25-publish-images-ghcr-design.md`
(Package visibility).

**Why:** The image holds only what the public repository already contains:
code and the tracked `data/` files. Secrets come from the runtime
environment. Public pulls need no `docker login` and no token on the deploy
host.

### Task 1: Docs

- [x] `docs/deploy.md`: the visibility section says the package is public on
  purpose, explains why, and says how to go private if private data is ever
  baked into the image.
- [x] Spec: record the decision in the heading and in Package visibility.
- [x] Run the docs and link tests, then `git diff --check`. Open the PR.

No code or workflow changes.
