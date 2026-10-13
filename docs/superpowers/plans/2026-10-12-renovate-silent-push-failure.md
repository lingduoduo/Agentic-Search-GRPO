# Renovate: fail the run when branches cannot be pushed: plan

**Spec:** docs/superpowers/specs/2026-10-12-renovate-silent-push-failure-design.md

1. Add `RENOVATE_LOG_FILE`/`RENOVATE_LOG_FILE_LEVEL` and the check step to
   `.github/workflows/renovate.yml` → verify: extracted script exits 0 on a
   clean log, 1 on an `Error updating branch` record, 1 on a missing file.
2. Document the step and correct the token-expiry wording in
   `docs/development/dependency-updates.md`.
3. After merge, trigger the workflow → verify: green run, PRs updated.
