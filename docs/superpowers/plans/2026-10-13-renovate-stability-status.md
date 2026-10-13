# Renovate: stop the repository-changed abort, and catch aborts: plan

**Spec:** docs/superpowers/specs/2026-10-13-renovate-stability-status-design.md

1. `renovate.json`: `statusCheckNames.minimumReleaseAge: null` → verify: strict validator.
2. `renovate.yml`: log file at info; the check fails unless `Repository finished`
   result is `done` → verify: extracted script on done, changed, push, empty and
   missing logs.
3. Update the docs.
4. After merge, trigger the workflow → verify: result done, PRs opened, check green.
