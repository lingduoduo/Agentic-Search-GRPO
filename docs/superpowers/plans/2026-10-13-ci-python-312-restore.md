# Restore CI Python 3.12 after #694: plan

**Spec:** docs/superpowers/specs/2026-10-13-ci-python-312-restore-design.md

1. Set the four `python-version: "3.15"` values in `ci.yml`/`eval-gate.yml` back to "3.12"
   → verify: `git grep 'python-version: "3'`.
2. CI validator reads `renovate-version` from `renovate.yml` → verify: the step's
   commands, run locally, validate with 44.155.1.
3. Update the version shown in `docs/development/dependency-updates.md`.
4. PR CI → verify: all checks green, including the Intent Routing Gate.
