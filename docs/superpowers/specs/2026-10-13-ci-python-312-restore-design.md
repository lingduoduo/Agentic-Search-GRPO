# Restore CI Python 3.12 after #694: design

## Problem
#694 (Renovate's github-actions group) was merged before #696 gated runtime
versions. It moved four `setup-python` jobs from Python 3.12 to 3.15:
- the lint job in `ci.yml`;
- all three `eval-gate.yml` jobs.

No torch wheel exists for 3.15, so the Intent Routing Gate cannot install.
#694 also bumped `renovate-version` to 44.155.1 in `renovate.yml`, while the
CI validator kept a copied 44.117.0 pin.

## Decision
- Set the four `python-version` values back to "3.12" and keep #694's action
  pins. Python 3.15 is now listed on the Dependency Dashboard (#693) as a
  "runtime versions" update waiting for approval, so it returns only when
  approved.
- The CI validator reads `renovate-version` from `renovate.yml` instead of
  pinning its own copy, so a Renovate self-bump cannot desynchronise them.
- Docs show 44.155.1 and point at `renovate.yml` as the source of truth.

- `minimumReleaseAge: "3 days"`. Renovate proposed a day-old Renovate
  release (44.155.1) and a days-old Python 3.15.0. Renovate 44 defaults to
  `minimumReleaseAgeBehaviour: timestamp-required`, and Docker tags carry no
  timestamp, so the docker datasource gets `timestamp-optional`. Otherwise every
  Docker update would stay pending forever, even after approval.

## Verification
- Dry run with the age rule: python 3.15.0, numpy 2.5.4, vitest 5.0.4 and
  lucide-react 1.55.0 are held, and the newest old-enough version is proposed.
  Docker updates (postgres, redis, node, python images) are not held.
- `git grep 'python-version: "3'` shows only "3.12".
- The validator step's version extraction, run locally, yields 44.155.1 and
  the strict validator passes.
- CI on this PR: Intent Routing Gate green on 3.12.
