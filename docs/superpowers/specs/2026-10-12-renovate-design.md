# Renovate dependency updates: design

## Goal
Automated dependency-update PRs for every ecosystem in the repo, run entirely
from a GitHub Actions workflow (self-hosted Renovate, no GitHub App), without
ever moving a load-bearing pin.

## Decisions
- **Runner:** `renovatebot/github-action` v46.3.5, pinned by SHA; Renovate CLI
  pinned to 44.117.0. Weekly (Mon 05:00 UTC) plus `workflow_dispatch`.
- **Credential:** fine-grained PAT, this repo only, read/write on Contents, Pull
  requests, Issues and Workflows, stored as secret `RENOVATE_TOKEN`. PRs opened
  with `GITHUB_TOKEN` trigger no workflows, so the run fails when the secret is
  missing rather than falling back.
- **Config:** `renovate.json` read as repository config from the default branch
  (no `configurationFile`), so it is validated and reviewed like code.
- **Policy:** Dependency Dashboard; majors need dashboard approval; python/node
  base-image minor/major bumps need approval; minor+patch grouped per ecosystem;
  Actions pinned to digests; `gomodTidy`; pre-commit manager on; 5 concurrent /
  2 per hour; label `dependencies`; prefix `chore(deps):`; no automerge.
- **Coverage:** pip_requirements' default pattern misses multi-hyphen names
  (`requirements-unit-test.txt`, `requirements-retrieval-heavy.txt`), so the
  manager gets `managerFilePatterns: /(^|/)requirements(-[\w-]+)?\.txt$/`.
- **Protected pins**, only where the repo documents a reason: fastapi `<0.137`,
  aiohttp disabled, openai `<3`, mcp `<2`, fastmcp `<4`. numpy==1.26.4 has no
  documented reason, so it is left to the major-approval rule.
- **CI:** a `renovate-config` job runs `renovate-config-validator --strict`
  under Node 24 on every PR.

## Verification
- Strict validator passes.
- Local `--platform=local --dry-run=lookup` under Node 24: all four
  requirements files detected; no proposal for any protected package (aiohttp
  reports `skipReason: disabled`); numpy 2.x shows as a major (dashboard-gated).
- First real run is manual, after the secret is added.

## Out of scope
Automerge, the Renovate GitHub App, lockfile maintenance, vulnerability-alert
tuning.
