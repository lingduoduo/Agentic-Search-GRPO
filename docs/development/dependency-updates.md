# Dependency updates (Renovate)

Dependency-update PRs come from a self-hosted Renovate run in GitHub Actions
([`.github/workflows/renovate.yml`](../../.github/workflows/renovate.yml)), not
from the Renovate GitHub App. The policy lives in
[`renovate.json`](../../renovate.json) at the repository root.

## Schedule and policy

- Runs every Monday at 05:00 UTC. You can also start it from **Actions → Renovate → Run workflow**.
- Minor and patch updates are grouped into one PR per ecosystem:
  - python (`requirements*.txt`, `pyproject.toml`)
  - npm (`web/`, root `package.json`)
  - go (`cli/go.mod`, followed by `go mod tidy`)
  - GitHub Actions
  - pre-commit hooks
- Majors are never opened automatically. They appear on the **Dependency Dashboard** issue, and ticking one there opens its PR on the next run.
- A minor or major bump of a `python`/`node` runtime version also waits for dashboard approval, in its own "runtime versions" group. That covers base images in the Dockerfiles and compose, and `python-version`/`node-version` in workflows. The application baseline and CI use Python 3.12.
- A release must be at least **3 days old** (`minimumReleaseAge`) before Renovate proposes it; until then it proposes the newest release that is old enough. Docker registries may omit release timestamps (for example GHCR), so those updates may proceed without this check; Docker releases with timestamps still wait 3 days. Only the two expected missing-timestamp notices are remapped to info, as [Renovate documents](https://docs.renovatebot.com/key-concepts/minimum-release-age/#minimumreleaseagebehaviourtimestamp-optional-warning). Other warnings remain visible. Renovate does not post its `renovate/stability-days` commit status (`statusCheckNames.minimumReleaseAge: null`). The token has no Commit statuses permission, and Renovate turns a failed status POST into a `repository-changed` abort.
- GitHub Actions are pinned to commit digests, with the version in a comment.
- At most 5 open Renovate PRs, and at most 2 created per hour.
- Every PR is labelled `dependencies` and titled `chore(deps): …`.
- Nothing is merged automatically.

## Protected pins

Some pins are load-bearing. Each has a `packageRules` entry marked `PROTECTED`
whose `description` gives the reason:

| Package | Rule | Why |
|---|---|---|
| Python runtime | `<3.13` | PyTorch/FAISS have no Python 3.15 wheels; the protected aiohttp pin has wheels only through 3.12 |
| `fastapi` | `<0.137` | 0.137.x+ regresses `app.include_router` (silently mounts no routes) |
| `aiohttp` | disabled | `==3.9.3` is load-bearing: with openai 3.x resolved against it, `import openai` fails |
| `openai` | `<3` | moves together with the aiohttp pin, by hand |
| `mcp` | `<2` | 2.x removed `streamablehttp_client`, which `src/internal/tools/mcp_client.py` imports (#650) |
| `fastmcp` | `<4` | 4.x requires mcp 2.x |

To protect another pin, add a rule with `matchPackageNames` and either
`allowedVersions` (cap the range) or `"enabled": false` (freeze it entirely).
Put the reason in `description`, and also in a comment next to the pin in the
requirements file. To lift a protection, delete the rule in the same PR that
fixes the reason it existed.

`renovate.json` is validated in CI by the **Renovate config** job
(`renovate-config-validator --strict`), using the `renovate-version` that
`renovate.yml` runs. To check it locally (Renovate 44 needs Node 24; substitute
that version for `44.157.0` if it has moved):

```bash
npx --yes -p node@24 -p renovate@44.157.0 renovate-config-validator --strict renovate.json
```

To preview what Renovate would propose without opening anything:

```bash
LOG_LEVEL=debug RENOVATE_GITHUB_COM_TOKEN=$(gh auth token) \
  npx --yes -p node@24 -p renovate@44.157.0 \
  renovate --platform=local --dry-run=lookup --onboarding=false
```

## One-time setup: `RENOVATE_TOKEN`

The workflow needs a token other than `GITHUB_TOKEN`: PRs opened with
`GITHUB_TOKEN` do not trigger other workflows, so Renovate PRs would never run
CI. If the secret is missing, the workflow fails at its first step and does not
fall back.

1. GitHub → **Settings → Developer settings → Fine-grained personal access tokens → Generate new token**.
2. **Repository access:** only this repository.
3. **Permissions (Read and write):**
   - Contents
   - Pull requests
   - Issues (for the Dependency Dashboard)
   - Workflows (to update `.github/workflows/*`)

   Fine-grained tokens default to read-only. With Contents read-only, Renovate
   can read the repo but cannot push a single branch.
4. Repository → **Settings → Secrets and variables → Actions → New repository secret**, name `RENOVATE_TOKEN`.
5. Run the workflow once from the Actions tab. That creates the Dependency Dashboard issue.

Renovate itself exits successfully when it cannot push a branch, and when it
aborts the repository part-way through. The workflow's **Fail on branch or
repository errors** step reads Renovate's log. It fails the run on any
`Error updating branch` warning, on any ERROR, and when the `Repository
finished` result is anything but `done` (for example `repository-changed`), so
a missing permission shows up as a red run. Other warnings appear as
annotations only.

The token expires according to the lifetime you chose. An expired token is
still a non-empty secret, so the first step passes and the run fails when
Renovate cannot authenticate: rotate the secret.

## Dependency dashboard #693 migration

The baseline now uses Node 24, NumPy 2.5 and Python 3.12+, Vite 8,
Vitest 5, TypeScript 7, and Charm v2 (canonical `charm.land/*/v2` modules).
Python 3.15 remains blocked by the inference stack; changing only the image
or workflow runtime would break installation. The runtime cap keeps those
unsupported updates out of routine proposals until the protected pins migrate.

PostgreSQL moves to 18 and Redis to 8. Existing PostgreSQL 16 volumes require
a dump/restore or `pg_upgrade`; see [the migration instructions](../deploy.md#postgresql-16-to-18).
Ruff 0.17 uses explicit Python file patterns and the previous lint rule set
(`E4`, `E7`, `E9`, `F`) so its expanded defaults do not add Markdown linting.
