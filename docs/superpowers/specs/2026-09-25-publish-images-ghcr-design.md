# Versioned images on GHCR and a rollback runbook: design

## Problem

There is no deploy pipeline. CI builds `agentic-search:ci` and never pushes it,
and `docker/docker-compose.yml` only builds from source. So "roll back" means
checking out an older commit and rebuilding: slow, not reproducible, and there
is no record of what was running.

## Decision (approved by the user: GHCR, private package)

### 1. The publish workflow: `.github/workflows/publish-image.yml`

**When it runs.**
- **Main.** On `workflow_run` of the `CI` workflow, `types: [completed]`,
  `branches: [main]`. The job runs only when
  `github.event.workflow_run.conclusion == 'success'`, so an image is
  published only for a `main` commit that passed CI. It checks out
  `github.event.workflow_run.head_sha`, never the moving branch tip.
- **Releases.** On `push` of tags `v*`: a release is an explicit human act,
  so it builds that tag.
- **Never from a pull request.** There is no `pull_request` trigger.

**Permissions.** Least privilege at the workflow level: `contents: read`,
`packages: write`. It logs in to `ghcr.io` with the built-in `GITHUB_TOKEN`,
so no new secrets are needed.

**Image.** `ghcr.io/${{ github.repository_owner }}/agentic-search`, built from
the root `Dockerfile`, the same image CI tests. It is tagged by
`docker/metadata-action`:
- `sha-<full commit sha>`, always. This is the immutable deploy and rollback
  handle.
- `main`, on `main` builds. This is a moving convenience tag, **never** used
  for rollback.
- `X.Y.Z` and `X.Y`, from `vX.Y.Z` tags.

It carries the standard OCI labels (`org.opencontainers.image.source`,
`revision`, `created`), which link the package to the repository. It uses the
GitHub Actions build cache. Concurrency is one publish per ref, and a newer run
does not cancel an older one mid-push.

**Package visibility.** *Corrected after the first publish.* This spec
assumed new GHCR packages start private, but a package linked to a public
repository inherits **public** visibility: the first image was anonymously
pullable. The runbook tells the owner to switch it to private once, in the
package settings. The workflow cannot set it with `GITHUB_TOKEN`.

### 2. Compose can run a published image

`retrieval` and `web` in `docker/docker-compose.yml` gain
`image: ${AGENTIC_SEARCH_IMAGE:-agentic-search:local}` next to their existing
`build:`.
- **Today's `up --build`** is unchanged. It builds and tags
  `agentic-search:local`, and the CI compose job keeps working.
- **Running a published image** is
  `AGENTIC_SEARCH_IMAGE=ghcr.io/<owner>/agentic-search:sha-<sha> docker
  compose up -d --no-build`.

### 3. The runbook: `docs/deploy.md`

It covers:
- the tag scheme;
- deploying a given `sha-` or version tag;
- **rolling back**:
  1. pick the previous `sha-` tag, from the package page or the git log;
  2. redeploy it with `--no-build`;
  3. check `/ready`.
- **database compatibility on rollback**, tied to #662:
  - if the older image's schema is at least the database's min-reader version,
    it opens the database unchanged;
  - otherwise it refuses with `SchemaVersionError`.

  So **back up the `app_data` volume before deploying a build that migrates
  the schema**, and restore that backup if a rollback is refused. The
  commands are included.
- checking the package visibility once.

`docs/configuration.md` gets the `AGENTIC_SEARCH_IMAGE` compose variable,
because the documented-env-vars test may require it.

## Out of scope

- Deploying to any environment.
- Signing or attestation.
- Multi-arch builds (a follow-up).
- Setting package visibility through the API.

## Testing

In `tests/unit/test_publish_workflow.py`, which parses the workflow YAML:

- **Triggers.** `workflow_run` on `CI`, `completed`, `branches: [main]`; tags
  `v*`; **no** `pull_request` trigger.
- **The job condition.** The success condition is present, and the checkout
  uses `workflow_run.head_sha` when present.
- **Permissions.** Exactly `contents: read` and `packages: write`.
- **The image and tags.** The image name is
  `ghcr.io/${{ github.repository_owner }}/agentic-search`, the tags include
  `type=sha,format=long`, and the build pushes the root `Dockerfile`.
- **actionlint.** It runs when available, and is skipped otherwise.

In the compose contract (`tests/unit/test_docker_stack_contract.py`):
`retrieval` and `web` declare
`image: ${AGENTIC_SEARCH_IMAGE:-agentic-search:local}` and keep `build`.

**Mutation checks:**
- Add a `pull_request` trigger and watch it go red.
- Widen the permissions (for example `contents: write`) and watch it go red.
- Drop the success condition and watch it go red.
