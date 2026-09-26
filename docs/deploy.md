# Deploy and roll back

Every `main` commit that passes CI is published as a container image. So is
every `v*` tag. The workflow is `.github/workflows/publish-image.yml`.

A deploy or a rollback means **running a published tag**. You never check out
code and rebuild.

## Image tags

Images are published to `ghcr.io/<owner>/agentic-search`, for example
`ghcr.io/lingduoduo/agentic-search`, as one multi-arch manifest per tag:

- **Platforms.** Each tag covers `linux/amd64` and `linux/arm64`, each built
  natively on its own runner. `docker pull` picks the right one for the host.
- **Tag names.** Every `sha-` tag names the commit CI tested, as does the
  image's `revision` label.

| Tag | When | Use |
|---|---|---|
| `sha-<full commit sha>` | Every published build | **The deploy and rollback handle.** It is immutable: one commit, one image. |
| `main` | Every CI-passed `main` commit | A moving convenience tag. **Never roll back to it.** |
| `X.Y.Z`, `X.Y` | Pushing a `vX.Y.Z` git tag | Releases. |

**What gets published.** A `main` image is published only after the `CI`
workflow succeeded, and it is built from the exact commit CI tested. Pull
requests never publish.

**Package visibility.** The package is created on the first publish. It
inherits the visibility of the repository it is linked to, so from this public
repository it starts **public**; this was measured on the first publish, when
an anonymous client could pull `:main`. To make it private, once:

1. Open
   `https://github.com/users/<owner>/packages/container/agentic-search/settings`.
2. Under **Change visibility**, choose **Private**. Later publishes keep that
   setting.

`GITHUB_TOKEN` cannot change it from the workflow. Pulling a private image
needs `docker login ghcr.io` with a token that has `read:packages`.

## Deploy a tag

`docker/docker-compose.yml` runs the image named by `AGENTIC_SEARCH_IMAGE`.
Unset, it falls back to `agentic-search:local`, which `up --build` builds from
source as before.

```bash
export AGENTIC_SEARCH_IMAGE=ghcr.io/<owner>/agentic-search:sha-<full sha>
docker compose -f docker/docker-compose.yml pull retrieval web
docker compose -f docker/docker-compose.yml up -d --no-build --wait
curl -fsS http://localhost:7860/ready   # 200 = store and retrieval reachable
```

## Roll back

1. **Pick the previous good tag.** Use the `sha-<sha>` of the last release
   that worked, from `git log --first-parent main` or the package page. Never
   use `main`.
2. **Redeploy it** with the commands under [Deploy a tag](#deploy-a-tag),
   using that tag.
3. **Check it.** `/ready` should return 200, and `/metrics` should show the
   error rates coming back down (see [alerts](observability-metrics.md#alerts)).

### The database on rollback

The SQLite store carries a schema version and a **min-reader version**, as
described in [architecture](architecture.md). When an older image opens a
database a newer image already migrated, one of two things happens:

- **Its schema is at least the min-reader version.** This is the normal case
  for additive changes. It opens the database without changing it, and the
  rollback just works.
- **Otherwise it refuses to start** with `SchemaVersionError`. That means the
  newer build made a change older builds cannot read. Either roll forward to
  a build that can read it, or restore the backup taken before the upgrade.

So **back up the data volume before deploying a build that migrates the
schema**. Compose names the volume after the project, for example
`docker_app_data`:

```bash
# before the upgrade
docker run --rm -v docker_app_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/app_data-$(date +%Y%m%d%H%M).tgz -C /data .

# to restore (stack stopped)
docker compose -f docker/docker-compose.yml down
docker run --rm -v docker_app_data:/data -v "$PWD":/backup alpine \
  sh -c 'rm -rf /data/* && tar xzf /backup/app_data-<stamp>.tgz -C /data'
```

## Not covered

- Deploying to a specific environment.
- Image signing and provenance attestation.
- Platforms beyond `linux/amd64` and `linux/arm64`.
