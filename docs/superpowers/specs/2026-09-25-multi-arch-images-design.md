# Multi-architecture images (amd64 + arm64): design

## Problem

The publish workflow (#663) builds `linux/amd64` only, so the image cannot run
natively on arm64 hosts: Apple Silicon, Graviton, Ampere. The Dockerfile
installs CPU torch from `download.pytorch.org/whl/cpu`, which publishes Linux
arm64 wheels too, so an arm64 build is feasible.

## Decision (approved by the user: native arm64 runners)

`.github/workflows/publish-image.yml` follows Docker's documented pattern for
building on multiple runners. The triggers, the permissions, the tags, the
never-from-a-PR rule and the CI-success gate are all unchanged.

1. **`build` job, a matrix.**
   - It has one leg per platform:
     - `linux/amd64` on `ubuntu-latest`;
     - `linux/arm64` on `ubuntu-24.04-arm`, free for public repositories.
   - Each leg builds **natively** (no QEMU) from the tested commit. It carries
     the same `if:` gate and checks out `head_sha`, falling back to
     `github.sha`.
   - Each leg pushes an **untagged image by digest**
     (`push-by-digest=true,name-canonical=true`) and uploads that digest as an
     artifact.
   - The legs cache per platform, with GHA cache scope = platform.
   - `fail-fast: false`. One platform failing must not cancel the other's
     diagnostics. But the merge job needs **both** legs, so a missing
     platform never publishes.
2. **`merge` job, which `needs: build`.**
   - It downloads both digests.
   - It computes the same tags as today with `docker/metadata-action`:
     `sha-<full sha>`, `main`, and semver.
   - It runs `docker buildx imagetools create` with those tags over the two
     digests. That publishes **one multi-arch manifest list** per tag.
   - It then inspects the result.

**Result.** `docker pull ghcr.io/<owner>/agentic-search:sha-<sha>` resolves to
the right architecture automatically. The rollback handle (`sha-`) is
unchanged. `docs/deploy.md` drops the "images are `linux/amd64`" limitation
and notes both platforms.

## Out of scope

- More platforms, for example arm/v7.
- Signing and provenance.
- Changing the Dockerfile. Not needed: the torch CPU index and the
  `node:20-slim` and `python:3.11-slim` bases are all multi-arch.

## Testing

`tests/unit/test_publish_workflow.py` is updated to the two-job shape:

- **Triggers, the missing PR trigger, and permissions:** unchanged
  assertions.
- **`build`:**
  - the matrix has exactly `linux/amd64`/`ubuntu-latest` and
    `linux/arm64`/`ubuntu-24.04-arm`;
  - `runs-on` comes from the matrix;
  - it has the success gate;
  - the checkout uses `head_sha` with a `github.sha` fallback;
  - `build-push-action` takes `platforms` from the matrix, pushes by digest,
    and builds the root Dockerfile;
  - there is **no QEMU step**;
  - the digest is uploaded.
- **`merge`:**
  - it `needs: build`;
  - metadata has `type=sha,format=long` and semver;
  - the step runs `docker buildx imagetools create`.
- **Tooling:** actionlint is clean.

**Mutation checks:**
- Drop the arm64 leg, and the test goes red.
- Add a QEMU step, and the test goes red.
- Drop `needs: build`, and the test goes red.

The real end-to-end check is the first multi-arch publish after merge,
verified by inspecting the manifest list for both platforms.
