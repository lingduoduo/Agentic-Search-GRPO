# Make the Docker stack startable, and guard its claims with tests

## Goal

Fix the two defects that stop the containerised stack working, and put the
class of mistake behind tests — because Docker appears nowhere in CI, so nothing
has ever exercised these files.

## Two claims I got wrong first

Worth recording, because both came from reading the artifacts instead of
exercising them, and both are the reason the guard tests exist.

**"The retrieval service exposes no `/health`, so the healthcheck can never
pass."** Wrong. Grepping `@app.` in `demo.py` and `hybrid.py` finds only
`POST /retrieve` — but `/health` is registered by the shared
`create_base_app`, which both use. Exercised against the real app,
`GET /health` returns `200 {"status": "ok"}`. The compose healthcheck was
always fine.

**"`mlx-lm` has no linux wheels, so the image cannot build."** Wrong. `mlx` has
none — verified, `--platform manylinux2014_x86_64` resolves "from versions:
none" — but `mlx_lm` is a pure-python wheel whose metadata reads
`Requires-Dist: mlx>=0.31.2; platform_system == "Darwin"`. It conditions the
native dependency on macOS itself, so a linux install never attempts it.

## The defects that are real

### 1. The editable install registers nothing

The Dockerfile ran `pip install -e .` before `COPY . .`, so
`[tool.setuptools.packages.find]` discovered zero packages. Reproduced in a
clean venv:

| Install order | finder MAPPING | `import src` from `/app` | from any other cwd |
|---|---|---|---|
| install → copy (as shipped) | `{}` | works | **ModuleNotFoundError** |
| copy → install | `{'src': '/app/src'}` | works | works |

The container booted only because `WORKDIR /app` puts the cwd on `sys.path`.
Anything running from elsewhere — a subprocess, a worker, a one-off
`docker run … python -c` — fails.

Fixed by installing after the copy. `requirements.txt` is still installed first,
so editing source does not reinstall dependencies, and the package install is
`--no-deps` because every runtime dependency including PyJWT is already pinned
there.

### 2. The corpus never reaches the container

Compose ran the retrieval service with `--corpus_path /data/corpus.jsonl`. Two
independent reasons that path is empty:

- `.dockerignore` excludes `data`, so the corpus is not in the build context —
  even though it *is* tracked in git, which is what makes this look fine.
- Compose mounts the `app_data` named volume at `/data`, which would shadow
  anything the image did bake there.

`resolve_corpus_docs("/data/corpus.jsonl")` raises
`ValueError: Unknown corpus spec` — it reports a nonexistent path as an unknown
spec rather than a missing file, which is its own small trap.

Fixed by serving the corpus from the image at `/app/data/corpus.jsonl`, with
`.dockerignore` negations re-including the two tracked files the demo stack
needs. `/data` stays as the writable volume for the sqlite database.

## The constraint that is not a defect

**One uvicorn worker is load-bearing.** `app.state` holds the
`ToolApprovalBroker` (a client polls for a decision and must reach the worker
that created it), the `RequestCaptureStore` the Dev Console reads back, the
loaded search-agent model, and the memory encoder. Adding `--workers` makes tool
approvals fail intermittently, loses captures, and loads the model once per
worker.

Documented at the `CMD` line, where someone would add the flag, rather than in a
deployment guide they may not read. A test fails if `--workers` appears in either
file as a directive, with the reason in the message; a second test asserts the
`app.state` attributes that make the constraint true still exist, so the
constraint gets re-examined rather than silently outliving its cause.

## Not changed, deliberately

**Non-root user.** Docker named volumes are root-owned, and the app writes the
sqlite database to the `/data` volume. Adding `USER` without also handling volume
ownership would replace a startable stack with one that cannot write — and with
no Docker in CI, nothing would catch that. It needs the ownership work in the
same change.

**Python 3.11 in the image** while CI tests 3.10 and 3.12. Real inconsistency,
but changing the runtime interpreter of an unbuilt, untested image is not
something to do blind.

**A Docker build job in CI.** The image pulls torch, transformers, faiss and
pyserini; a build-and-boot job is minutes and gigabytes per run. The contract
tests here catch the specific class of mistake that occurred — an artifact
claiming something the code or build context does not provide — at unit-test
speed. If the image is going to be shipped, the slow job is still worth adding.

## Testing

`tests/unit/test_docker_stack_contract.py` exercises the artifacts rather than
reading them: the compose healthcheck path must be a route the real app serves
*and* return 200; the corpus argument must not sit under a volume mount, must
exist in the repo, and must survive `.dockerignore`; the editable install must
follow the source copy; and the web server must not be given `--workers`.

The two corpus/install tests were watched failing on the real defects. The
`--workers` guard was mutation-checked — and caught my own documentation on the
first run, since the comment explaining the constraint names the flag, so the
check now ignores comment lines.

`PyYAML` is declared in `requirements-unit-test.txt` because this module parses
the compose file. It was previously only a transitive of
starlette/pydantic-settings/pyserini, which is exactly how `httpx` went
undeclared in #607. Declared rather than `importorskip`ed, so a missing
dependency is an error instead of a silently vanished guard.
