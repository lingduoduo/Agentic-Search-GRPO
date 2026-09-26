"""The Docker stack's claims must match what the code and build context provide.

Docker appears nowhere in CI, so nothing exercised these files. Two claims in
them were wrong in opposite directions, and both were things a grep answered
incorrectly:

- The compose healthcheck curls ``/health`` on the retrieval service. Grepping
  ``@app.`` in ``demo.py`` finds no such route -- it is registered by the shared
  ``create_base_app``, so the healthcheck is fine. Checking the *app* rather than
  the file is the only way to know.
- The retrieval command passes ``--corpus_path``. The corpus is tracked in git,
  so it looks available -- but ``.dockerignore`` excludes ``data`` and compose
  mounts an empty volume over the path, so it never reaches the container.

These tests exercise the artifacts instead of reading them.
"""

from __future__ import annotations

import re
import shlex
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml  # declared in requirements-unit-test.txt; a skip here would hide the guard

REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "docker" / "docker-compose.yml"
DOCKERFILE = REPO / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"
IMAGE_WORKDIR = "/app"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def _healthcheck_paths(service: dict) -> list[str]:
    """URL paths a service's healthcheck requests."""
    test = (service.get("healthcheck") or {}).get("test") or []
    if isinstance(test, str):
        test = shlex.split(test)
    return [
        match.group(1)
        for part in test
        if isinstance(part, str)
        for match in [re.search(r"https?://[^/\s]+(/[^\s'\"]*)", part)]
        if match
    ]


def _dockerignore_excludes(rel_path: str) -> bool:
    """Whether *rel_path* is excluded from the build context.

    Last matching rule wins, including negations under an excluded directory.
    An earlier version of this helper implemented a "docker does not descend into
    an excluded directory" rule instead, on the strength of one build that was
    actually run against a pre-#622 `.dockerignore` with no negations at all.
    That rule held for the legacy builder; buildkit honours the negations.
    ``test_dockerignore_semantics_match_measured_behaviour`` pins the three forms
    that were then measured properly.
    """
    return _excluded_by(_dockerignore_rules(), rel_path)


def _dockerignore_rules() -> list[tuple[bool, str]]:
    rules = []
    for raw in DOCKERIGNORE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        rules.append((negate, (line[1:] if negate else line).rstrip("/")))
    return rules


def _excluded_by(rules: list[tuple[bool, str]], rel_path: str) -> bool:
    excluded = False
    for negate, pattern in rules:
        if (
            rel_path == pattern
            or fnmatch(rel_path, pattern)
            or rel_path.startswith(pattern + "/")
        ):
            excluded = not negate
    return excluded


@pytest.mark.parametrize(
    ("label", "rules", "expected_excluded"),
    [
        # Measured with `docker build --no-cache` against buildkit 29.2.1, one
        # form per build, reading only timestamped output lines -- an earlier
        # attempt matched the FAIL string inside the echoed RUN command and
        # reported every form as failing.
        ("bare exclude, no negation (pre-#622)", [(False, "data")], True),
        (
            "bare exclude plus negations (#622)",
            [(False, "data"), (True, "data/corpus.jsonl")],
            False,
        ),
        (
            "contents glob plus negations (current)",
            [(False, "data/*"), (True, "data/corpus.jsonl")],
            False,
        ),
    ],
)
def test_dockerignore_semantics_match_measured_behaviour(
    label: str, rules: list[tuple[bool, str]], expected_excluded: bool
) -> None:
    """Only the first form keeps the corpus out; both negated forms let it in."""
    assert _excluded_by(rules, "data/corpus.jsonl") is expected_excluded, label


def test_unnegated_siblings_under_data_stay_excluded() -> None:
    assert _excluded_by(
        [(False, "data/*"), (True, "data/corpus.jsonl")],
        "data/eval/domain_relevance.json",
    )


# ---------------------------------------------------------------------------
# Healthchecks must name routes that exist
# ---------------------------------------------------------------------------


def _retrieval_app():
    from src.internal.servers.retrieval.demo import create_app

    class _Retriever:
        def retrieve(self, queries, topk=5):
            return [[] for _ in queries]

    return create_app(_Retriever())


def _app_routes(app) -> set[tuple[str, str]]:
    return {
        (route.path, method)
        for route in app.routes
        if hasattr(route, "methods")
        for method in route.methods
    }


def test_retrieval_healthcheck_names_a_route_the_app_serves():
    """Registered by create_base_app, not by demo.py -- a grep of the file misses it."""
    paths = _healthcheck_paths(_compose()["services"]["retrieval"])
    assert paths, "the retrieval service declares no healthcheck URL"

    routes = _app_routes(_retrieval_app())
    for path in paths:
        assert (path, "GET") in routes, (
            f"compose healthchecks {path} but the retrieval app serves "
            f"{sorted(p for p, m in routes if m == 'GET')}"
        )


def test_retrieval_healthcheck_actually_returns_ok():
    """Route existence is not the same as a 200; the healthcheck uses curl -f."""
    from fastapi.testclient import TestClient

    client = TestClient(_retrieval_app())
    for path in _healthcheck_paths(_compose()["services"]["retrieval"]):
        response = client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}"


# ---------------------------------------------------------------------------
# The corpus the retrieval command names must reach the container
# ---------------------------------------------------------------------------


def _retrieval_corpus_arg() -> str | None:
    command = _compose()["services"]["retrieval"].get("command") or ""
    if isinstance(command, list):
        parts = command
    else:
        parts = shlex.split(command)
    for flag in ("--corpus_path", "--corpus"):
        if flag in parts:
            return parts[parts.index(flag) + 1]
    return None


def test_the_retrieval_corpus_is_present_in_the_build_context():
    """A path under a volume mount or excluded from the context never arrives.

    The corpus is tracked in git, which makes it look available. `.dockerignore`
    excluding `data` and compose mounting an empty volume over `/data` are both
    invisible to that check.
    """
    corpus = _retrieval_corpus_arg()
    assert corpus, "the retrieval service passes no corpus argument"

    if not corpus.startswith("/"):
        pytest.skip("a registered corpus name, not a path")

    volume_targets = {
        entry.split(":")[1].rstrip("/")
        for entry in _compose()["services"]["retrieval"].get("volumes", [])
        if ":" in entry
    }
    for target in volume_targets:
        assert not corpus.startswith(target + "/"), (
            f"{corpus} sits under the {target} volume mount, which shadows "
            f"whatever the image baked there"
        )

    assert corpus.startswith(IMAGE_WORKDIR + "/"), (
        f"{corpus} is not under the image WORKDIR {IMAGE_WORKDIR}, so nothing "
        f"in the build context can supply it"
    )
    rel = corpus[len(IMAGE_WORKDIR) + 1 :]
    assert (REPO / rel).exists(), f"{rel} does not exist in the repo"
    assert not _dockerignore_excludes(rel), (
        f"{rel} exists in the repo but .dockerignore excludes it from the image"
    )


# ---------------------------------------------------------------------------
# The editable install must register the package, not rely on the cwd
# ---------------------------------------------------------------------------


def test_the_package_is_installed_after_the_source_is_copied():
    """`pip install -e .` before `COPY . .` discovers zero packages.

    setuptools writes an editable finder with an empty MAPPING, so `import src`
    resolves only because WORKDIR puts the cwd on sys.path. Any process with a
    different cwd -- a subprocess, a worker, a one-off `docker run ... python`
    -- raises ModuleNotFoundError.
    """
    lines = DOCKERFILE.read_text().splitlines()
    copy_all = next(
        (i for i, line in enumerate(lines) if re.match(r"\s*COPY\s+\.\s+\.", line)),
        None,
    )
    editable = next(
        (
            i
            for i, line in enumerate(lines)
            if re.search(r"pip install[^\n]*-e\s+\.", line)
        ),
        None,
    )
    assert copy_all is not None, "Dockerfile never copies the source"
    assert editable is not None, "Dockerfile never installs the package"
    assert editable > copy_all, (
        f"`pip install -e .` is on line {editable + 1}, before `COPY . .` on line "
        f"{copy_all + 1}; the editable install would register no packages"
    )


# ---------------------------------------------------------------------------
# One worker is a constraint, not a default
# ---------------------------------------------------------------------------


def test_the_web_server_runs_a_single_worker():
    """Per-process ``app.state`` makes multi-worker silently wrong, not slow.

    ``ToolApprovalBroker`` (a client polls for a decision and must reach the
    worker that created it), ``RequestCaptureStore`` (read back by the Dev
    Console), the loaded search-agent model and the memory encoder all live on
    ``app.state``. Adding ``--workers`` breaks approvals intermittently, loses
    captures, and loads the model once per worker. Moving the broker and the
    capture store to the Redis already in the compose stack is the prerequisite.
    """

    def _uncommented(text: str) -> str:
        # The constraint is documented in these files, and the documentation
        # names the flag. Match directives, not prose.
        return "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("#")
        )

    sources = {
        "Dockerfile": _uncommented(DOCKERFILE.read_text()),
        "docker-compose.yml": _uncommented(COMPOSE.read_text()),
    }
    for name, text in sources.items():
        for flag in ("--workers", "-w "):
            assert flag not in text, (
                f"{name} passes {flag!r} to the web server. Per-process "
                f"app.state (tool approvals, request captures, the loaded model) "
                f"is not shared across workers -- move it to Redis first."
            )


def test_the_in_process_state_that_makes_that_true_still_exists():
    """If this state ever moves off app.state, the constraint above can be lifted.

    Asserted against the source rather than a live app: building one loads a
    model. A failure here is a prompt to re-check the worker constraint, not a
    defect in itself.
    """
    app_source = (REPO / "src" / "internal" / "servers" / "web" / "app.py").read_text()
    for attribute in ("tool_approval_broker", "request_captures"):
        assert f"app.state.{attribute}" in app_source, (
            f"app.state.{attribute} is gone -- re-evaluate the single-worker "
            f"constraint documented in the Dockerfile"
        )


def test_app_services_can_run_a_published_image():
    """`AGENTIC_SEARCH_IMAGE=ghcr.io/...:sha-<sha> docker compose up --no-build`
    runs a published tag (deploy / rollback); unset, `up --build` still builds
    and tags a local image exactly as before (docs/deploy.md)."""
    services = _compose()["services"]
    for name in ("retrieval", "web"):
        assert (
            services[name]["image"] == "${AGENTIC_SEARCH_IMAGE:-agentic-search:local}"
        )
        assert services[name]["build"]["dockerfile"] == "Dockerfile"
