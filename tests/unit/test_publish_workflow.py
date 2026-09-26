"""The image publish workflow: only CI-passed main commits and release tags,
least privilege, immutable sha tags (spec: 2026-09-25-publish-images-ghcr-design.md)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
IMAGE = "ghcr.io/${{ github.repository_owner }}/agentic-search"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers() -> dict:
    doc = _workflow()
    return doc.get("on", doc.get(True))  # PyYAML reads the bare key `on` as True


def _steps() -> list[dict]:
    (job,) = _workflow()["jobs"].values()
    return job["steps"]


def _step(uses_prefix: str) -> dict:
    matches = [s for s in _steps() if str(s.get("uses", "")).startswith(uses_prefix)]
    assert len(matches) == 1, uses_prefix
    return matches[0]


def test_publishes_only_ci_passed_main_commits_and_release_tags():
    triggers = _triggers()
    assert set(triggers) == {"workflow_run", "push"}
    run = triggers["workflow_run"]
    assert run["workflows"] == ["CI"]
    assert run["types"] == ["completed"]
    assert run["branches"] == ["main"]
    assert triggers["push"] == {"tags": ["v*"]}


def test_never_publishes_from_a_pull_request():
    assert "pull_request" not in _triggers()
    assert "pull_request_target" not in _triggers()


def test_a_main_build_requires_ci_success():
    (job,) = _workflow()["jobs"].values()
    condition = job["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in condition
    assert "github.event_name == 'push'" in condition


def test_checks_out_the_commit_ci_tested_not_the_branch_tip():
    checkout = _step("actions/checkout@")
    ref = checkout["with"]["ref"]
    assert "github.event.workflow_run.head_sha" in ref
    assert "github.sha" in ref  # tag pushes have no workflow_run payload


def test_least_privilege_permissions():
    assert _workflow()["permissions"] == {"contents": "read", "packages": "write"}


def test_pushes_immutable_sha_tags_of_the_root_dockerfile():
    meta = _step("docker/metadata-action@")["with"]
    assert meta["images"].strip() == IMAGE
    tags = meta["tags"]
    assert "type=sha,format=long" in tags
    assert "type=semver,pattern={{version}}" in tags

    build = _step("docker/build-push-action@")["with"]
    assert build["push"] is True
    assert build["context"] == "."
    assert build.get("file", "./Dockerfile") in {"Dockerfile", "./Dockerfile"}
    assert "steps.meta.outputs.tags" in build["tags"]

    login = _step("docker/login-action@")["with"]
    assert login["registry"] == "ghcr.io"
    assert login["password"] == "${{ secrets.GITHUB_TOKEN }}"


@pytest.mark.skipif(shutil.which("actionlint") is None, reason="actionlint not on PATH")
def test_actionlint_is_clean():
    subprocess.run(["actionlint", str(WORKFLOW)], check=True, cwd=REPO)
