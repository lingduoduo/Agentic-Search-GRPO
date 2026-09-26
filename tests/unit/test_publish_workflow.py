"""The image publish workflow: only CI-passed main commits and release tags,
least privilege, immutable sha tags, one multi-arch manifest built natively per
platform (specs: 2026-09-25-publish-images-ghcr-design.md,
2026-09-25-multi-arch-images-design.md)."""

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


def _job(name: str) -> dict:
    jobs = _workflow()["jobs"]
    assert set(jobs) == {"build", "merge"}
    return jobs[name]


def _step(job: str, uses_prefix: str) -> dict:
    matches = [
        s for s in _job(job)["steps"] if str(s.get("uses", "")).startswith(uses_prefix)
    ]
    assert len(matches) == 1, (job, uses_prefix)
    return matches[0]


def _run_script(job: str) -> str:
    return "\n".join(s.get("run", "") for s in _job(job)["steps"])


# --- when it publishes ------------------------------------------------------


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
    condition = _job("build")["if"]
    assert "github.event.workflow_run.conclusion == 'success'" in condition
    assert "github.event_name == 'push'" in condition


def test_checks_out_the_commit_ci_tested_not_the_branch_tip():
    ref = _step("build", "actions/checkout@")["with"]["ref"]
    assert "github.event.workflow_run.head_sha" in ref
    assert "github.sha" in ref  # tag pushes have no workflow_run payload


def test_least_privilege_permissions():
    assert _workflow()["permissions"] == {"contents": "read", "packages": "write"}


# --- build: one native leg per platform ----------------------------------------


def test_builds_each_platform_natively_on_its_own_runner():
    build = _job("build")
    legs = {
        (leg["platform"], leg["runner"])
        for leg in build["strategy"]["matrix"]["include"]
    }
    assert legs == {
        ("linux/amd64", "ubuntu-latest"),
        ("linux/arm64", "ubuntu-24.04-arm"),
    }
    assert build["runs-on"] == "${{ matrix.runner }}"
    uses = [str(s.get("uses", "")) for s in build["steps"]]
    assert not any(u.startswith("docker/setup-qemu-action") for u in uses), (
        "no emulation"
    )


def test_each_leg_pushes_the_root_dockerfile_by_digest():
    build = _step("build", "docker/build-push-action@")["with"]
    assert build["platforms"] == "${{ matrix.platform }}"
    assert build["context"] == "."
    assert build.get("file", "./Dockerfile") in {"Dockerfile", "./Dockerfile"}
    assert "push-by-digest=true" in build["outputs"]
    assert "push=true" in build["outputs"]

    # The revision label names the tested commit, not github.sha (the moving
    # branch tip for workflow_run). Set explicitly: metadata-action's
    # `context: git` fails on the detached HEAD a head_sha checkout leaves
    # ("Cannot find detached HEAD ref"), which broke every publish.
    meta = _step("build", "docker/metadata-action@")["with"]
    assert "context" not in meta
    assert (
        "org.opencontainers.image.revision="
        "${{ github.event.workflow_run.head_sha || github.sha }}"
    ) in meta["labels"]

    upload = _step("build", "actions/upload-artifact@")["with"]
    assert upload["name"].startswith("digests-")

    login = _step("build", "docker/login-action@")["with"]
    assert login["registry"] == "ghcr.io"
    assert login["password"] == "${{ secrets.GITHUB_TOKEN }}"


# --- merge: one tagged manifest list ----------------------------------------


def test_merge_waits_for_every_platform():
    assert _job("merge")["needs"] in ("build", ["build"])


def test_merge_publishes_immutable_sha_tags_as_one_manifest():
    meta = _step("merge", "docker/metadata-action@")["with"]
    assert meta["images"].strip() == IMAGE
    # The rollback handle names the commit CI tested. metadata-action's
    # type=sha reads github.sha, which for workflow_run is the branch tip when
    # the publish starts -- a later merge would mis-tag this image.
    assert "type=sha" not in meta["tags"]
    assert (
        "type=raw,value=sha-${{ github.event.workflow_run.head_sha || github.sha }}"
        in meta["tags"]
    )
    assert "type=semver,pattern={{version}}" in meta["tags"]
    _step("merge", "actions/download-artifact@")
    assert "docker buildx imagetools create" in _run_script("merge")

    login = _step("merge", "docker/login-action@")["with"]
    assert login["password"] == "${{ secrets.GITHUB_TOKEN }}"


@pytest.mark.skipif(shutil.which("actionlint") is None, reason="actionlint not on PATH")
def test_actionlint_is_clean():
    subprocess.run(["actionlint", str(WORKFLOW)], check=True, cwd=REPO)
