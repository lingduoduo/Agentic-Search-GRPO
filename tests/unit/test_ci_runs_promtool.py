"""CI runs the Prometheus alert-rule tests with a checksum-verified promtool,
instead of skipping them (spec: 2026-09-25-breaker-readiness-metrics-design.md)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CI = REPO / ".github" / "workflows" / "ci.yml"


def _job() -> dict:
    jobs = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]
    assert "alert-rules" in jobs, "no alert-rules job in ci.yml"
    return jobs["alert-rules"]


def _script() -> str:
    return "\n".join(step.get("run", "") for step in _job()["steps"])


def test_ci_downloads_a_pinned_promtool_and_verifies_its_checksum():
    env = _job()["env"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["PROMETHEUS_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["PROMETHEUS_SHA256"]), "no pinned sha256"
    script = _script()
    assert "${PROMETHEUS_VERSION}" in script
    assert "${PROMETHEUS_SHA256}" in script
    assert "sha256sum -c" in script


def test_ci_runs_the_alert_rule_tests():
    script = _script()
    assert "promtool check rules" in script
    assert "promtool test rules" in script
