"""The shipped Prometheus alert rules: valid, tested, and in step with the
exporter (spec: 2026-09-25-prometheus-alert-rules-design.md)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
RULES_DIR = REPO / "deploy" / "prometheus"
RULES = RULES_DIR / "agentic-search-alerts.yml"
RULE_TESTS = RULES_DIR / "agentic-search-alerts.test.yml"
EXPORTER = REPO / "src" / "internal" / "observability" / "prometheus.py"
_HISTOGRAM_SUFFIXES = ("_sum", "_count", "_bucket")

promtool = shutil.which("promtool")
needs_promtool = pytest.mark.skipif(promtool is None, reason="promtool not on PATH")


def _alerts() -> list[dict]:
    doc = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    return [
        rule for group in doc["groups"] for rule in group["rules"] if "alert" in rule
    ]


@needs_promtool
def test_rules_are_valid():
    subprocess.run([promtool, "check", "rules", str(RULES)], check=True)


@needs_promtool
def test_rule_unit_tests_pass():
    subprocess.run(
        [promtool, "test", "rules", RULE_TESTS.name], check=True, cwd=RULES_DIR
    )


def test_every_alert_has_severity_and_annotations():
    alerts = _alerts()
    assert alerts
    for alert in alerts:
        assert alert["labels"]["severity"] in {"page", "ticket"}, alert["alert"]
        assert alert["annotations"]["summary"], alert["alert"]
        assert alert["annotations"]["description"], alert["alert"]


def test_every_alert_has_a_firing_and_a_quiet_test():
    tests = yaml.safe_load(RULE_TESTS.read_text(encoding="utf-8"))["tests"]
    firing: set[str] = set()
    quiet: set[str] = set()
    for case in tests:
        for check in case.get("alert_rule_test", []):
            (firing if check.get("exp_alerts") else quiet).add(check["alertname"])
    names = {alert["alert"] for alert in _alerts()}
    assert names - firing == set(), "alerts without a firing test"
    assert names - quiet == set(), "alerts without a quiet test"


def test_rules_only_reference_exported_metrics():
    exported = set(re.findall(r'"(agentic_search_[a-z_]+)"', EXPORTER.read_text()))
    referenced = set(re.findall(r"agentic_search_[a-z_]+", RULES.read_text()))

    def declared(name: str) -> bool:
        if name in exported:
            return True
        return any(
            name.endswith(suffix) and name[: -len(suffix)] in exported
            for suffix in _HISTOGRAM_SUFFIXES
        )

    assert referenced, "no metrics referenced"
    assert {name for name in referenced if not declared(name)} == set()
