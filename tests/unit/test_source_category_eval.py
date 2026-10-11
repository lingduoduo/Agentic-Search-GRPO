"""Offline tests for the three-arm source-category evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.internal.retrieval.domain_eval import (
    DiskCache,
    SourceLabelledQuery,
    load_labels,
    load_source_labels,
    on_list_share,
    run_source_query,
)
from src.internal.tools.search import restricted_query

SUFFIX = "etf fees financial market data filings"


class _Page:
    def __init__(self, url, error=None):
        self.url, self.error = url, error


class _Stub:
    def __init__(self, by_query):
        self.by_query, self.calls = by_query, []

    async def __call__(self, query, **_kwargs):
        self.calls.append(query)
        return [_Page(u) for u in self.by_query.get(query, [])]


def _write(tmp_path, payload):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload))
    return path


def test_source_loader_requires_the_marker(tmp_path):
    with pytest.raises(ValueError, match="source_categories"):
        load_source_labels(_write(tmp_path, {"domains": {}}))


def test_source_loader_rejects_categories_without_sites(tmp_path):
    payload = {
        "taxonomy": "source_categories",
        "domains": {"product": [{"query": "q", "authority_hosts": ["a.com"]}]},
    }
    with pytest.raises(ValueError, match="source_sites"):
        load_source_labels(_write(tmp_path, payload))


def test_historical_loader_rejects_a_marked_file(tmp_path):
    payload = {"taxonomy": "source_categories", "domains": {}}
    with pytest.raises(ValueError, match="load_source_labels"):
        load_labels(_write(tmp_path, payload))


def test_on_list_share_counts_subdomains_and_paths():
    assert on_list_share("financial", {"www.sec.gov", "blog.example"}) == 0.5
    assert on_list_share("jobs", {"linkedin.com"}) == 1.0


@pytest.mark.asyncio
async def test_three_arms_and_a_cached_fallback(tmp_path):
    scoped = restricted_query("etf fees", "financial")
    stub = _Stub(
        {
            "etf fees": ["https://blog.example/a"],
            SUFFIX: ["https://blog.example/a", "https://blog.example/b"],
            scoped: ["https://sec.gov/x"],
        }
    )
    label = SourceLabelledQuery("financial", "etf fees", {"sec.gov"})
    out = await run_source_query(label, search_fn=stub, cache=DiskCache(tmp_path))
    # plain, suffix, scoped; the fallback (= suffix) is a cache hit.
    assert stub.calls == ["etf fees", SUFFIX, scoped]
    assert out.fallback_fired
    # One restricted hit (< 3) is topped up from the cached suffix response.
    assert out.category.urls == [
        "https://sec.gov/x",
        "https://blog.example/a",
        "https://blog.example/b",
    ]
    assert out.delta_vs_plain == pytest.approx(1 / 3)
    assert out.delta_vs_suffix == pytest.approx(1 / 3)
    assert out.suffix.seconds is not None


@pytest.mark.asyncio
async def test_web_control_has_identical_arms(tmp_path):
    stub = _Stub({"q": ["https://a.example/1"]})
    out = await run_source_query(
        SourceLabelledQuery("web", "q", {"a.example"}),
        search_fn=stub,
        cache=DiskCache(tmp_path),
    )
    assert stub.calls == ["q"]
    assert out.delta_vs_suffix == 0.0 and out.jaccard_vs_suffix == 1.0


@pytest.mark.asyncio
async def test_category_arm_ignores_the_operator_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    scoped = restricted_query("etf fees", "financial")
    stub = _Stub({})
    await run_source_query(
        SourceLabelledQuery("financial", "etf fees", {"sec.gov"}),
        search_fn=stub,
        cache=DiskCache(tmp_path),
    )
    assert scoped in stub.calls


def _cli():
    path = (
        Path(__file__).resolve().parents[2] / "examples" / "run_source_category_eval.py"
    )
    spec = importlib.util.spec_from_file_location("run_source_category_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outcome(domain, d_suffix, *, excluded=False, fallback=False):
    from src.internal.retrieval.domain_eval import ArmResult, SourceQueryOutcome

    arm = ArmResult(urls=["u"], empty=False, seconds=1.0)
    return SourceQueryOutcome(
        domain,
        "q",
        arm,
        arm,
        arm,
        fallback,
        d_suffix,
        d_suffix,
        1.0 if domain == "web" else 0.5,
        excluded,
        0.25,
    )


def test_summary_applies_the_preregistered_rule():
    cli = _cli()
    worse = cli.summarise_source([_outcome("financial", -0.5) for _ in range(12)])
    assert worse["decision"] == "worse"
    better = cli.summarise_source([_outcome("financial", 0.5) for _ in range(12)])
    assert better["decision"] == "better"
    null = cli.summarise_source([_outcome("financial", d) for d in (0.1, -0.1) * 6])
    assert null["decision"] == "null"


def test_summary_excludes_controls_and_excluded_queries():
    cli = _cli()
    summary = cli.summarise_source(
        [_outcome("web", 0.0)]
        + [_outcome("financial", 0.5)] * 3
        + [_outcome("legal", 0.9, excluded=True)]
    )
    assert summary["paired_vs_suffix"]["n"] == 3
    assert summary["control"]["n"] == 1
    assert summary["excluded_count"] == 1


def test_summary_reports_fallback_rate_and_circularity_share():
    cli = _cli()
    summary = cli.summarise_source(
        [_outcome("financial", 0.1, fallback=True), _outcome("financial", 0.1)]
    )
    assert summary["fallback_rate"] == 0.5
    assert summary["fallback_added_seconds"] == 1.0
    assert summary["on_list_share"] == 0.25
