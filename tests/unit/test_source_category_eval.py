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
        self.url, self.error, self.timed_out = url, error, False


class _Stub:
    def __init__(self, by_query):
        self.by_query, self.calls = by_query, []

    async def __call__(self, query, **_kwargs):
        self.calls.append(query)
        return [
            u if isinstance(u, _Page) else _Page(u)
            for u in self.by_query.get(query, [])
        ]


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


def test_source_loader_rejects_duplicate_pairs(tmp_path):
    entry = {"query": "q", "authority_hosts": ["sec.gov"]}
    payload = {
        "taxonomy": "source_categories",
        "domains": {"financial": [entry, entry]},
    }
    with pytest.raises(ValueError, match="duplicate"):
        load_source_labels(_write(tmp_path, payload))


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


@pytest.mark.asyncio
async def test_failed_restricted_response_excludes_and_is_not_cached(tmp_path):
    scoped = restricted_query("etf fees", "financial")
    label = SourceLabelledQuery("financial", "etf fees", {"sec.gov"})
    ok = {"etf fees": ["https://a.example/1"], SUFFIX: ["https://a.example/1"]}
    cache = DiskCache(tmp_path)
    bad = _Stub({**ok, scoped: [_Page(None, error="x")]})
    out = await run_source_query(label, search_fn=bad, cache=cache)
    assert out.excluded
    fresh = _Stub({**ok, scoped: ["https://sec.gov/x"]})
    again = await run_source_query(label, search_fn=fresh, cache=cache)
    assert fresh.calls == [scoped]
    assert not again.excluded


@pytest.mark.asyncio
async def test_raising_restricted_search_excludes(tmp_path):
    scoped = restricted_query("etf fees", "financial")

    class _Raises(_Stub):
        async def __call__(self, query, **kw):
            if query == scoped:
                raise RuntimeError("boom")
            return await super().__call__(query, **kw)

    stub = _Raises(
        {"etf fees": ["https://a.example/1"], SUFFIX: ["https://a.example/1"]}
    )
    out = await run_source_query(
        SourceLabelledQuery("financial", "etf fees", {"sec.gov"}),
        search_fn=stub,
        cache=DiskCache(tmp_path),
    )
    assert out.excluded


@pytest.mark.asyncio
async def test_genuinely_empty_restricted_response_is_a_real_fallback(tmp_path):
    scoped = restricted_query("etf fees", "financial")
    stub = _Stub(
        {
            "etf fees": ["https://a.example/1"],
            SUFFIX: ["https://a.example/1"],
            scoped: [],
        }
    )
    out = await run_source_query(
        SourceLabelledQuery("financial", "etf fees", {"sec.gov"}),
        search_fn=stub,
        cache=DiskCache(tmp_path),
    )
    assert not out.excluded and out.fallback_fired


@pytest.mark.asyncio
async def test_failed_suffix_arm_is_excluded_and_not_cached(tmp_path):
    cache = DiskCache(tmp_path)
    label = SourceLabelledQuery("financial", "etf fees", {"sec.gov"})
    stub = _Stub(
        {"etf fees": ["https://a.example/1"], SUFFIX: [_Page(None, error="x")]}
    )
    out = await run_source_query(label, search_fn=stub, cache=cache)
    assert out.excluded and out.suffix.error
    from src.internal.retrieval.domain_eval import cache_key

    assert cache.get(cache_key(SUFFIX, provider="serpapi")) is None


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
    ctl = [_outcome("web", 0.0)]
    worse = cli.summarise_source(ctl + [_outcome("financial", -0.5) for _ in range(12)])
    assert worse["decision"] == "worse"
    better = cli.summarise_source(ctl + [_outcome("financial", 0.5) for _ in range(12)])
    assert better["decision"] == "better"
    null = cli.summarise_source(
        ctl + [_outcome("financial", d) for d in (0.1, -0.1) * 6]
    )
    assert null["decision"] == "null"
    assert null["control_ok"] is True


def test_decision_is_withheld_without_a_passing_control():
    cli = _cli()
    rows = [_outcome("financial", 0.5) for _ in range(12)]
    none = cli.summarise_source(rows)
    assert none["decision"] is None and none["control_ok"] is False
    bad = cli.summarise_source([_outcome("web", 0.1)] + rows)
    assert bad["decision"] is None and bad["control_ok"] is False


def test_failed_controls_do_not_pass_the_gate():
    # A failed control arm scores delta 0 and Jaccard 1 by construction.
    cli = _cli()
    rows = [_outcome("financial", 0.5) for _ in range(12)]
    failed = [_outcome("web", 0.0, excluded=True) for _ in range(2)]
    out = cli.summarise_source(failed + rows)
    assert out["control_ok"] is False and out["decision"] is None
    mixed = cli.summarise_source([_outcome("web", 0.0)] + failed[:1] + rows)
    assert mixed["control_ok"] is False and mixed["decision"] is None


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
    assert summary["fallback_added_seconds_n"] == 1
    assert summary["on_list_share"] == 0.25


# --- pilot flags: --per-category and --no-plain ------------------------------


def test_per_category_keeps_the_first_n_in_file_order():
    cli = _cli()
    labels = [
        SourceLabelledQuery("academic", "a1", {"x"}),
        SourceLabelledQuery("code", "c1", {"x"}),
        SourceLabelledQuery("academic", "a2", {"x"}),
        SourceLabelledQuery("code", "c2", {"x"}),
        SourceLabelledQuery("academic", "a3", {"x"}),
    ]
    assert [x.query for x in cli.select_labels(labels, 1)] == ["a1", "c1"]
    assert [x.query for x in cli.select_labels(labels, 2)] == ["a1", "c1", "a2", "c2"]
    assert cli.select_labels(labels, None) == labels


def test_per_category_one_over_the_frozen_file_is_one_query_each():
    cli = _cli()
    picked = cli.select_labels(load_source_labels(), 1)
    assert len(picked) == 16
    assert len({x.domain for x in picked}) == 16


@pytest.mark.asyncio
async def test_no_plain_never_calls_the_plain_query(tmp_path):
    scoped = restricted_query("etf fees", "financial")
    stub = _Stub({SUFFIX: ["https://blog.example/a"], scoped: ["https://sec.gov/x"]})
    label = SourceLabelledQuery("financial", "etf fees", {"sec.gov"})
    out = await run_source_query(
        label, search_fn=stub, cache=DiskCache(tmp_path), include_plain=False
    )
    assert "etf fees" not in stub.calls
    assert stub.calls == [SUFFIX, scoped]
    assert out.plain is None and out.delta_vs_plain is None
    # A skipped plain arm is not a failed arm.
    assert not out.excluded


def test_pending_calls_counts_the_pilot_selection(tmp_path):
    cli = _cli()
    labels = cli.select_labels(load_source_labels(), 1)
    labels += [SourceLabelledQuery("web", "what is a bond", {"example.com"})]
    cache = DiskCache(tmp_path)
    assert cli._pending_calls(labels, cache, "serpapi", include_plain=False) == 33
    assert cli._pending_calls(labels, cache, "serpapi", include_plain=True) == 49


def test_summary_without_plain_reports_it_as_not_run():
    from src.internal.retrieval.domain_eval import ArmResult, SourceQueryOutcome

    cli = _cli()
    arm = ArmResult(urls=["u"], empty=False, seconds=1.0)

    def row(domain, delta):
        return SourceQueryOutcome(
            domain,
            "q",
            None,
            arm,
            arm,
            False,
            delta,
            None,
            1.0 if domain == "web" else 0.5,
            False,
            0.25,
        )

    summary = cli.summarise_source([row("web", 0.0)] + [row("financial", 0.5)] * 12)
    assert summary["paired_vs_plain"] is None
    assert summary["empty_rate"]["plain"] is None
    assert summary["paired_vs_suffix"]["n"] == 12
    assert summary["decision"] == "better"


def test_main_dry_run_reports_the_pilot_call_count(tmp_path, capsys):
    cli = _cli()
    cli.main(
        [
            "--per-category",
            "1",
            "--no-plain",
            "--control",
            "what is a bond",
            "--cache-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert (
        "17 labelled queries; 33 live provider calls needed." in capsys.readouterr().out
    )
