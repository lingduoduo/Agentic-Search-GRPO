"""Offline tests for the domain relevance evaluation.

No test in this file opens a socket: the runner is driven with a stub
provider so the suite stays fast and spends no SerpAPI quota.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.internal.retrieval.domain_eval import (
    DEFAULT_LABELS_PATH,
    DiskCache,
    LabelledQuery,
    authority_precision,
    host_matches,
    jaccard,
    load_labels,
    run_query,
)
from src.internal.tools.search import AVAILABLE_DOMAINS


@pytest.mark.parametrize(
    "host,pattern,expected",
    [
        ("stanford.edu", "stanford.edu", True),
        ("cs.stanford.edu", "stanford.edu", True),
        # A suffix collision is not a subdomain: the boundary is a dot.
        ("notstanford.edu", "stanford.edu", False),
        ("stanford.edu.evil.com", "stanford.edu", False),
        ("ARXIV.ORG", "arxiv.org", True),
        ("", "arxiv.org", False),
    ],
)
def test_host_matches_requires_a_dot_boundary(host, pattern, expected):
    assert host_matches(host, pattern) is expected


def test_authority_precision_all_and_none():
    hosts = {"arxiv.org"}
    assert (
        authority_precision(["https://arxiv.org/abs/1", "http://arxiv.org/x"], hosts)
        == 1.0
    )
    assert authority_precision(["https://example.com/a"], hosts) == 0.0


def test_authority_precision_is_a_share_of_results():
    hosts = {"arxiv.org"}
    urls = [
        "https://arxiv.org/abs/1",
        "https://example.com/a",
        "https://b.com/c",
        "https://d.com",
    ]
    assert authority_precision(urls, hosts) == 0.25


def test_authority_precision_of_no_results_is_zero():
    # An empty arm scores zero rather than raising, so an empty provider
    # response and a wholly irrelevant one are comparable.
    assert authority_precision([], {"arxiv.org"}) == 0.0


def test_authority_precision_ignores_unparseable_urls():
    assert (
        authority_precision(["not a url", "https://arxiv.org/x"], {"arxiv.org"}) == 0.5
    )


def test_jaccard_identical_and_disjoint():
    assert jaccard(["a", "b"], ["b", "a"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0


def test_jaccard_of_two_empty_arms_is_one():
    # Two arms that both returned nothing are identical, not undefined.
    assert jaccard([], []) == 1.0


def test_labels_cover_every_topic_domain():
    # The taxonomy and the evaluation must not drift apart. `general` has an
    # empty hint and is carried separately as a control.
    labels = load_labels(DEFAULT_LABELS_PATH)
    covered = {lab.domain for lab in labels}
    expected = {d for d in AVAILABLE_DOMAINS if d != "general"}
    assert covered == expected


def test_every_domain_has_three_queries():
    labels = load_labels(DEFAULT_LABELS_PATH)
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab.domain] = counts.get(lab.domain, 0) + 1
    assert set(counts.values()) == {3}, counts


def test_every_query_carries_authority_hosts():
    for lab in load_labels(DEFAULT_LABELS_PATH):
        assert lab.authority_hosts, f"{lab.domain}: {lab.query}"


def test_missing_label_file_fails_loudly(tmp_path):
    # Failing before any provider call beats silently evaluating fewer domains.
    with pytest.raises(FileNotFoundError) as exc:
        load_labels(tmp_path / "absent.json")
    assert "absent.json" in str(exc.value)


def test_malformed_label_file_names_the_path(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError) as exc:
        load_labels(bad)
    assert "bad.json" in str(exc.value)


class _Page:
    def __init__(self, url):
        self.url = url
        self.error = None


class _StubSearch:
    """Stands in for search_tool; records every query it is asked for."""

    def __init__(self, by_query: dict[str, list[str]], fail_on: str | None = None):
        self.by_query = by_query
        self.fail_on = fail_on
        self.calls: list[str] = []

    async def __call__(self, query, **_kwargs):
        self.calls.append(query)
        if self.fail_on is not None and query == self.fail_on:
            raise RuntimeError("provider exploded")
        return [_Page(u) for u in self.by_query.get(query, [])]


@pytest.mark.asyncio
async def test_run_query_scores_both_arms(tmp_path):
    label = LabelledQuery("academic", "attention mechanism", {"arxiv.org"})
    stub = _StubSearch(
        {
            "attention mechanism": ["https://blog.com/a", "https://b.com/c"],
            "attention mechanism academic research": [
                "https://arxiv.org/abs/1",
                "https://b.com/c",
            ],
        }
    )
    out = await run_query(label, search_fn=stub, cache=DiskCache(tmp_path))
    assert out.general.urls == ["https://blog.com/a", "https://b.com/c"]
    assert out.delta == pytest.approx(0.5)
    assert out.jaccard == pytest.approx(1 / 3)
    assert out.excluded is False


@pytest.mark.asyncio
async def test_a_failed_arm_excludes_the_query(tmp_path):
    # A delta against a failed call measures the failure, not the domain.
    label = LabelledQuery("academic", "q", {"arxiv.org"})
    stub = _StubSearch({"q": ["https://arxiv.org/a"]}, fail_on="q academic research")
    out = await run_query(label, search_fn=stub, cache=DiskCache(tmp_path))
    assert out.excluded is True
    assert out.domain_arm.error is not None


@pytest.mark.asyncio
async def test_an_empty_arm_excludes_the_query(tmp_path):
    label = LabelledQuery("academic", "q", {"arxiv.org"})
    stub = _StubSearch({"q": ["https://arxiv.org/a"], "q academic research": []})
    out = await run_query(label, search_fn=stub, cache=DiskCache(tmp_path))
    assert out.domain_arm.empty is True
    assert out.excluded is True


@pytest.mark.asyncio
async def test_the_cache_prevents_a_second_provider_call(tmp_path):
    label = LabelledQuery("academic", "q", {"arxiv.org"})
    cache = DiskCache(tmp_path)
    stub = _StubSearch(
        {"q": ["https://arxiv.org/a"], "q academic research": ["https://arxiv.org/b"]}
    )
    await run_query(label, search_fn=stub, cache=cache)
    assert len(stub.calls) == 2
    await run_query(label, search_fn=stub, cache=cache)
    assert len(stub.calls) == 2, "second run must be served from cache"


@pytest.mark.asyncio
async def test_general_control_pairs_identical_arms(tmp_path):
    # An empty hint means both arms issue the same query, so the second is a
    # cache hit and the delta is structurally zero.
    label = LabelledQuery("general", "q", {"arxiv.org"})
    cache = DiskCache(tmp_path)
    stub = _StubSearch({"q": ["https://arxiv.org/a"]})
    out = await run_query(label, search_fn=stub, cache=cache)
    assert len(stub.calls) == 1
    assert out.delta == 0.0
    assert out.jaccard == 1.0


def test_the_measurement_core_stays_torch_free():
    """Importing post_training pulls torch in via package __init__ side effects.

    A retrieval-side module that did that would drop out of the torch-free CI
    job, which has broken silently here before. Run in a subprocess so the
    check cannot be fooled by torch already living in this session's modules.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import src.internal.retrieval.domain_eval; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
    )
    assert result.returncode == 0, "domain_eval must not import torch"
