"""Offline tests for the domain relevance evaluation.

No test in this file opens a socket: the runner is driven with a stub
provider so the suite stays fast and spends no SerpAPI quota.
"""

from __future__ import annotations

import pytest

from src.internal.retrieval.domain_eval import (
    DEFAULT_LABELS_PATH,
    authority_precision,
    host_matches,
    jaccard,
    load_labels,
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
