"""Offline tests for the domain relevance evaluation.

No test in this file opens a socket: the runner is driven with a stub
provider so the suite stays fast and spends no SerpAPI quota.
"""

from __future__ import annotations

import pytest

from src.internal.retrieval.domain_eval import (
    authority_precision,
    host_matches,
    jaccard,
)


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
