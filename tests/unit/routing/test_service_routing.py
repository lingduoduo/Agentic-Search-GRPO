import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.router import Router


class _StubBackend:
    def search_sparse(self, query, top_k, filters=None):
        return [RetrievalResult(doc_id="d1", title="t", text="x", url=None, score=1.0)]

    def search_dense(self, query, top_k, filters=None):
        raise NotImplementedError


class _StubCache:
    """A result cache that always hits, so the cached exit is exercised."""

    def __init__(self, results):
        self._results = results

    def get(self, query, filters, top_k):
        return list(self._results)

    def set(self, query, filters, top_k, results):
        pass


def _routed_service(**kwargs):
    return RetrievalService(
        _StubBackend(), router=Router(RouteRegistry(DEFAULT_ROUTES)), **kwargs
    )


def test_routing_disabled_runs_retrieval():
    svc = RetrievalService(_StubBackend())  # no router
    results, mode = svc.search("how many docs are there", top_k=3)
    assert results and results[0].doc_id == "d1"
    assert "routed:" not in mode


def test_routing_to_hybrid_runs_retrieval():
    svc = _routed_service()
    results, mode = svc.search("what is reciprocal rank fusion", top_k=3)
    assert results and results[0].doc_id == "d1"
    assert "routed:" not in mode


@pytest.mark.parametrize(
    "query,target",
    [
        ("how many papers per year", "sql"),
        ("papers related to BM25", "graph"),
        ("latest FAISS release", "api"),
    ],
)
def test_unbacked_target_still_returns_results(query, target):
    """An unbacked route annotates the search; it must not cost the results."""
    svc = _routed_service()
    results, mode = svc.search(query, top_k=3)
    assert results and results[0].doc_id == "d1"
    assert mode.endswith(f"+routed:{target}")


def test_unbacked_target_annotates_a_cached_hit():
    """The cache exit reports the routing decision like the uncached one."""
    cached = [RetrievalResult(doc_id="c1", title="t", text="x", url=None, score=1.0)]
    svc = _routed_service(result_cache=_StubCache(cached))
    results, mode = svc.search("how many papers per year", top_k=3)
    assert results and results[0].doc_id == "c1"
    assert mode == "cached+routed:sql"
