from src.internal.routing.route import RetrieverTarget, Route, RouteDecision


def test_retriever_target_values():
    assert RetrieverTarget.SQL.value == "sql"
    assert {t.value for t in RetrieverTarget} == {
        "sparse",
        "dense",
        "hybrid",
        "metadata",
        "sql",
        "graph",
        "api",
    }


def test_route_is_frozen():
    r = Route("docs", "general docs", ("local",), RetrieverTarget.HYBRID)
    assert r.retriever is RetrieverTarget.HYBRID
    try:
        r.name = "x"  # type: ignore[misc]
        raise AssertionError("Route should be frozen")
    except Exception:
        pass


def test_route_decision_defaults():
    d = RouteDecision(
        domain="docs",
        sources=["local"],
        retriever=RetrieverTarget.HYBRID,
    )
    assert d.confidence == 1.0
    assert d.strategy == "heuristic"
