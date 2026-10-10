from __future__ import annotations

from unittest.mock import MagicMock

from src.context.query_enhancer import QueryBundle, QueryEnhancer


def _llm(response: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = response
    return m


# ---------------------------------------------------------------------------
# QueryBundle
# ---------------------------------------------------------------------------


def test_query_bundle_all_queries_deduped():
    b = QueryBundle(
        original="what is FAISS?",
        sub_queries=[
            "what is FAISS?",
            "FAISS vector search",
            "facebook AI similarity search",
        ],
        hyde_text="FAISS is a library for efficient similarity search developed by Facebook AI.",
    )
    queries = b.all_queries()
    assert queries[0] == "what is FAISS?"
    assert "FAISS vector search" in queries
    assert "facebook AI similarity search" in queries
    assert "FAISS is a library" in queries[-1]
    # original not duplicated
    assert queries.count("what is FAISS?") == 1


def test_bundle_retains_original_and_normalizes_variants():
    bundle = QueryBundle(
        original="Original query",
        sub_queries=[" original  QUERY ", " ", "detail"],
        hyde_text="DETAIL",
        step_back_query="background",
    )
    assert bundle.all_queries() == ["Original query", "detail", "background"]


def test_blank_bundle_has_no_queries():
    assert QueryBundle(original=" ", sub_queries=["", "\t"]).all_queries() == []


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


def test_decompose_returns_sub_queries():
    llm = _llm(
        "FAISS vector search\nfacebook AI similarity search\nnearest neighbour index"
    )
    enhancer = QueryEnhancer(llm)
    result = enhancer.decompose("what is FAISS and how does it compare to ScaNN?")
    assert len(result) >= 1
    assert all(isinstance(q, str) and q for q in result)


def test_decompose_excludes_original():
    query = "what is FAISS?"
    llm = _llm(f"{query}\nFAISS vector search")
    enhancer = QueryEnhancer(llm)
    result = enhancer.decompose(query)
    assert query not in result


def test_decompose_returns_original_when_llm_none():
    enhancer = QueryEnhancer(llm=None)
    result = enhancer.decompose("what is FAISS?")
    assert result == ["what is FAISS?"]


def test_decompose_returns_original_on_llm_failure():
    llm = _llm.__wrapped__ if hasattr(_llm, "__wrapped__") else MagicMock()
    llm.complete.side_effect = RuntimeError("LLM unavailable")
    enhancer = QueryEnhancer(llm)
    result = enhancer.decompose("what is FAISS?")
    assert result == ["what is FAISS?"]


def test_decompose_returns_original_on_empty_response():
    enhancer = QueryEnhancer(_llm(""))
    result = enhancer.decompose("what is FAISS?")
    assert result == ["what is FAISS?"]


# ---------------------------------------------------------------------------
# hyde
# ---------------------------------------------------------------------------


def test_hyde_returns_hypothetical_text():
    hypothetical = "FAISS is a library developed by Facebook AI Research for efficient similarity search."
    enhancer = QueryEnhancer(_llm(hypothetical))
    result = enhancer.hyde("what is FAISS?")
    assert result == hypothetical


def test_hyde_returns_none_when_llm_none():
    enhancer = QueryEnhancer(llm=None)
    assert enhancer.hyde("what is FAISS?") is None


def test_hyde_returns_none_on_llm_failure():
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("timeout")
    enhancer = QueryEnhancer(llm)
    assert enhancer.hyde("what is FAISS?") is None


# ---------------------------------------------------------------------------
# enhance
# ---------------------------------------------------------------------------


def test_enhance_returns_bundle():
    llm = MagicMock()
    llm.complete.side_effect = [
        "FAISS vector search\nFacebook similarity search",  # decompose call
        "FAISS is a fast similarity search library.",  # hyde call
        "What are vector similarity search algorithms?",  # step_back call
    ]
    enhancer = QueryEnhancer(llm)
    bundle = enhancer.enhance("what is FAISS?")
    assert bundle.original == "what is FAISS?"
    assert len(bundle.sub_queries) >= 1
    assert bundle.hyde_text is not None


def test_enhance_no_llm_returns_passthrough_bundle():
    enhancer = QueryEnhancer(llm=None)
    bundle = enhancer.enhance("what is FAISS?")
    assert bundle.original == "what is FAISS?"
    assert bundle.sub_queries == ["what is FAISS?"]
    assert bundle.hyde_text is None


# ---------------------------------------------------------------------------
# step_back
# ---------------------------------------------------------------------------


def test_step_back_returns_broader_query():
    broader = "What are approximate nearest-neighbour search algorithms?"
    enhancer = QueryEnhancer(_llm(broader))
    result = enhancer.step_back(
        "How does FAISS handle GPU indexing for billion-scale datasets?"
    )
    assert result == broader


def test_step_back_returns_none_when_llm_none():
    enhancer = QueryEnhancer(llm=None)
    assert enhancer.step_back("any query") is None


def test_step_back_returns_none_on_llm_failure():
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("timeout")
    enhancer = QueryEnhancer(llm)
    assert enhancer.step_back("any query") is None


def test_step_back_returns_none_on_empty_response():
    enhancer = QueryEnhancer(_llm(""))
    assert enhancer.step_back("any query") is None


def test_query_bundle_includes_step_back_in_all_queries():
    b = QueryBundle(
        original="How does FAISS handle GPU indexing?",
        sub_queries=["FAISS GPU index", "FAISS IVF flat GPU"],
        hyde_text="FAISS supports GPU indexes via the faiss-gpu package.",
        step_back_query="What are approximate nearest-neighbour search algorithms?",
    )
    queries = b.all_queries()
    assert "What are approximate nearest-neighbour search algorithms?" in queries
    assert (
        queries.count("What are approximate nearest-neighbour search algorithms?") == 1
    )


def test_enhance_includes_step_back():
    llm = MagicMock()
    llm.complete.side_effect = [
        "FAISS GPU index\nFAISS IVF GPU",  # decompose
        "FAISS is a library by Facebook.",  # hyde
        "What are ANN search algorithms?",  # step_back
    ]
    enhancer = QueryEnhancer(llm)
    bundle = enhancer.enhance("How does FAISS handle GPU indexing?")
    assert bundle.step_back_query == "What are ANN search algorithms?"
    assert "What are ANN search algorithms?" in bundle.all_queries()


# ---------------------------------------------------------------------------
# rewrite
# ---------------------------------------------------------------------------


def test_rewrite_returns_cleaned_query():
    llm = _llm("What is FAISS?")
    enhancer = QueryEnhancer(llm)
    result = enhancer.rewrite("uhh wht is  faiss???")
    assert result == "What is FAISS?"


def test_rewrite_falls_back_to_none_without_llm():
    assert QueryEnhancer(None).rewrite("anything") is None


# ---------------------------------------------------------------------------
# enhance_async
# ---------------------------------------------------------------------------


async def test_enhance_async_matches_enhance():
    """The async path returns exactly what the sync path returns.

    Each strategy's stubbed response must be distinct (branched on the
    prompt text), or a swap of hyde_text/step_back_query in enhance_async's
    gather unpacking would go undetected: with a single constant response,
    hyde() and step_back() -- both plain `raw or None` -- would collapse to
    the same value and the comparison would still pass.
    """
    from src.context.query_enhancer import QueryEnhancer

    class _LLM:
        def complete(self, messages, **kw):
            prompt = messages[0].content
            if "sub-questions" in prompt:
                return "sub one\nsub two"
            if "ideal answer" in prompt:
                return "hyde answer text"
            if "broader background question" in prompt:
                return "step back question text"
            raise AssertionError(f"unrecognized prompt: {prompt!r}")

    enhancer = QueryEnhancer(_LLM())
    assert await enhancer.enhance_async("q") == enhancer.enhance("q")


async def test_enhance_async_runs_strategies_concurrently():
    """decompose, hyde and step_back overlap instead of queueing.

    Guards the parallelism: run one after another they cost three LLM round
    trips of latency where they need only cost one.
    """
    import time

    from src.context.query_enhancer import QueryEnhancer

    in_flight = 0
    peak = 0

    class _SlowLLM:
        def complete(self, messages, **kw):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            time.sleep(0.05)
            in_flight -= 1
            return "sub one"

    await QueryEnhancer(_SlowLLM()).enhance_async("q")
    assert peak > 1, f"enhancement strategies were serialized (peak {peak})"
