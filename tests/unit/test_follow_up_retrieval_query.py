import asyncio
from unittest.mock import MagicMock

from src.agents.search.agentic_rag import AgenticRAGConfig, AgenticRAGLoop
from src.context import SearchContextBundle
from src.context.query_enhancer import QueryBundle


def _loop(monkeypatch, seen, max_rounds=1):
    async def fake_retrieve(queries, **kwargs):
        seen["retrieved"].extend(queries)
        return [SearchContextBundle(query=q, documents=[]) for q in queries]

    monkeypatch.setattr(
        "src.agents.search.agentic_rag.retrieve_contexts", fake_retrieve
    )
    llm = MagicMock()
    llm.complete.side_effect = lambda *args, **kwargs: "yes"
    loop = AgenticRAGLoop(AgenticRAGConfig(max_rounds=max_rounds), llm=llm)

    # Enhancement normally rewrites the query (decompose/HyDE/step-back), so a
    # stub LLM would replace it; the spy records what enhancement was given and
    # searches for exactly that.
    async def spy_enhance(query):
        seen["enhanced"] = query
        return QueryBundle(original=query, sub_queries=[query])

    monkeypatch.setattr(loop._enhancer, "enhance_async", spy_enhance)
    return loop


def test_agentic_rag_retrieves_on_retrieval_query(monkeypatch):
    seen = {"retrieved": []}
    loop = _loop(monkeypatch, seen)
    asyncio.run(
        loop.run(
            "Does it support GPUs?",
            retrieval_query="What is FAISS?\nDoes it support GPUs?",
        )
    )
    assert seen["enhanced"] == "What is FAISS?\nDoes it support GPUs?"
    assert seen["retrieved"] == ["What is FAISS?\nDoes it support GPUs?"]


def test_agentic_rag_without_retrieval_query_is_unchanged(monkeypatch):
    seen = {"retrieved": []}
    loop = _loop(monkeypatch, seen)
    asyncio.run(loop.run("Does it support GPUs?"))
    assert seen["enhanced"] == "Does it support GPUs?"
    assert seen["retrieved"] == ["Does it support GPUs?"]


def test_answer_with_retrieval_retrieves_on_retrieval_query(monkeypatch):
    from src.context import pipeline

    seen = {}

    async def fake_retrieve_context(question, **kwargs):
        seen["retrieved"] = question
        return SearchContextBundle(query=question, documents=[])

    monkeypatch.setattr(pipeline, "retrieve_context", fake_retrieve_context)
    result = asyncio.run(
        pipeline.answer_with_retrieval(
            "Does it support GPUs?",
            retrieval_query="What is FAISS?\nDoes it support GPUs?",
        )
    )
    assert seen["retrieved"] == "What is FAISS?\nDoes it support GPUs?"
    assert result is not None


def test_answer_with_retrieval_defaults_to_question(monkeypatch):
    from src.context import pipeline

    seen = {}

    async def fake_retrieve_context(question, **kwargs):
        seen["retrieved"] = question
        return SearchContextBundle(query=question, documents=[])

    monkeypatch.setattr(pipeline, "retrieve_context", fake_retrieve_context)
    asyncio.run(pipeline.answer_with_retrieval("Does it support GPUs?"))
    assert seen["retrieved"] == "Does it support GPUs?"


def test_sufficiency_and_gap_queries_use_retrieval_query(monkeypatch):
    # max_rounds=2 so round 1 is not the last and both checks actually run.
    seen = {"retrieved": []}
    loop = _loop(monkeypatch, seen, max_rounds=2)
    asked = {}

    async def spy_sufficient(question, context):
        asked["sufficiency"] = question
        return False, False

    async def spy_followup(question, context):
        asked["gap"] = question
        return ["FAISS GPU support"]

    monkeypatch.setattr(loop, "_is_sufficient", spy_sufficient)
    monkeypatch.setattr(loop, "_generate_followup", spy_followup)
    asyncio.run(
        loop.run(
            "Does it support GPUs?",
            retrieval_query="What is FAISS?\nDoes it support GPUs?",
        )
    )
    assert asked == {
        "sufficiency": "What is FAISS?\nDoes it support GPUs?",
        "gap": "What is FAISS?\nDoes it support GPUs?",
    }
    assert "FAISS GPU support" in seen["retrieved"]
