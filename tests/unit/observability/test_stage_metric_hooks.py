"""The two choke points feed the request's stage metrics: every retrieval
caller goes through ``SearchClient.retrieve`` and every generation through an
LLM backend, so hooking those covers direct search, hybrid, the agent loops and
the tool agent alike."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.context.retrieval.client import SearchClient, SearchClientConfig
from src.internal.cache import serving
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.observability import stage_metrics as sm


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, posts: list) -> None:
        self._posts = posts
        self.closed = False

    def post(self, url, json):
        self._posts.append(json)
        rows = [
            [{"document": {"title": q}}, {"document": {"title": q}}]
            for q in json["queries"]
        ]
        return _FakeResponse({"results": rows})

    async def close(self):
        self.closed = True


@pytest.fixture
def posts(monkeypatch) -> list:
    posts: list = []
    monkeypatch.setattr(
        "src.context.retrieval.client.aiohttp.ClientSession",
        lambda *, timeout: _FakeSession(posts),
    )
    return posts


@pytest.fixture
def request_scope():
    token = sm.start_request()
    yield
    sm.finish_request(token)


def _client() -> SearchClient:
    return SearchClient(SearchClientConfig(url="http://localhost:8001/retrieve"))


def test_search_client_notes_elapsed_and_docs(posts, request_scope):
    asyncio.run(_client().retrieve(["a", "b"]))
    snap = sm.current().snapshot()["retrieval"]
    assert snap["calls"] == 1
    assert snap["docs"] == 4
    assert snap["cache_hits"] == 0
    assert snap["ms"] >= 0.0


def test_search_client_full_cache_hit_counts_as_hit(posts, request_scope):
    serving.configure_serving_cache(60)
    try:
        client = _client()
        asyncio.run(client.retrieve(["a"]))
        asyncio.run(client.retrieve(["a"]))
    finally:
        serving.reset_serving_cache()
    snap = sm.current().snapshot()["retrieval"]
    assert snap["calls"] == 2
    assert snap["cache_hits"] == 1
    assert snap["docs"] == 4


def test_search_client_outside_a_request_is_silent(posts):
    asyncio.run(_client().retrieve(["a"]))
    assert sm.current() is None


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        LLMConfig(model_provider="openai", model_name="gpt-4o-mini", api_key="sk")
    )


def _complete(llm, payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    with patch.object(llm._session, "post", return_value=resp):
        llm.complete([{"role": "user", "content": "hello"}])


_WITH_USAGE = {
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


def test_openai_compatible_complete_outside_answer_scope_is_auxiliary(request_scope):
    # Query transforms, sufficiency checks and intent recognition all go
    # through `complete` too; they must not inflate the generation bucket.
    _complete(_llm(), _WITH_USAGE)
    snap = sm.current().snapshot()
    assert snap["generation"]["calls"] == 0
    assert snap["auxiliary"]["calls"] == 1
    assert snap["auxiliary"]["prompt_tokens"] == 12
    assert snap["auxiliary"]["completion_tokens"] == 3
    assert snap["auxiliary"]["ms"] >= 0.0


def test_openai_compatible_complete_inside_generate_answer_is_the_answer(
    request_scope,
):
    with sm.answer_generation():
        _complete(_llm(), _WITH_USAGE)
    snap = sm.current().snapshot()
    assert snap["generation"] == {
        "calls": 1,
        "ms": snap["generation"]["ms"],
        "prompt_tokens": 12,
        "completion_tokens": 3,
    }
    assert snap["auxiliary"]["calls"] == 0


def test_generate_answer_is_marked_as_the_answer():
    from src.context.pipeline import generate_answer

    # The decorator is what files `generate_answer`'s LLM call as the answer.
    assert getattr(generate_answer, "__wrapped__", None) is not None


def test_openai_compatible_complete_without_usage_still_counts_the_call(
    request_scope,
):
    _complete(_llm(), {"choices": [{"message": {"content": "hi"}}]})
    snap = sm.current().snapshot()["auxiliary"]
    assert snap == {
        "calls": 1,
        "ms": snap["ms"],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def test_local_server_manager_generate_notes_token_lengths(request_scope):
    from src.model.serving import LocalServerManager

    manager = LocalServerManager.__new__(LocalServerManager)
    manager.model_path = "fake"
    manager._tokenizer = MagicMock(decode=lambda ids, **_: "x")
    manager._generate_sync = lambda prompt_ids, params, request_id: [7, 8, 9]
    asyncio.run(manager.generate("r1", [1, 2, 3, 4], {}))
    snap = sm.current().snapshot()
    # An agent loop's turn is the answer the user reads, never auxiliary.
    assert snap["generation"]["calls"] == 1
    assert snap["generation"]["prompt_tokens"] == 4
    assert snap["generation"]["completion_tokens"] == 3
    assert snap["auxiliary"]["calls"] == 0
