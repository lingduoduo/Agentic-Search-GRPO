"""Adapters that turn failures into typed results keep timeout identity.

The registry counts an attempt as ``timeout`` from ``ToolFailure.is_timeout``.
Adapters that catch the exception themselves (public-data ``guarded``, the
search providers, the corpus and web search tools) must carry that identity
from the exception type -- never from message text -- or every timeout behind
them is counted as a generic error.
"""

from __future__ import annotations

import asyncio

import pytest

from src.internal.tools import search as search_mod
from src.internal.tools import routing_tools
from src.internal.tools.base import FailureCategory
from src.internal.tools.public_data import _http
from src.internal.tools.public_data._http import get_json, guarded
from src.internal.tools.registry import ToolRegistry
from src.internal.tools.search import MultiQueryWebSearchTool, SearchPage


# --- public data ------------------------------------------------------------


def _failing_session(exc: BaseException):
    class _Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def request(self, *args, **kwargs):
            raise exc

    class _Aiohttp:
        ClientSession = _Session

        @staticmethod
        def ClientTimeout(total):  # noqa: N802 - mirrors aiohttp
            return total

    return _Aiohttp


async def _no_sleep(_seconds):
    return None


@pytest.mark.parametrize(
    ("exc", "timed_out"),
    [(asyncio.TimeoutError(), True), (OSError("reset"), False)],
)
def test_public_data_transport_failure_keeps_timeout_identity(
    monkeypatch, exc, timed_out
):
    monkeypatch.setattr(_http, "aiohttp", _failing_session(exc))
    monkeypatch.setattr(_http.asyncio, "sleep", _no_sleep)

    @guarded
    async def tool():
        return await get_json("https://example.org/x")

    failure = asyncio.run(tool()).failure
    assert failure.category is FailureCategory.TRANSIENT  # recovery unchanged
    assert failure.is_timeout is timed_out


def test_guarded_marks_a_raised_timeout_without_changing_its_category():
    @guarded
    async def tool():
        raise TimeoutError()

    failure = asyncio.run(tool()).failure
    assert failure.category is FailureCategory.UNKNOWN
    assert failure.is_timeout is True


# --- search providers -------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "timed_out"),
    [(asyncio.TimeoutError(), True), (ValueError("bad json"), False)],
)
def test_serpapi_error_page_keeps_timeout_identity(monkeypatch, exc, timed_out):
    async def fake_get_json(*args, **kwargs):
        raise exc

    monkeypatch.setattr(search_mod, "_get_json", fake_get_json)
    pages = asyncio.run(search_mod.serpapi_search("q", api_key="k"))
    assert pages[0].timed_out is timed_out


def test_error_page_text_is_unchanged_by_timeout_tracking(monkeypatch):
    """Metrics must not change recovery (spec). ``str(asyncio.TimeoutError())``
    is empty, and today an empty error page reads as an empty success -- which
    web_search's recovery depends on. Changing that is a separate decision."""

    async def fake_get_json(*args, **kwargs):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(search_mod, "_get_json", fake_get_json)
    pages = asyncio.run(search_mod.serpapi_search("q", api_key="k"))
    assert pages[0].error == ""


@pytest.mark.parametrize(
    ("cause", "timed_out"),
    [(asyncio.TimeoutError(), True), (ConnectionError("refused"), False)],
)
def test_retrieval_search_reads_the_exhausted_retry_cause(
    monkeypatch, cause, timed_out
):
    """SearchClient raises RuntimeError ``from`` its last attempt's error."""

    class _Client:
        def __init__(self, config):
            pass

        async def retrieve_one(self, *args, **kwargs):
            raise RuntimeError("failed after 3 retries") from cause

        async def aclose(self):
            return None

    monkeypatch.setattr(search_mod, "SearchClient", _Client)
    pages = asyncio.run(
        search_mod.retrieval_search("q", search_url="http://r/retrieve")
    )
    assert pages[0].timed_out is timed_out


# --- aggregating tools ------------------------------------------------------


async def _invoke(tool, arguments):
    registry = ToolRegistry()
    registry.register(tool)
    return await registry.invoke_detailed(tool.name, arguments)


@pytest.mark.parametrize(
    ("pages", "timed_out"),
    [
        ([SearchPage(error="a", timed_out=True)], True),
        ([SearchPage(error="a", timed_out=True), SearchPage(error="b")], True),
        ([SearchPage(error="b")], False),
    ],
)
async def test_corpus_search_tool_is_timeout_when_any_error_timed_out(
    monkeypatch, pages, timed_out
):
    async def fake_search_tool(*args, **kwargs):
        return pages

    monkeypatch.setattr(routing_tools, "search_tool", fake_search_tool)
    tool = routing_tools.build_search_routing_tool(
        search_url="http://r/retrieve", top_k=3
    )
    result = await _invoke(tool, {"query": "q"})
    assert result.failure.category is FailureCategory.TRANSIENT
    assert result.failure.is_timeout is timed_out


@pytest.mark.parametrize(
    ("pages", "timed_out"),
    [
        ([SearchPage(error="a", timed_out=True)], True),
        ([SearchPage(error="b")], False),
    ],
)
async def test_web_search_tool_is_timeout_when_any_error_timed_out(pages, timed_out):
    async def fake(query, *, provider, search_url, page_size, timeout_seconds):
        return pages

    result = await _invoke(MultiQueryWebSearchTool(search_fn=fake), {"queries": ["q"]})
    assert result.failure.category is FailureCategory.UNKNOWN
    assert result.failure.is_timeout is timed_out


async def test_default_cascade_timeout_is_a_timeout_despite_the_advisory_page():
    """No browser URL (the default) adds a non-exception advisory error page;
    an explicit timeout still takes precedence (spec: tool timeout rate)."""

    async def serpapi_timed_out(query, **kwargs):
        return [SearchPage(error="Timeout on reading data", timed_out=True)]

    cascade = search_mod.make_web_cascade_search(
        browser_search_url=None, serpapi_fn=serpapi_timed_out
    )
    result = await _invoke(
        MultiQueryWebSearchTool(search_fn=cascade), {"queries": ["q"]}
    )
    assert result.failure.is_timeout is True
