"""Each serving dependency skips its call once its breaker is open."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import aiohttp
import pytest
from yarl import URL

from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies
from src.internal.resilience.circuit_breaker import get_breaker


@contextmanager
def threshold(n: int):
    overrides = {"circuit_breaker": {"failure_threshold": n}}
    with use_timeout_policies(load_timeout_policies({}, overrides=overrides)):
        yield


def _aiohttp_status_error(status: int) -> aiohttp.ClientResponseError:
    url = URL("http://dep.test/x")
    info = aiohttp.RequestInfo(url=url, method="GET", headers={}, real_url=url)
    return aiohttp.ClientResponseError(info, (), status=status, message="x")


# --- serpapi -----------------------------------------------------------------


def _serp(monkeypatch, outcome):
    """Fake _get_json: raise ``outcome`` if it is an exception, else return it."""
    from src.internal.tools import search

    calls = []

    async def fake_get_json(url, **kw):
        calls.append(url)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(search, "_get_json", fake_get_json)
    monkeypatch.setenv("SERPAPI_API_KEY", "k")
    return search, calls


def test_serpapi_open_breaker_skips_the_call(monkeypatch):
    search, calls = _serp(monkeypatch, asyncio.TimeoutError())
    with threshold(2):
        for _ in range(2):
            asyncio.run(search.serpapi_search("q"))
        assert len(calls) == 2
        pages = asyncio.run(search.serpapi_search("q"))
    assert len(calls) == 2  # no outbound call
    assert [p.error for p in pages] == [
        "SerpAPI is temporarily skipped after repeated failures (circuit open)."
    ]


@pytest.mark.parametrize("status", [429, 503])
def test_serpapi_429_and_5xx_count(monkeypatch, status):
    search, _ = _serp(monkeypatch, _aiohttp_status_error(status))
    with threshold(2):
        for _ in range(2):
            asyncio.run(search.serpapi_search("q"))
    assert get_breaker("serpapi").snapshot().state == "open"


def test_serpapi_4xx_does_not_count(monkeypatch):
    search, calls = _serp(monkeypatch, _aiohttp_status_error(401))
    with threshold(2):
        for _ in range(3):
            asyncio.run(search.serpapi_search("q"))
    assert len(calls) == 3
    snap = get_breaker("serpapi").snapshot()
    assert (snap.state, snap.consecutive_failures) == ("closed", 0)


def test_serpapi_success_resets(monkeypatch):
    search, _ = _serp(monkeypatch, {"organic_results": []})
    b = get_breaker("serpapi")
    b.record_failure()
    assert asyncio.run(search.serpapi_search("q")) == []
    assert b.snapshot().consecutive_failures == 0


def test_serpapi_missing_key_never_touches_the_breaker(monkeypatch):
    from src.internal.resilience.circuit_breaker import breaker_snapshots
    from src.internal.tools import search

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    pages = asyncio.run(search.serpapi_search("q"))
    assert "required" in pages[0].error
    assert breaker_snapshots() == []


# --- browser_search ----------------------------------------------------------


def _cascade(browser_outcome):
    from src.internal.tools.search import SearchPage, make_web_cascade_search

    calls = []

    async def fake_serp(query, **kw):
        return [SearchPage(error="serp down")]

    async def fake_browser(query, **kw):
        calls.append(query)
        if isinstance(browser_outcome, BaseException):
            raise browser_outcome
        return browser_outcome

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    return fn, calls


def test_browser_open_breaker_skips_the_call():
    fn, calls = _cascade(RuntimeError("connection refused"))
    with threshold(2):
        for _ in range(2):
            asyncio.run(fn("q"))
        pages = asyncio.run(fn("q"))
    assert len(calls) == 2
    errors = [p.error for p in pages]
    assert "serp down" in errors
    assert (
        "Browser search is temporarily skipped after repeated failures (circuit open)."
        in errors
    )


def test_browser_all_error_pages_count_as_failure():
    from src.internal.tools.search import SearchPage

    fn, _ = _cascade([SearchPage(error="503")])
    with threshold(2):
        for _ in range(2):
            asyncio.run(fn("q"))
    assert get_breaker("browser_search").snapshot().state == "open"


def test_browser_empty_result_is_a_success():
    fn, calls = _cascade([])
    get_breaker("browser_search").record_failure()
    with threshold(2):
        for _ in range(3):
            asyncio.run(fn("q"))
    assert len(calls) == 3
    snap = get_breaker("browser_search").snapshot()
    assert (snap.state, snap.consecutive_failures) == ("closed", 0)


# --- rerank ------------------------------------------------------------------


def _rerank_stage(monkeypatch, outcome):
    """Patch httpx.AsyncClient: post raises ``outcome`` or returns it."""
    import httpx

    from src.internal.search import stages

    calls = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *, json, timeout):
            calls.append(url)
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, int):
                return httpx.Response(outcome, request=httpx.Request("POST", url))
            return httpx.Response(200, json=outcome, request=httpx.Request("POST", url))

    monkeypatch.setattr(stages.httpx, "AsyncClient", _Client)
    return stages.RerankHTTPRankingStage("http://r"), calls


def _candidates():
    from src.context.search import SearchResult
    from src.internal.search.models import CandidateSet

    return CandidateSet(
        query="q",
        candidates=[SearchResult(contents="one", title="One", score=0.2)],
        provider="retrieval",
    )


def test_rerank_open_breaker_raises_without_calling(monkeypatch):
    import httpx

    from src.internal.resilience.circuit_breaker import CircuitOpenError

    stage, calls = _rerank_stage(monkeypatch, httpx.ConnectError("down"))
    with threshold(2):
        for _ in range(2):
            with pytest.raises(httpx.ConnectError):
                asyncio.run(stage.rank("q", _candidates(), 1))
        with pytest.raises(CircuitOpenError):
            asyncio.run(stage.rank("q", _candidates(), 1))
    assert len(calls) == 2


def test_rerank_5xx_counts_and_4xx_does_not(monkeypatch):
    import httpx

    stage, _ = _rerank_stage(monkeypatch, 422)
    with threshold(2):
        for _ in range(3):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(stage.rank("q", _candidates(), 1))
        assert get_breaker("rerank").snapshot().state == "closed"
        stage, _ = _rerank_stage(monkeypatch, 503)
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(stage.rank("q", _candidates(), 1))
    assert get_breaker("rerank").snapshot().state == "open"


def test_rerank_cache_hit_never_touches_the_breaker(monkeypatch):
    from src.internal.resilience.circuit_breaker import breaker_snapshots
    from src.internal.search import stages

    class _Cache:
        def get(self, key):
            return [{"document": {"_idx": "0"}, "score": 0.9}]

        def set(self, key, value):
            raise AssertionError("a hit is not re-cached")

    monkeypatch.setattr(stages, "serving_cache", lambda: _Cache())
    stage, calls = _rerank_stage(monkeypatch, RuntimeError("must not be called"))
    result = asyncio.run(stage.rank("q", _candidates(), 1))
    assert [d.title for d in result.evidence] == ["One"]
    assert calls == []
    assert breaker_snapshots() == []


def test_default_ranking_reports_circuit_open():
    from src.internal.resilience.circuit_breaker import CircuitOpenError
    from src.internal.search.ranking import DefaultRankingStage

    class _OpenReranker:
        async def rank(self, query, candidates, top_k):
            raise CircuitOpenError("rerank", 12.0)

    result = asyncio.run(
        DefaultRankingStage(_OpenReranker()).rank("q", _candidates(), 1)
    )
    assert result.metadata["rerank_status"] == "circuit_open"
    assert result.metadata["degraded"] is True
    assert [d.title for d in result.evidence] == ["One"]


# --- remote_llm --------------------------------------------------------------


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "prompt"

    def encode(self, text):
        return [1] * len(text)


class _Resp:
    def __init__(self, *, status=200, lines=()):
        self.status = status
        self.content = self._iter(lines)

    @staticmethod
    async def _iter(lines):
        for line in lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise _aiohttp_status_error(self.status)

    async def json(self):
        return {"choices": [{"text": "ok"}], "usage": {}}


def _llm(monkeypatch, outcome, base_url="http://llm"):
    """A remote manager whose session.post raises ``outcome`` or returns it."""
    from src.model.serving import OpenAIServerManager

    calls = []

    class _Session:
        closed = False

        def post(self, url, json):
            calls.append(url)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    m = OpenAIServerManager(tokenizer=_Tokenizer(), base_url=base_url, model="m")
    monkeypatch.setattr(m, "_get_session", lambda: _Session())
    return m, calls


async def _noop(text):
    return None


def _call(m, stream):
    if stream:
        return asyncio.run(m.generate_stream("r", [1], {}, _noop))
    return asyncio.run(m.generate("r", [1], {}))


@pytest.mark.parametrize("stream", [False, True])
def test_remote_llm_open_breaker_raises_without_calling(monkeypatch, stream):
    m, calls = _llm(monkeypatch, asyncio.TimeoutError())
    with threshold(2):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Cannot connect"):
                _call(m, stream)
        with pytest.raises(RuntimeError) as info:
            _call(m, stream)
    assert len(calls) == 2
    assert str(info.value) == (
        "Inference server at http://llm is temporarily skipped after "
        "repeated failures (circuit open)."
    )


@pytest.mark.parametrize("stream", [False, True])
def test_remote_llm_5xx_and_429_count(monkeypatch, stream):
    for status in (503, 429):
        from src.internal.resilience.circuit_breaker import reset_breakers

        reset_breakers()
        m, _ = _llm(monkeypatch, _Resp(status=status))
        with threshold(2):
            for _ in range(2):
                with pytest.raises(aiohttp.ClientResponseError):
                    _call(m, stream)
        assert get_breaker("remote_llm:http://llm").snapshot().state == "open"


@pytest.mark.parametrize("stream", [False, True])
def test_remote_llm_4xx_is_reraised_and_not_a_failure(monkeypatch, stream):
    get_breaker("remote_llm:http://llm").record_failure()
    m, calls = _llm(monkeypatch, _Resp(status=400))
    with threshold(2):
        for _ in range(3):
            with pytest.raises(aiohttp.ClientResponseError) as info:
                _call(m, stream)
            assert info.value.status == 400
    assert len(calls) == 3
    snap = get_breaker("remote_llm:http://llm").snapshot()
    assert (snap.state, snap.consecutive_failures) == ("closed", 0)


def test_remote_llm_other_transport_error_counts(monkeypatch):
    m, _ = _llm(monkeypatch, aiohttp.ServerDisconnectedError())
    with threshold(2):
        for _ in range(2):
            with pytest.raises(aiohttp.ServerDisconnectedError):
                _call(m, False)
    assert get_breaker("remote_llm:http://llm").snapshot().state == "open"


def test_remote_llm_success_resets(monkeypatch):
    get_breaker("remote_llm:http://llm").record_failure()
    m, _ = _llm(monkeypatch, _Resp())
    assert _call(m, False) == [1, 1]
    stream_m, _ = _llm(
        monkeypatch,
        _Resp(lines=[b'data: {"choices": [{"text": "hi"}]}', b"data: [DONE]"]),
    )
    get_breaker("remote_llm:http://llm").record_failure()
    assert _call(stream_m, True) == [1, 1]
    assert get_breaker("remote_llm:http://llm").snapshot().consecutive_failures == 0


def test_stream_callback_error_is_not_a_failure(monkeypatch):
    # A client that disconnects mid-stream says nothing about the LLM server.
    m, _ = _llm(monkeypatch, _Resp(lines=[b'data: {"choices": [{"text": "hi"}]}']))

    async def on_token(text):
        raise ConnectionResetError("client went away")

    with threshold(1):
        with pytest.raises(ConnectionResetError):
            asyncio.run(m.generate_stream("r", [1], {}, on_token))
    assert get_breaker("remote_llm:http://llm").snapshot().consecutive_failures == 0


def test_remote_llm_breaker_is_per_server(monkeypatch):
    down, _ = _llm(monkeypatch, asyncio.TimeoutError())
    with threshold(1):
        with pytest.raises(RuntimeError, match="Cannot connect"):
            _call(down, stream=False)
        up, calls = _llm(monkeypatch, _Resp(status=200), base_url="http://other")
        assert _call(up, stream=False)
    assert calls == ["http://other/v1/completions"]
