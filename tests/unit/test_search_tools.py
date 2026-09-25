"""Unit tests for function-calling search tools."""

from __future__ import annotations

import asyncio

from src.internal.tools import ToolEffect
from src.internal.tools.search import (
    MultiQueryWebSearchTool,
    SearchPage,
    build_search_tool,
    fetch_pages_concurrently,
    format_search_pages,
    google_custom_search,
    search_for_detail,
    search_for_list,
    search_for_tool_string,
    serper_dev_search,
    serpapi_search,
    _normalize_queries_input,
    _sanitize_query,
)


class _FakeResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload or {}
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, *, payload=None, text="", calls=None, timeout=None):
        del timeout
        self._payload = payload or {}
        self._text = text
        self._calls = calls if calls is not None else []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True
        return False

    def get(self, url, **kwargs):
        self._calls.append((url, kwargs))
        return _FakeResponse(payload=self._payload, text=self._text)


def _make_fake_aiohttp(payload):
    import types
    import aiohttp as real_aiohttp

    class _FakeResp:
        def __init__(self):
            self.headers = {"content-type": "application/json"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        async def json(self):
            return payload

        async def text(self):
            return ""

    class _FakeSess:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, url, **kw):
            return _FakeResp()

        def post(self, url, **kw):
            return _FakeResp()

    return types.SimpleNamespace(
        ClientSession=_FakeSess,
        ClientTimeout=real_aiohttp.ClientTimeout,
    )


def _make_fake_aiohttp_error():
    import types
    import aiohttp as real_aiohttp

    class _FakeErrResp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def raise_for_status(self):
            raise Exception("HTTP error")

        async def json(self):
            raise Exception("HTTP error")

        async def text(self):
            return ""

    class _FakeSess:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, url, **kw):
            return _FakeErrResp()

        def post(self, url, **kw):
            return _FakeErrResp()

    return types.SimpleNamespace(
        ClientSession=_FakeSess,
        ClientTimeout=real_aiohttp.ClientTimeout,
    )


def test_format_search_pages_handles_errors_and_empty_results():
    assert format_search_pages([]) == "No results found."
    assert (
        format_search_pages([SearchPage(title="T", summary="S", url="https://e.test")])
        == "Title: T\nSummary: S\nURL: https://e.test"
    )
    assert format_search_pages([SearchPage(error="boom")]) == "Error: boom"


def test_google_custom_search_maps_results_and_pagination(monkeypatch):
    calls = []

    def _session_factory(*, timeout):
        return _FakeSession(
            timeout=timeout,
            calls=calls,
            payload={
                "items": [
                    {"title": "One", "snippet": "Summary", "link": "https://one.test"}
                ]
            },
        )

    monkeypatch.setattr(
        "src.internal.tools.search.aiohttp.ClientSession",
        _session_factory,
    )

    pages = asyncio.run(
        google_custom_search(
            "dense retrieval",
            page=2,
            page_size=3,
            api_key="key",
            cse_id="cx",
        )
    )

    assert pages == [SearchPage(title="One", summary="Summary", url="https://one.test")]
    assert calls[0][1]["params"]["start"] == 4
    assert calls[0][1]["params"]["num"] == 3


def test_serpapi_search_accepts_serp_api_key_env_alias(monkeypatch):
    calls = []
    monkeypatch.setenv("SERP_API_KEY", "serp-key")

    def _session_factory(*, timeout):
        return _FakeSession(
            timeout=timeout,
            calls=calls,
            payload={
                "answer_box": {"title": "Answer", "answer": "42", "link": "https://a"},
                "organic_results": [
                    {"title": "Organic", "snippet": "Body", "link": "https://o"}
                ],
            },
        )

    monkeypatch.setattr(
        "src.internal.tools.search.aiohttp.ClientSession",
        _session_factory,
    )

    pages = asyncio.run(serpapi_search("answer", page_size=2))

    assert pages[0].title == "Answer"
    assert pages[1].title == "Organic"
    assert calls[0][1]["params"]["api_key"] == "serp-key"


def test_search_for_list_and_tool_string_use_retrieval_client(monkeypatch):
    async def _fake_retrieval_search(**kwargs):
        assert kwargs["query"] == "faiss"
        assert kwargs["page_size"] == 2
        return [SearchPage(title="FAISS", summary="Vector search", url="https://faiss")]

    monkeypatch.setattr(
        "src.internal.tools.search.retrieval_search",
        lambda query, **kwargs: _fake_retrieval_search(query=query, **kwargs),
    )

    rows = asyncio.run(search_for_list("faiss", page_size=2))
    text = asyncio.run(search_for_tool_string("faiss", page_size=2))

    assert rows == [
        {"title": "FAISS", "summary": "Vector search", "url": "https://faiss"}
    ]
    assert "Title: FAISS" in text
    assert "Summary: Vector search" in text


def test_build_search_tool_wraps_formatted_search(monkeypatch):
    async def _fake_search_for_tool_string(query, **kwargs):
        assert query == "faiss"
        assert kwargs["page_size"] == 3
        return "formatted"

    monkeypatch.setattr(
        "src.internal.tools.search.search_for_tool_string",
        _fake_search_for_tool_string,
    )

    tool = build_search_tool(page_size=3)
    assert tool.effect is ToolEffect.READ_ONLY
    text, raw, meta = asyncio.run(tool.execute("default", {"query": "faiss"}))

    assert text == "formatted"
    assert raw == "formatted"
    assert meta == {}


def test_search_for_detail_fetches_pages_concurrently(monkeypatch):
    async def _fake_search_tool(*args, **kwargs):
        del args, kwargs
        return [SearchPage(title="T", summary="S", url="https://t")]

    async def _fake_fetch_url(url, **kwargs):
        assert url == "https://t"
        assert kwargs["max_length"] == 20
        return "content"

    monkeypatch.setattr("src.internal.tools.search.search_tool", _fake_search_tool)
    monkeypatch.setattr("src.internal.tools.search.fetch_url", _fake_fetch_url)

    detail = asyncio.run(search_for_detail("query", chunk_size=20))

    assert detail == "Title: T\nURL: https://t\nContent: content"


def test_search_for_detail_keeps_content_aligned_across_errored_pages(monkeypatch):
    """Content is fetched only for healthy pages but must stay paired with them."""

    async def _fake_search_tool(*args, **kwargs):
        del args, kwargs
        return [
            SearchPage(title="A", summary="SA", url="https://a"),
            SearchPage(error="boom"),
            SearchPage(title="B", summary="SB", url="https://b"),
        ]

    async def _fake_fetch_url(url, **kwargs):
        del kwargs
        return f"content-for-{url.rsplit('/', 1)[-1]}"

    monkeypatch.setattr("src.internal.tools.search.search_tool", _fake_search_tool)
    monkeypatch.setattr("src.internal.tools.search.fetch_url", _fake_fetch_url)

    detail = asyncio.run(search_for_detail("query"))

    assert detail == (
        "Title: A\nURL: https://a\nContent: content-for-a"
        "\n\n"
        "Error: boom"
        "\n\n"
        "Title: B\nURL: https://b\nContent: content-for-b"
    )


def test_html_to_text_prefers_article_element():
    """_html_to_text should extract <article> content before falling back to <p> tags."""
    from src.internal.tools.html_text import _html_to_text

    html = """
    <html><body>
      <header><p>Navigation noise</p></header>
      <article><p>Main article paragraph one.</p><p>Paragraph two.</p></article>
      <footer><p>Footer noise</p></footer>
    </body></html>
    """
    text = _html_to_text(html)
    assert "Main article paragraph one." in text
    # Should NOT include noise from header/footer when article has enough content
    assert "Navigation noise" not in text


def test_html_to_text_falls_back_to_p_tags_when_no_article():
    from src.internal.tools.html_text import _html_to_text

    html = (
        "<html><body><div><p>Content here.</p><p>More content.</p></div></body></html>"
    )
    text = _html_to_text(html)
    assert "Content here." in text
    assert "More content." in text


def test_fetch_pages_concurrently_replaces_summary_with_fetched_content(monkeypatch):
    async def _fake_fetch_url(url, *, max_length, timeout_seconds):
        return f"fetched:{url}"

    monkeypatch.setattr("src.internal.tools.search.fetch_url", _fake_fetch_url)

    pages = [
        SearchPage(title="A", summary="short", url="https://a.test"),
        SearchPage(title="B", summary="short", url="https://b.test"),
    ]
    enriched = asyncio.run(fetch_pages_concurrently(pages, max_chars=2000))

    assert enriched[0].summary == "fetched:https://a.test"
    assert enriched[1].summary == "fetched:https://b.test"
    assert enriched[0].title == "A"  # title preserved


def test_fetch_pages_concurrently_skips_error_and_empty_url_pages(monkeypatch):
    async def _fake_fetch_url(url, **kwargs):
        return "fetched"

    monkeypatch.setattr("src.internal.tools.search.fetch_url", _fake_fetch_url)

    pages = [
        SearchPage(error="oops"),
        SearchPage(title="NoURL", summary="s", url=""),
        SearchPage(title="OK", summary="s", url="https://ok.test"),
    ]
    enriched = asyncio.run(fetch_pages_concurrently(pages, max_chars=2000))

    assert enriched[0].error == "oops"  # error page unchanged
    assert enriched[1].summary == "s"  # no-URL page unchanged
    assert enriched[2].summary == "fetched"


def test_fetch_pages_concurrently_keeps_original_on_fetch_error(monkeypatch):
    async def _fake_fetch_url(url, **kwargs):
        return "[fetch error] timeout"

    monkeypatch.setattr("src.internal.tools.search.fetch_url", _fake_fetch_url)

    pages = [SearchPage(title="T", summary="original", url="https://t.test")]
    enriched = asyncio.run(fetch_pages_concurrently(pages, max_chars=2000))

    assert enriched[0].summary == "original"


class TestSanitizeQuery:
    def test_removes_null_bytes(self):
        assert _sanitize_query("hello\x00world") == "hello world"

    def test_removes_control_chars(self):
        assert _sanitize_query("a\x01b\x1fc") == "a b c"

    def test_removes_del_char(self):
        assert _sanitize_query("abc\x7fdef") == "abcdef"

    def test_normalizes_whitespace(self):
        assert _sanitize_query("  foo   bar  ") == "foo bar"

    def test_passthrough_clean_query(self):
        assert _sanitize_query("What is FAISS?") == "What is FAISS?"


class TestNormalizeQueriesInput:
    def test_string_becomes_list(self):
        assert _normalize_queries_input("hello") == ["hello"]

    def test_list_passthrough(self):
        assert _normalize_queries_input(["a", "b"]) == ["a", "b"]

    def test_drops_empty_strings(self):
        assert _normalize_queries_input(["ok", "", "  "]) == ["ok"]

    def test_non_list_non_string_returns_empty(self):
        assert _normalize_queries_input(42) == []
        assert _normalize_queries_input(None) == []

    def test_sanitizes_each_entry(self):
        assert _normalize_queries_input(["hello\x00world"]) == ["hello world"]

    def test_none_items_dropped(self):
        assert _normalize_queries_input(["a", None, "b"]) == ["a", "b"]


class TestSerperDevSearch:
    def test_returns_mapped_results(self, monkeypatch):
        import src.internal.tools.search as mod

        payload = {
            "organic": [
                {
                    "title": "SerperOne",
                    "link": "https://serper.test/1",
                    "snippet": "snip1",
                },
                {
                    "title": "SerperTwo",
                    "link": "https://serper.test/2",
                    "snippet": "snip2",
                },
            ]
        }
        monkeypatch.setattr(mod, "aiohttp", _make_fake_aiohttp(payload))

        pages = asyncio.run(serper_dev_search("test", api_key="key"))
        assert len(pages) == 2
        assert pages[0].title == "SerperOne"
        assert pages[0].url == "https://serper.test/1"
        assert pages[0].summary == "snip1"

    def test_returns_error_on_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)

        pages = asyncio.run(serper_dev_search("test", api_key=None))
        assert len(pages) == 1
        assert pages[0].error is not None

    def test_returns_error_page_on_http_failure(self, monkeypatch):
        import src.internal.tools.search as mod

        monkeypatch.setattr(mod, "aiohttp", _make_fake_aiohttp_error())

        pages = asyncio.run(serper_dev_search("test", api_key="k"))
        assert len(pages) == 1
        assert pages[0].error is not None


class TestMultiQueryWebSearchTool:
    def test_schema_has_queries_field(self):
        async def _noop(q, **kw):
            return []

        tool = MultiQueryWebSearchTool(search_fn=_noop)
        schema = tool.schema
        assert schema.name == "web_search"
        props = schema.parameters["properties"]
        assert "queries" in props
        assert props["queries"]["type"] == "array"

    def test_execute_runs_queries_in_parallel(self):
        seen = []

        async def _fake(query, **kwargs):
            seen.append(query)
            return [SearchPage(title=query, summary="s", url=f"https://{query}.test")]

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        result_str, raw, meta = asyncio.run(
            tool.execute("inst1", {"queries": ["alpha", "beta"]})
        )
        assert "alpha" in result_str
        assert "beta" in result_str
        assert set(seen) == {"alpha", "beta"}

    def test_execute_deduplicates_by_url(self):
        async def _fake(query, **kwargs):
            return [SearchPage(title="Same", summary="s", url="https://same.test")]

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        result_str, raw, _ = asyncio.run(
            tool.execute("inst1", {"queries": ["q1", "q2"]})
        )
        assert result_str.count("https://same.test") == 1

    def test_execute_sanitizes_queries(self):
        seen = []

        async def _fake(query, **kwargs):
            seen.append(query)
            return []

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        asyncio.run(tool.execute("inst1", {"queries": ["hello\x00world", "  ok  "]}))
        assert seen == ["hello world", "ok"]

    def test_execute_returns_empty_json_array_when_empty(self):
        async def _fake(query, **kwargs):
            return []

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        result_str, raw, _ = asyncio.run(
            tool.execute("inst1", {"queries": ["nothing"]})
        )
        assert result_str == "[]"

    def test_execute_accepts_string_queries(self):
        seen = []

        async def _fake(query, **kwargs):
            seen.append(query)
            return []

        tool = MultiQueryWebSearchTool(search_fn=_fake)
        asyncio.run(tool.execute("inst1", {"queries": "single query"}))
        assert seen == ["single query"]


def test_search_page_preserves_score_from_result():
    from src.context.search import SearchResult
    from src.internal.tools.search import SearchPage

    result = SearchResult(contents="FAISS body", score=0.42, title="FAISS", url="u")
    page = SearchPage.from_search_result(result)
    assert page.score == 0.42


def test_documents_from_search_pages_maps_score():
    from src.internal.servers.web.app import _documents_from_search_pages
    from src.internal.tools.search import SearchPage

    pages = [SearchPage(title="FAISS", summary="body", url="u", score=0.42)]
    docs = _documents_from_search_pages(
        pages, source_provider="retrieval", query="FAISS"
    )
    assert docs[0].score == 0.42
