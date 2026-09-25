"""Each migrated site reads its value from the timeout policies."""

import asyncio
from contextlib import contextmanager

import pytest

from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies


@contextmanager
def overridden(overrides: dict):
    with use_timeout_policies(load_timeout_policies({}, overrides=overrides)) as p:
        yield p


def test_public_data_attempts_follow_the_policy(monkeypatch):
    from src.internal.tools.public_data import _http
    from tests.unit.test_public_data_http import _SequencedSession

    real_sleep = asyncio.sleep
    shared = _SequencedSession([503, 503, 503, 503, 503])
    _SequencedSession.calls = []

    class _Aiohttp:
        @staticmethod
        def ClientTimeout(total=None):
            return total

        @staticmethod
        def ClientSession(timeout=None):
            return shared

    monkeypatch.setattr(_http, "aiohttp", _Aiohttp)
    monkeypatch.setattr(_http.asyncio, "sleep", lambda s: real_sleep(0))
    with overridden({"tools": {"public_data": {"max_attempts": 5}}}):
        with pytest.raises(_http.PublicDataError):
            asyncio.run(_http.get_json("https://x.test"))
    assert (
        len(_SequencedSession.calls) == 5
    )  # 5 attempts, 2 backoff steps: last step reused


def test_public_data_timeout_follows_the_policy_and_kwarg_wins(monkeypatch):
    from src.internal.tools.public_data import _http

    seen = []

    async def fake_fetch(method, url, *, timeout_seconds, **kw):
        seen.append(timeout_seconds)
        return {}

    monkeypatch.setattr(_http, "_fetch", fake_fetch)
    with overridden({"tools": {"public_data": {"timeout_seconds": 3}}}):
        asyncio.run(_http.get_json("https://x.test"))
        asyncio.run(_http.get_json("https://x.test", timeout_seconds=9))
    assert seen == [3.0, 9]


def test_search_tool_uses_router_policy_and_kwarg_wins(monkeypatch):
    from src.internal.tools import search

    seen = []

    async def fake_retrieval(query, **kw):
        seen.append((kw["timeout_seconds"], kw["max_retries"]))
        return []

    monkeypatch.setattr(search, "retrieval_search", fake_retrieval)
    with overridden(
        {"tools": {"search_router": {"timeout_seconds": 4, "max_retries": 2}}}
    ):
        asyncio.run(search.search_tool("q"))
        asyncio.run(search.search_tool("q", timeout_seconds=6, max_retries=1))
    assert seen == [(4.0, 2), (6, 1)]


def test_retrieval_search_uses_client_policy(monkeypatch):
    from src.internal.tools import search

    configs = []

    class FakeClient:
        def __init__(self, config):
            configs.append(config)

        async def retrieve_one(self, *a, **k):
            return []

        async def aclose(self):
            pass

    monkeypatch.setattr(search, "SearchClient", FakeClient)
    with overridden(
        {"retrieval": {"client": {"timeout_seconds": 7, "max_retries": 2}}}
    ):
        asyncio.run(search.retrieval_search("q", search_url="http://x"))
    assert (configs[0].timeout_seconds, configs[0].max_retries) == (7.0, 2)


def test_google_search_uses_web_search_policy(monkeypatch):
    from src.internal.tools import search

    seen = []

    async def fake_get_json(url, *, params, timeout_seconds, **kw):
        seen.append(timeout_seconds)
        return {"items": []}

    monkeypatch.setattr(search, "_get_json", fake_get_json)
    with overridden({"tools": {"web_search": {"google_timeout_seconds": 3}}}):
        asyncio.run(search.google_custom_search("q", api_key="k", cse_id="c"))
    assert seen == [3.0]


def test_corpus_search_reports_router_attempts(monkeypatch):
    from src.internal.tools import routing_tools
    from src.internal.tools.search import SearchPage

    async def failing(*a, **k):
        return [SearchPage(error="down")]

    monkeypatch.setattr(routing_tools, "search_tool", failing)
    tool = routing_tools.build_search_routing_tool(search_url="http://x", top_k=3)
    with overridden({"tools": {"search_router": {"max_retries": 5}}}):
        text, _raw, meta = asyncio.run(tool.execute("i", {"query": "q"}))
    assert meta["failure"].provider_attempts == 5


def _openapi_schema() -> str:
    import json

    return json.dumps(
        {
            "openapi": "3.0.0",
            "info": {"title": "Demo", "description": "Demo API"},
            "servers": [{"url": "https://api.example.test"}],
            "paths": {
                "/ping": {
                    "get": {
                        "operationId": "ping",
                        "description": "Ping.",
                        "parameters": [],
                    }
                }
            },
        }
    )


class _FakeOpenApiResponse:
    headers = {"content-type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return {"ok": True}

    async def text(self):
        return "ok"


class _FakeOpenApiSession:
    def __init__(self, *, timeout):
        del timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, **kwargs):
        del kwargs
        return _FakeOpenApiResponse()


def test_openapi_tool_timeout_follows_policy(monkeypatch):
    from src.internal.tools import api as api_tools

    seen = []
    real = api_tools.aiohttp.ClientTimeout
    monkeypatch.setattr(
        api_tools.aiohttp,
        "ClientTimeout",
        lambda total: seen.append(total) or real(total=total),
    )
    monkeypatch.setattr(
        api_tools.aiohttp,
        "ClientSession",
        lambda *, timeout: _FakeOpenApiSession(timeout=timeout),
    )

    registry = api_tools.ApiToolRegistry()
    provider = registry.create_provider(name="demo", openapi_schema=_openapi_schema())
    tool = next(t for t in registry.build_tools(provider.id) if t.name == "ping")

    with overridden({"tools": {"openapi": {"timeout_seconds": 2}}}):
        asyncio.run(tool.execute("default", {}))

    assert seen == [2.0]


def test_mcp_client_passes_policy_timeouts(monkeypatch):
    pytest.importorskip("mcp")
    from contextlib import asynccontextmanager

    from src.internal.tools import mcp_client
    import mcp.client.streamable_http as streamable_http

    seen = []

    @asynccontextmanager
    async def fake_streamablehttp_client(url, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop after recording")
        yield  # pragma: no cover

    monkeypatch.setattr(
        streamable_http, "streamablehttp_client", fake_streamablehttp_client
    )

    spec = mcp_client.McpServerSpec(name="local", url="http://x/")
    with overridden(
        {"tools": {"mcp": {"timeout_seconds": 3, "sse_read_timeout_seconds": 9}}}
    ):
        with pytest.raises(RuntimeError, match="stop after recording"):
            asyncio.run(_enter_and_exit(mcp_client._connect(spec)))

    assert seen[0]["timeout"] == 3.0
    assert seen[0]["sse_read_timeout"] == 9.0


async def _enter_and_exit(cm):
    async with cm as value:
        return value


def test_search_client_config_defaults_follow_policy():
    from src.context.retrieval.client import SearchClientConfig

    with overridden(
        {"retrieval": {"client": {"timeout_seconds": 7, "max_retries": 2}}}
    ):
        cfg = SearchClientConfig(url="http://x")
    assert (cfg.timeout_seconds, cfg.max_retries) == (7.0, 2)
    assert SearchClientConfig(url="http://x", timeout_seconds=1).timeout_seconds == 1


def test_search_runner_defaults_follow_policy(monkeypatch):
    from src.context.enums import SearchType
    from src.context.models import SearchRequest
    from src.context.retrieval import search_runner

    configs = []

    class FakeSearchClient:
        def __init__(self, config):
            configs.append(config)

        async def retrieve_one(self, *a, **k):
            return []

        async def retrieve(self, queries, *a, **k):
            return [[] for _ in queries]

        async def aclose(self):
            pass

    monkeypatch.setattr(search_runner, "SearchClient", FakeSearchClient)
    request = SearchRequest(query="q", provider=SearchType.RETRIEVAL, top_k=3)
    with overridden(
        {"retrieval": {"search_runner": {"timeout_seconds": 4, "max_retries": 2}}}
    ):
        asyncio.run(search_runner.run_search(request))
        asyncio.run(search_runner.build_search_context(request))
        asyncio.run(search_runner.build_search_contexts(["q"]))
    assert configs and all(
        (c.timeout_seconds, c.max_retries) == (4.0, 2) for c in configs
    )


def test_agent_search_configs_follow_client_policy():
    from src.agents.generation.single_turn import SingleTurnAgentLoopConfig
    from src.agents.search.search import SearchAgentLoopConfig

    with overridden(
        {"retrieval": {"client": {"timeout_seconds": 7, "max_retries": 2}}}
    ):
        for cfg in (SearchAgentLoopConfig(), SingleTurnAgentLoopConfig()):
            assert (cfg.search_timeout_seconds, cfg.search_max_retries) == (7.0, 2)


def test_rerank_stage_timeout_follows_policy():
    from src.internal.search.stages import RerankHTTPRankingStage

    with overridden({"rerank": {"timeout_seconds": 2}}):
        stage = RerankHTTPRankingStage("http://r", document_contents=lambda c: "")
    assert stage._timeout == 2.0


def test_web_hybrid_uses_policy(monkeypatch):
    from src.internal.servers.web import app

    seen = []

    async def fake_search_tool(query, *, provider, **kwargs):
        seen.append((kwargs.get("timeout_seconds"), kwargs.get("max_retries")))
        return []

    monkeypatch.setattr(app, "search_tool", fake_search_tool)

    async def _run():
        return await app._run_hybrid_search(
            "q",
            llm=None,
            search_url="http://retrieval",
            browser_search_url=None,
            rerank_url=None,
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )

    with overridden(
        {
            "retrieval": {
                "web_hybrid": {"provider_timeout_seconds": 3, "provider_max_retries": 2}
            }
        }
    ):
        asyncio.run(_run())
    assert seen and all(s == (3.0, 2) for s in seen)


def _make_default_llm(multi_llm):
    return multi_llm.LitellmLLM(
        api_key="test-key",
        model_provider="openai",
        model_name="test-model",
        max_input_tokens=1024,
    )


def test_llm_socket_read_timeout_follows_policy():
    from src.internal.llm import multi_llm

    with overridden({"llm": {"socket_read_timeout_seconds": 33}}):
        llm = _make_default_llm(multi_llm)
    assert llm._timeout == 33.0


def test_remote_server_manager_timeout_follows_policy():
    from src.model.serving import OpenAIServerManager

    with overridden({"llm": {"remote_total_timeout_seconds": 44}}):
        m = OpenAIServerManager(tokenizer=None, base_url="http://x", model="m")
    assert m.timeout_seconds == 44.0


def test_local_generation_timeout_policy_and_explicit_none():
    from src.model.serving import LocalServerManager

    with overridden(
        {"llm": {"local_generation_timeout_seconds": 50, "local_heartbeat_seconds": 2}}
    ):
        m = LocalServerManager(model_path="unused/model", device="cpu")
    assert (m.generation_timeout_seconds, m.generation_heartbeat_seconds) == (50.0, 2.0)

    with overridden({"llm": {"local_generation_timeout_seconds": 50}}):
        m_none = LocalServerManager(
            model_path="unused/model", device="cpu", generation_timeout_seconds=None
        )
    assert m_none.generation_timeout_seconds is None


def test_sufficiency_and_grounded_retries_follow_policy():
    from src.agents.search.agentic_rag import AgenticRAGConfig
    from src.context.models import GroundedGenerationConfig

    with overridden(
        {"llm": {"sufficiency_timeout_seconds": 2, "grounded_max_retries": 0}}
    ):
        assert AgenticRAGConfig().sufficiency_timeout_s == 2.0
        assert GroundedGenerationConfig().max_retries == 0


def test_grounded_clamp_reads_policy():
    from src.context import (
        AnswerGenerationRequest,
        GroundedGenerationConfig,
        generate_answer,
    )
    from tests.unit.test_grounded_generation import SequenceLLM, _bundle, _draft

    llm = SequenceLLM(
        _draft(("Tomorrow's forecast predicts rain.", ["D1"])),
        _draft(("FAISS enables vector similarity search.", ["D1"])),
    )
    with overridden({"llm": {"grounded_max_retries": 0}}):
        generate_answer(
            AnswerGenerationRequest(
                question="What is FAISS?",
                context=_bundle(),
                grounded_generation=GroundedGenerationConfig(max_retries=1),
            ),
            llm=llm,
        )
    assert len(llm.calls) == 1


def test_tool_loop_config_and_recovery_follow_policy():
    from src.agents.tool.recovery import RecoveryPolicy
    from src.agents.tool.tool_calling import ToolAgentLoopConfig

    with overridden(
        {
            "tool_loop": {
                "approval_timeout_seconds": 11,
                "escalation_timeout_seconds": 22,
                "max_escalations": 4,
                "recovery": {
                    "max_retries": 1,
                    "backoff_seconds": [0.1],
                    "retry_budget_seconds": 3,
                },
            }
        }
    ):
        cfg = ToolAgentLoopConfig()
        policy = RecoveryPolicy()
    assert (
        cfg.approval_timeout_seconds,
        cfg.escalation_timeout_seconds,
        cfg.max_escalations,
    ) == (11.0, 22.0, 4)
    assert (policy.max_retries, policy.backoff, policy.retry_budget) == (1, (0.1,), 3.0)
    assert RecoveryPolicy(max_retries=0).max_retries == 0


def test_brokers_follow_policy():
    from src.internal.servers.web.tool_approval import (
        ToolApprovalBroker,
        ToolEscalationBroker,
    )

    with overridden(
        {
            "tool_loop": {
                "approval_timeout_seconds": 11,
                "escalation_timeout_seconds": 22,
            }
        }
    ):
        assert ToolApprovalBroker()._timeout_seconds == 11.0
        assert ToolEscalationBroker()._timeout_seconds == 22.0


def test_escalation_timeout_is_overridable_end_to_end(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.internal.configs import AppSettings, AuthSettings
    from src.internal.configs.timeouts import TIMEOUTS_PATH_ENV
    from src.internal.servers.web.app import SearchExperienceSettings, create_web_app

    f = tmp_path / "t.toml"
    f.write_text("[tool_loop]\nescalation_timeout_seconds = 42\n")
    monkeypatch.setenv(TIMEOUTS_PATH_ENV, str(f))

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=AppSettings(auth=AuthSettings(super_users=("admin",))),
    )
    with TestClient(app) as client:
        assert client is not None
        assert app.state.tool_escalation_broker._timeout_seconds == 42.0


def test_tool_evidence_timeout_follows_policy(monkeypatch):
    import src.context.tool_evidence as te
    from src.context.tool_evidence import ToolDescriptor, ToolRequest, ToolSafety

    seen = []
    real_wait_for = asyncio.wait_for

    async def spy(aw, timeout):
        seen.append(timeout)
        return await real_wait_for(aw, timeout)

    monkeypatch.setattr(te.asyncio, "wait_for", spy)

    class Registry:
        def list_tools(self):
            return [ToolDescriptor(name="t1", safety=ToolSafety.READ_ONLY)]

        async def invoke(self, request):
            return {"ok": True}

    class Selector:
        def select(self, query, tools):
            return [ToolRequest(tool_name="t1")]

    with overridden({"tool_loop": {"tool_evidence_timeout_seconds": 2}}):
        asyncio.run(te.collect_tool_evidence("q", Registry(), Selector()))
    assert seen and set(seen) == {2.0}


def test_sse_heartbeat_env_still_wins(monkeypatch):
    from src.internal.servers import sse

    monkeypatch.setenv("AGENTIC_SEARCH_SSE_HEARTBEAT_SECONDS", "0")
    assert sse.heartbeat_seconds() == 0.0
