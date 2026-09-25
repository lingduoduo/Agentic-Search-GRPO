# tests/unit/test_intent_routing.py
import asyncio
import json
import json as _json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.agents.core.base import AgentLoopOutput
from src.internal.configs import AppSettings
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web.intent import RouteStrategy, recognize_intent
from src.internal.servers.web.intent import similarity
from src.internal.servers.web.tool_agent_runner import _infer_intent_from_output
from src.internal.tools.routing_tools import (
    build_rag_routing_tool,
    build_search_routing_tool,
)
from src.internal.tools import ToolEffect
from src.model.pre_training.intents.model import DEFAULT_ENCODER
from src.model.pre_training.intents.model import (
    INDEX_FILENAME,
    CanonicalExample,
    IntentIndex,
)


def _make_output(
    action_trace: str | None = None, final_answer: str | None = None
) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        action_trace=action_trace,
        final_answer=final_answer,
    )


# --- _infer_intent_from_output ---


def _trace(tool_name: str) -> str:
    return json.dumps(
        {
            "tool_name": tool_name,
            "status": "completed",
            "result": "ok",
            "performance": {},
            "error_code": None,
            "error_message": None,
            "optimization_suggestions": [],
            "retry_count": 0,
        }
    )


def test_infer_intent_corpus_search_tool():
    output = _make_output(action_trace=_trace("search"))
    assert _infer_intent_from_output(output) == "search"


def test_infer_intent_rag_routing_tool():
    output = _make_output(action_trace=_trace("rag_routing_tool"))
    assert _infer_intent_from_output(output) == "chat"


def test_infer_intent_mcp_tool():
    output = _make_output(action_trace=_trace("custom_api"))
    assert _infer_intent_from_output(output) == "tool"


def test_infer_intent_no_trace_defaults_to_chat():
    output = _make_output(action_trace=None)
    assert _infer_intent_from_output(output) == "chat"


def test_infer_intent_malformed_trace_defaults_to_chat():
    output = _make_output(action_trace="not json")
    assert _infer_intent_from_output(output) == "chat"


# --- routing tool builder tests ---


def test_build_search_routing_tool_schema():
    tool = build_search_routing_tool(
        search_url="http://localhost:8001/retrieve", top_k=3
    )
    assert tool.schema.name == "search"
    assert tool.effect is ToolEffect.READ_ONLY
    assert "query" in tool.schema.parameters.get("properties", {})


def test_build_rag_routing_tool_schema():
    tool = build_rag_routing_tool(
        llm=None, search_url="http://localhost:8001/retrieve", top_k=3
    )
    assert tool.schema.name == "rag_routing_tool"
    assert tool.effect is ToolEffect.READ_ONLY
    assert "query" in tool.schema.parameters.get("properties", {})


def test_search_routing_tool_returns_json():
    from src.internal.tools.search import SearchPage

    fake_pages = [SearchPage(title="Doc A", summary="Content A", url="http://a.com")]

    async def run():
        with patch(
            "src.internal.tools.routing_tools.search_tool",
            new=AsyncMock(return_value=fake_pages),
        ):
            tool = build_search_routing_tool(
                search_url="http://localhost:8001/retrieve", top_k=3
            )
            response_text, raw, _meta = await tool.execute(
                "default", {"query": "FAISS"}
            )
        return response_text

    result = asyncio.run(run())
    data = _json.loads(result)
    assert isinstance(data, list)
    assert data[0]["title"] == "Doc A"


def test_rag_routing_tool_returns_answer():
    from src.context.models import (
        AnswerGenerationResult,
        PromptBundle,
        SearchContextBundle,
    )

    fake_context = SearchContextBundle(query="q", documents=[])
    fake_prompt = PromptBundle(system="", user="", messages=[])
    fake_result = AnswerGenerationResult(
        answer="42",
        citations=["[D1]"],
        context=fake_context,
        prompt=fake_prompt,
    )

    async def run():
        with patch(
            "src.context.answer_with_retrieval", new=AsyncMock(return_value=fake_result)
        ):
            tool = build_rag_routing_tool(
                llm=None, search_url="http://localhost:8001/retrieve", top_k=3
            )
            response_text, raw, _meta = await tool.execute(
                "default", {"query": "What is the answer?"}
            )
        return response_text

    result = asyncio.run(run())
    data = _json.loads(result)
    assert data["answer"] == "42"


def test_recognize_intent_is_unchanged_by_a_none_returning_model(monkeypatch):
    """A margin abstention must look exactly like having no model at all."""
    monkeypatch.setattr(similarity, "predict_route", lambda q, settings=None: None)
    calls = []

    class _LLM:
        def complete(self, messages, **kwargs):
            calls.append(messages)
            return "search"

    decision = recognize_intent(
        "where does the reranker timeout live",
        llm=_LLM(),
        explicit_source=False,
    )

    assert decision.strategy is RouteStrategy.SEARCH
    assert calls, "the LLM classifier must still be consulted"


# --- recognize_intent end-to-end: exactly one intent_model capture stage ---

_AXIS = {"search": 0, "chat": 1, "tool": 2}
_MODULE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}

# Deliberately not "?" and more than 3 words with no tool/search/chat cue, so
# _regex_route defers and the query actually reaches predict_route instead of
# being decided deterministically at cascade step 2.
_MODEL_STAGE_QUERY = "the vendor contract renewal terms"


def _write_routing_index(tmp_path):
    examples, rows = [], []
    for route, axis in _AXIS.items():
        for position in range(12):
            examples.append(
                CanonicalExample(
                    f"{route}-{position}",
                    f"{route} {position}",
                    route,
                    (_MODULE[route],),
                )
            )
            rows.append(np.eye(3, dtype=np.float32)[axis])
    directory = tmp_path / "index"
    IntentIndex(examples, np.stack(rows), DEFAULT_ENCODER, "sha256:x").save(
        directory / INDEX_FILENAME
    )
    return directory


class _ChatLLM:
    def complete(self, messages, **kwargs):
        return "chat"


@pytest.mark.parametrize(
    "vector, expect_composite",
    [
        # Served: clear margin, no abstention.
        pytest.param([0.9950, 0.1005, 0.0], False, id="served"),
        # Margin abstention: the top two routes tie. predict_route returns the
        # decision with abstain_reason set, and recognize_intent records the one
        # stage -- previously predict_route returned None and recorded its own,
        # which is what kept this deferral out of production telemetry.
        pytest.param([0.707, 0.707, 0.0], False, id="margin_only"),
        # Margin abstention where the close runner-up ("tool") has an action
        # module, which is the composite signature.
        pytest.param([0.71, 0.0, 0.70], True, id="margin_and_composite"),
    ],
)
def test_recognize_intent_records_exactly_one_intent_model_stage(
    tmp_path, monkeypatch, vector, expect_composite
):
    """Parametrisation lost its confidence-abstention case with that gate.

    Two of the three cases used to be driven by an absolute-confidence floor.
    That gate was removed for changing 3 decisions in 416, so the cases are now
    served / margin-abstained / margin-abstained-and-composite -- which is the
    full set of shapes a decision can still take.
    """
    similarity._INTENT_INDEXES.clear()
    monkeypatch.setattr(
        similarity, "encode_texts", lambda texts: np.array([vector], dtype=np.float32)
    )
    settings = AppSettings(
        intent_index_path=_write_routing_index(tmp_path),
        intent_min_route_margin=0.05,
        intent_min_module_score=0.4,
    )
    token = rc.start_capture("r", _MODEL_STAGE_QUERY)
    try:
        decision = recognize_intent(
            _MODEL_STAGE_QUERY,
            llm=_ChatLLM(),
            explicit_source=False,
            settings=settings,
        )
        model_stages = [s for s in rc.active().stages if s.stage == "intent_model"]

        assert len(model_stages) == 1
        captured = model_stages[0].payload
        assert captured["composite"] is expect_composite
        assert (
            decision.metadata["route_predicted_intent"] == captured["predicted_intent"]
        )
        assert decision.metadata["route_confidence"] == captured["confidence"]
        assert decision.metadata["route_abstained"] == captured["abstained"]
        assert decision.metadata["route_model_latency_ms"] == captured["latency_ms"]
        assert decision.metadata["route_modules"] == captured["modules"]
        assert decision.metadata["route_composite"] == captured["composite"]
        if captured["abstained"]:
            assert (
                decision.metadata["route_fallback_reason"]
                == captured["fallback_reason"]
            )
    finally:
        rc.reset_capture(token)
        similarity._INTENT_INDEXES.clear()


def _route_with_telemetry(tmp_path, monkeypatch, vector, **overrides):
    """Route one query with **no capture active** — production conditions.

    Starting no capture is the point: ``request_capture`` only records under
    the debug panels, so anything asserted here is reaching the telemetry dict
    that is actually persisted with the session.
    """
    similarity._INTENT_INDEXES.clear()
    monkeypatch.setattr(
        similarity, "encode_texts", lambda texts: np.array([vector], dtype=np.float32)
    )
    settings = AppSettings(
        intent_index_path=_write_routing_index(tmp_path),
        intent_min_route_margin=0.05,
        intent_min_module_score=0.4,
        **overrides,
    )
    try:
        decision = recognize_intent(
            _MODEL_STAGE_QUERY,
            llm=_ChatLLM(),
            explicit_source=False,
            settings=settings,
        )
    finally:
        similarity._INTENT_INDEXES.clear()
    return decision, decision.metadata


def test_margin_abstention_is_distinguishable_in_production_telemetry(
    tmp_path, monkeypatch
):
    """The gate that does all the abstaining under e5 must be countable.

    The confidence floor cannot fire under this encoder (in-scope scores sit
    far above it), so every deferral the router makes is a margin abstention.
    It previously reached ``None`` inside predict_route, indistinguishable from
    "no index configured", and the only record of it was a capture stage that
    runs solely under the debug panels. "How often does the router defer, and
    why" was therefore unanswerable from production data.
    """
    # Confidence clears the default floor; the top two routes tie on margin.
    _, telemetry = _route_with_telemetry(tmp_path, monkeypatch, [0.707, 0.707, 0.0])

    assert telemetry["route_abstained"] is True
    assert telemetry["route_fallback_reason"] == "margin_below_threshold"
    # The only reason there is: the confidence gate that produced the other
    # was removed for changing 3 decisions in 416.
    assert telemetry["route_fallback_reason"] == "margin_below_threshold"


def test_modules_and_composite_reach_production_telemetry(tmp_path, monkeypatch):
    """Both were recorded only in the dev-only capture stage.

    The composite flag exists solely so a future plan-aware router can be
    designed against measured data; observable only under a debug panel, it
    gathered none.
    """
    _, telemetry = _route_with_telemetry(tmp_path, monkeypatch, [1.0, 0.0, 0.0])

    assert telemetry["route_abstained"] is False
    assert telemetry["route_modules"] == list(telemetry["route_modules"])
    assert telemetry["route_modules"], "a served route must carry its modules"
    assert telemetry["route_composite"] is False


def test_margin_abstention_still_defers_to_the_classifier(tmp_path, monkeypatch):
    """Observability only. The routing behavior must not have moved.

    ``_ChatLLM`` answers "chat", so a decision of CHAT proves the abstention
    fell through to the classifier rather than being served as the model's own
    (search) answer.
    """
    decision, telemetry = _route_with_telemetry(
        tmp_path, monkeypatch, [0.707, 0.707, 0.0]
    )

    assert decision.strategy is RouteStrategy.CHAT
    assert telemetry["route_mechanism"] == "classifier"


# --- shadow mode: observe without acting ---


def test_shadow_mode_records_the_prediction_without_acting_on_it(tmp_path, monkeypatch):
    """The whole point: production data on a router that is still dark.

    A vector that would be served confidently must still reach the LLM
    classifier, while the prediction it would have made is recorded. Asserting
    both halves matters — recording without the fall-through would be a silent
    promotion, and falling through without recording would gather nothing.
    """
    similarity._INTENT_INDEXES.clear()
    monkeypatch.setattr(
        similarity,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    settings = AppSettings(
        intent_index_path=_write_routing_index(tmp_path),
        intent_min_route_margin=0.05,
        intent_min_module_score=0.4,
        intent_shadow_mode=True,
    )
    token = rc.start_capture("r", _MODEL_STAGE_QUERY)
    try:
        decision = recognize_intent(
            _MODEL_STAGE_QUERY,
            llm=_ChatLLM(),
            explicit_source=False,
            settings=settings,
        )
        captured = next(
            stage.payload
            for stage in rc.active().stages
            if stage.stage == "intent_model"
        )
    finally:
        rc.reset_capture(token)
        similarity._INTENT_INDEXES.clear()

    telemetry = decision.metadata

    # Observed: the router would have said "search".
    assert telemetry["route_shadow_intent"] == "search"
    assert telemetry["route_shadow_abstained"] is False
    assert telemetry["route_shadow_intent"] == captured["predicted_intent"]
    assert telemetry["route_shadow_abstained"] == captured["abstained"]
    assert telemetry["route_shadow_fallback_reason"] == captured["fallback_reason"]
    # Not acted on: the classifier decided, and it answers "chat".
    assert decision.strategy is RouteStrategy.CHAT
    assert telemetry["route_mechanism"] == "classifier"
    assert telemetry["route_fallback_reason"] == "shadow_mode"


def test_shadow_fields_are_distinct_from_served_fields(tmp_path, monkeypatch):
    """A shadow run must never be readable as a served one.

    Both modes populate `route_predicted_intent`, because the prediction is
    genuinely made either way. Only shadow populates `route_shadow_intent`, and
    only a served route reaches `route_mechanism == "model"` — so the two are
    separable when the telemetry is read back in aggregate.
    """
    similarity._INTENT_INDEXES.clear()
    monkeypatch.setattr(
        similarity,
        "encode_texts",
        lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    index_path = _write_routing_index(tmp_path)
    common = {
        "intent_index_path": index_path,
        "intent_min_route_margin": 0.05,
        "intent_min_module_score": 0.4,
    }
    served: dict = {}
    shadow: dict = {}
    try:
        decision = recognize_intent(
            _MODEL_STAGE_QUERY,
            llm=_ChatLLM(),
            explicit_source=False,
            settings=AppSettings(**common),
        )
        served.update(decision.metadata)
        similarity._INTENT_INDEXES.clear()
        decision = recognize_intent(
            _MODEL_STAGE_QUERY,
            llm=_ChatLLM(),
            explicit_source=False,
            settings=AppSettings(**common, intent_shadow_mode=True),
        )
        shadow.update(decision.metadata)
    finally:
        similarity._INTENT_INDEXES.clear()

    assert served["route_mechanism"] == "model"
    assert "route_shadow_intent" not in served
    assert shadow["route_mechanism"] == "classifier"
    assert shadow["route_shadow_intent"] == served["route_predicted_intent"]


def test_shadow_mode_is_off_by_default():
    """Promotion-adjacent machinery must never arrive switched on."""
    assert AppSettings().intent_shadow_mode is False


def test_classify_route_bounds_the_llm_call_by_policy():
    from src.internal.configs.timeouts import (
        load_timeout_policies,
        use_timeout_policies,
    )
    from src.internal.servers.web.intent.recognizer import classify_route

    seen = []

    class _LLM:
        def complete(self, messages, **kwargs):
            seen.append(kwargs.get("timeout_override"))
            return "search"

    classify_route("where is the reranker timeout", _LLM())
    policies = load_timeout_policies(
        {}, overrides={"llm": {"route_classifier_timeout_seconds": 1.5}}
    )
    with use_timeout_policies(policies):
        classify_route("where is the reranker timeout", _LLM())
    assert seen == [3.0, 1.5]


def test_classifier_timeout_falls_back_to_the_rules_router(monkeypatch):
    from src.context.models import LLMTimeoutError

    monkeypatch.setattr(similarity, "predict_route", lambda q, settings=None: None)

    calls = []

    class _Slow:
        def complete(self, messages, **kwargs):
            calls.append(1)
            raise LLMTimeoutError("LLM request timed out")

    # A query the regex rules and the (patched-out) model leave to the LLM.
    decision = recognize_intent(
        "where does the reranker timeout live", llm=_Slow(), explicit_source=False
    )
    assert calls == [1]  # the classifier was reached, then timed out
    assert decision.metadata.get("route_mechanism") in {"heuristic_default", "clarify"}
