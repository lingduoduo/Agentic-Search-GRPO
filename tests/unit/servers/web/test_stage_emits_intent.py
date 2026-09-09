from __future__ import annotations

import logging

from src.context.models import ChatMessage
from src.internal.servers.web import request_capture as rc
from src.internal.servers.web.intent import recognize_intent
from src.internal.servers.web.intent import recognizer as ir
from src.internal.servers.web.intent import similarity
from src.internal.servers.web.intent.types import IntentModelDecision


class _FakeLLM:
    def __init__(self, reply: str = "search") -> None:
        self.calls = 0
        self.reply = reply

    def complete(self, messages: list[ChatMessage], **_) -> str:
        self.calls += 1
        return self.reply


def _strategy_for(query, **kwargs):
    return recognize_intent(query, **kwargs).strategy


def _intent_stages():
    return [s for s in rc.active().stages if s.stage == "intent"]


def test_recognize_intent_emits_regex_intent_stage():
    token = rc.start_capture("r", "What is FAISS?")
    try:
        strategy = _strategy_for(
            "What is FAISS?",
            llm=_FakeLLM(),
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].label == "rules"
        assert stages[0].payload["mechanism"] == "rules"
        assert stages[0].payload["strategy"] == strategy.value  # "chat"
    finally:
        rc.reset_capture(token)


def test_recognize_intent_emits_classifier_intent_stage_with_detail():
    # A phrase _regex_route defers on → classifier path, preserving only the
    # safe raw label rather than the prompt containing the user's query.
    token = rc.start_capture("r", "the procurement approval flow")
    try:
        _strategy_for(
            "the procurement approval flow",
            llm=_FakeLLM(
                "search — request was private_sentinel procurement approval flow"
            ),
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].label == "classifier"
        assert stages[0].payload["mechanism"] == "classifier"
        assert stages[0].payload["raw_label"] == "search"
        assert "prompt" not in stages[0].payload
        assert "private_sentinel" not in repr(stages[0].payload)
    finally:
        rc.reset_capture(token)


def test_unexpected_classifier_output_is_redacted_from_capture_and_logs(caplog):
    token = rc.start_capture("r", "the private_sentinel procurement approval flow")
    try:
        with caplog.at_level(logging.WARNING, logger=ir.__name__):
            _strategy_for(
                "the private_sentinel procurement approval flow",
                llm=_FakeLLM(
                    "unrecognized response echoing private_sentinel procurement"
                ),
                explicit_source=False,
            )

        payload = _intent_stages()[0].payload
        assert payload["raw_label"] == "unexpected"
        assert "private_sentinel" not in repr(payload)
        assert "private_sentinel" not in caplog.text
    finally:
        rc.reset_capture(token)


def test_classifier_exception_message_is_redacted_while_rule_fallback_runs(
    monkeypatch, caplog
):
    class _FailingLLM:
        def complete(self, messages: list[ChatMessage], **_) -> str:
            raise RuntimeError("private_sentinel from classifier prompt")

    monkeypatch.setattr(
        similarity, "predict_route", lambda query, *, settings=None: None
    )
    query = "please send an email to Bob"
    token = rc.start_capture("r", query)
    try:
        with caplog.at_level(logging.WARNING, logger=ir.__name__):
            strategy = _strategy_for(
                query,
                llm=_FailingLLM(),
                explicit_source=False,
            )

        payload = _intent_stages()[0].payload
        assert strategy is ir.RouteStrategy.TOOL
        assert payload["mechanism"] == "heuristic_default"
        assert payload["strategy"] == "tool"
        assert "Route classifier failed, using rule-based." in caplog.text
        assert "private_sentinel" not in caplog.text
        assert "private_sentinel" not in repr(payload)
    finally:
        rc.reset_capture(token)


def test_recognize_intent_emits_explicit_source_intent_stage():
    token = rc.start_capture("r", "anything at all")
    try:
        _strategy_for("anything at all", llm=None, explicit_source=True)
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].payload["mechanism"] == "explicit_source"
        assert stages[0].payload["strategy"] == "search"
    finally:
        rc.reset_capture(token)


def test_recognize_intent_emits_clarify_intent_stage_without_llm():
    # No LLM and no dominant heuristic cue → a guess, recorded as "clarify".
    token = rc.start_capture("r", "the procurement approval flow")
    try:
        _strategy_for(
            "the procurement approval flow",
            llm=None,
            explicit_source=False,
        )
        stages = _intent_stages()
        assert len(stages) == 1
        assert stages[0].payload["mechanism"] == "clarify"
    finally:
        rc.reset_capture(token)


def test_recognize_intent_no_capture_does_not_raise():
    # With no active capture the emit is a silent no-op.
    _strategy_for("q", llm=_FakeLLM(), explicit_source=False)


def test_confident_model_records_evaluation_and_final_intent(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "predict_route",
        lambda query, *, settings=None: IntentModelDecision(
            strategy=ir.RouteStrategy.SEARCH,
            confidence=0.91,
            latency_ms=2.5,
        ),
    )
    llm = _FakeLLM()
    token = rc.start_capture("r", "the vendor contract renewal terms")
    try:
        strategy = _strategy_for(
            "the vendor contract renewal terms",
            llm=llm,
            explicit_source=False,
        )
        model_stages = [s for s in rc.active().stages if s.stage == "intent_model"]
        intent_stages = _intent_stages()

        assert strategy is ir.RouteStrategy.SEARCH
        assert len(model_stages) == 1
        assert model_stages[0].label == "evaluation"
        assert model_stages[0].payload == {
            "predicted_intent": "search",
            "confidence": 0.91,
            # Carried on the decision now that predict_route no longer records
            # its own stage; without it the margin would be lost from the
            # capture entirely.
            "margin": 0.0,
            "abstained": False,
            "fallback_reason": None,
            "latency_ms": 2.5,
            "modules": [],
            "composite": False,
        }
        assert len(intent_stages) == 1
        assert intent_stages[0].label == "model"
        assert intent_stages[0].payload["mechanism"] == "model"
        assert intent_stages[0].payload["predicted_intent"] == "search"
        assert intent_stages[0].payload["confidence"] == 0.91
        assert intent_stages[0].payload["latency_ms"] == 2.5
        assert llm.calls == 0
    finally:
        rc.reset_capture(token)


def test_abstaining_model_records_evaluation_and_classifier_fallback(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "predict_route",
        # Abstention is now expressed on the decision itself. It used to be
        # inferred from confidence < threshold, and that gate is gone.
        lambda query, *, settings=None: IntentModelDecision(
            strategy=ir.RouteStrategy.TOOL,
            confidence=0.42,
            latency_ms=3.25,
            abstain_reason="margin_below_threshold",
        ),
    )
    token = rc.start_capture("r", "the vendor contract renewal terms")
    try:
        strategy = _strategy_for(
            "the vendor contract renewal terms",
            llm=_FakeLLM(
                "search — request was private_sentinel vendor contract renewal terms"
            ),
            explicit_source=False,
        )
        model_stages = [s for s in rc.active().stages if s.stage == "intent_model"]
        intent_stages = _intent_stages()

        assert strategy is ir.RouteStrategy.SEARCH
        assert model_stages[0].payload["predicted_intent"] == "tool"
        assert model_stages[0].payload["abstained"] is True
        assert model_stages[0].payload["fallback_reason"] == "margin_below_threshold"
        assert intent_stages[0].label == "classifier"
        assert intent_stages[0].payload["raw_label"] == "search"
        assert intent_stages[0].payload["fallback_reason"] == "margin_below_threshold"
        assert "prompt" not in intent_stages[0].payload
        assert "private_sentinel" not in repr(intent_stages[0].payload)
    finally:
        rc.reset_capture(token)


def test_missing_model_does_not_fabricate_prediction_detail(monkeypatch):
    monkeypatch.setattr(
        similarity, "predict_route", lambda query, *, settings=None: None
    )
    token = rc.start_capture("r", "the vendor contract renewal terms")
    try:
        _strategy_for(
            "the vendor contract renewal terms",
            llm=_FakeLLM(),
            explicit_source=False,
        )

        assert not [s for s in rc.active().stages if s.stage == "intent_model"]
        payload = _intent_stages()[0].payload
        assert "predicted_intent" not in payload
        assert "confidence" not in payload
        assert "fallback_reason" not in payload
    finally:
        rc.reset_capture(token)
