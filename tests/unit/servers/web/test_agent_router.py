"""Unit tests for the 3-way agentic router decision logic."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from src.context.models import ChatMessage
from src.internal.configs import AppSettings
from src.internal.servers.web.intent import RouteStrategy, recognize_intent
from src.internal.servers.web.intent import recognizer as ir
from src.internal.servers.web.intent import similarity
from src.internal.servers.web.intent.recognizer import classify_route
from src.internal.servers.web.intent.rules import (
    _regex_route,
    _rule_based_route,
)
from src.internal.servers.web.intent.types import IntentModelDecision


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.call_kwargs: list[dict] = []

    def complete(self, messages: list[ChatMessage], **kwargs) -> str:
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        return self._reply


def _strategy_for(query, **kwargs):
    return recognize_intent(query, **kwargs).strategy


def test_classify_route_uses_deterministic_decoding():
    # The strategy classifier must decode deterministically (temperature 0), so
    # the same query always routes to the same strategy/source run-to-run.
    llm = _FakeLLM("chat")
    classify_route("compare dense and sparse retrieval", llm)

    assert llm.call_kwargs, "the classifier consulted the LLM"
    assert llm.call_kwargs[0].get("temperature") == 0.0


# --- recognition cascade ---


def test_explicit_source_routes_to_search_agent():
    # An explicit source provider is an unambiguous search command.
    strategy = _strategy_for(
        "anything at all",
        llm=_FakeLLM("chat"),
        explicit_source=True,
    )
    assert strategy is RouteStrategy.SEARCH


def test_recognize_intent_without_llm_uses_rule_based():
    # No LLM → rule-based route; a search verb yields SEARCH.
    strategy = _strategy_for(
        "find the latest pricing sheet",
        llm=None,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.SEARCH


def test_recognize_intent_uses_llm_classifier_when_available():
    llm = _FakeLLM("tool")
    strategy = _strategy_for(
        "create a Jira ticket for the outage",
        llm=llm,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.TOOL
    assert llm.calls  # the classifier consulted the LLM


def test_recognize_intent_bare_lookup_is_search_and_skips_classifier():
    # A bare entity/term lookup like "FAISS" is unambiguously a grounded search,
    # so it must NOT reach the (over-eager) LLM classifier that would otherwise
    # send it to chat. Deterministic regardless of the LLM reply.
    llm = _FakeLLM("chat")
    strategy = _strategy_for("FAISS", llm=llm, explicit_source=False)
    assert strategy is RouteStrategy.SEARCH
    assert llm.calls == []  # classifier was never consulted


def test_recognize_intent_descriptive_phrase_still_uses_classifier():
    # A multi-word descriptive phrase is NOT a bare lookup → classifier decides.
    llm = _FakeLLM("chat")
    strategy = _strategy_for(
        "the procurement approval flow",
        llm=llm,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.CHAT
    assert llm.calls  # the classifier was consulted


# --- _rule_based_route ---


def test_rule_based_bare_lookup_routes_to_search():
    assert _rule_based_route("FAISS") is RouteStrategy.SEARCH
    assert _rule_based_route("vector database") is RouteStrategy.SEARCH


def test_rule_based_action_routes_to_tool():
    assert _rule_based_route("send an email to the team") is RouteStrategy.TOOL
    assert _rule_based_route("create a ticket for this bug") is RouteStrategy.TOOL


def test_rule_based_search_verb_routes_to_search():
    assert _rule_based_route("find the Q3 revenue report") is RouteStrategy.SEARCH
    assert _rule_based_route("look up the latest release notes") is RouteStrategy.SEARCH


def test_rule_based_generative_routes_to_chat():
    assert _rule_based_route("write a haiku about the sea") is RouteStrategy.CHAT
    assert _rule_based_route("translate this sentence to French") is RouteStrategy.CHAT


def test_rule_based_default_no_signal_routes_to_chat():
    assert _rule_based_route("the procurement approval flow") is RouteStrategy.CHAT


# --- classify_route ---


def test_classify_route_parses_each_label():
    for label, expected in [
        ("chat", RouteStrategy.CHAT),
        ("search", RouteStrategy.SEARCH),
        ("tool", RouteStrategy.TOOL),
    ]:
        assert classify_route("q", _FakeLLM(label))[0] is expected


def test_classify_route_reports_none_on_garbage():
    # An unusable response returns None; recognize_intent supplies
    # the chat default, and decides whether that default is a guess.
    assert classify_route("q", _FakeLLM("nonsense reply"))[0] is None
    assert classify_route("q", _FakeLLM(""))[0] is None


@pytest.mark.parametrize(
    "reply", ["not chat; search", "search or tool", "chat, search, tool"]
)
def test_classifier_rejects_conflicting_labels(reply):
    strategy, detail = classify_route("query", _FakeLLM(reply))

    assert strategy is None
    assert detail == {"raw_label": "unexpected"}


def test_bare_lookup_excludes_greetings_and_generative():
    from src.internal.servers.web.intent.rules import _is_bare_lookup

    assert _is_bare_lookup("hello") is False
    assert _is_bare_lookup("poem") is False
    assert _is_bare_lookup("translate this") is False
    # A genuine entity lookup is still a bare lookup.
    assert _is_bare_lookup("FAISS") is True


def test_recognize_intent_greeting_routes_to_chat_without_llm():
    # No LLM → rule-based; a bare greeting must NOT short-circuit to SEARCH.
    strategy = _strategy_for("hello", llm=None, explicit_source=False)
    assert strategy is RouteStrategy.CHAT


@pytest.mark.parametrize(
    "query", ["hi", "HI!", "hello", "hi there", "thanks", "thank you."]
)
def test_standalone_greeting_is_chat_without_clarification(query):
    decision = recognize_intent(query, llm=None, explicit_source=False)

    assert decision.strategy is RouteStrategy.CHAT
    assert decision.clarification is None


@pytest.mark.parametrize(
    ("query", "classifier_calls"),
    [("hello world tutorial", 1), ("hi, find the report", 1)],
)
def test_greeting_words_inside_requests_do_not_force_chat(query, classifier_calls):
    llm = _FakeLLM("search")

    decision = recognize_intent(query, llm=llm, explicit_source=False)

    assert decision.strategy is RouteStrategy.SEARCH
    assert len(llm.calls) == classifier_calls


@pytest.mark.parametrize(
    "first_import",
    [
        "src.internal.servers.web.intent",
        "src.internal.servers.web.intent.similarity",
    ],
)
def test_intent_import_order_does_not_require_encoder_dependencies(first_import):
    script = textwrap.dedent(
        f"""
        import importlib
        import sys

        class BlockEncoderDependencies:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {{'torch', 'sentence_transformers'}}:
                    raise ImportError(f'blocked optional dependency: {{fullname}}')
                return None

        sys.meta_path.insert(0, BlockEncoderDependencies())
        importlib.import_module({first_import!r})
        from src.internal.configs import AppSettings
        from src.internal.servers.web.intent import recognize_intent

        decision = recognize_intent(
            'hi',
            llm=None,
            explicit_source=False,
            settings=AppSettings(intent_index_path=None),
        )
        assert decision.strategy.value == 'chat'
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_omitted_settings_apply_shadow_mode(monkeypatch):
    settings = AppSettings(intent_shadow_mode=True)
    monkeypatch.setattr(ir, "load_app_settings", lambda: settings)
    observed_settings = []

    def _predict(_query, *, settings=None):
        observed_settings.append(settings)
        return IntentModelDecision(RouteStrategy.SEARCH, 0.91, 1.5)

    monkeypatch.setattr(similarity, "predict_route", _predict)

    decision = recognize_intent(
        "the vendor contract renewal terms",
        llm=_FakeLLM("chat"),
        explicit_source=False,
    )

    assert observed_settings == [settings]
    assert decision.strategy is RouteStrategy.CHAT
    assert decision.metadata["route_shadow_intent"] == "search"
    assert decision.metadata["route_mechanism"] == "classifier"


def test_omitted_settings_disable_clarification(monkeypatch):
    monkeypatch.setattr(
        ir, "load_app_settings", lambda: AppSettings(route_clarification=False)
    )
    monkeypatch.setattr(similarity, "predict_route", lambda *_a, **_k: None)

    decision = recognize_intent(
        "Review the vendor renewal terms", llm=None, explicit_source=False
    )

    assert decision.strategy is RouteStrategy.CHAT
    assert decision.clarification is None
    assert decision.metadata["route_mechanism"] == "heuristic_default"


def test_classify_route_ignores_substring_false_positives():
    # Word-boundary match: "research" must not count as the "search" label.
    assert classify_route("q", _FakeLLM("researching options"))[0] is None
    assert classify_route("q", _FakeLLM("chatbot style"))[0] is None
    # Exact labels still parse.
    assert classify_route("q", _FakeLLM("search"))[0] is RouteStrategy.SEARCH


# --- _regex_route (deterministic pre-LLM pass) ---


@pytest.mark.parametrize(
    "query,expected",
    [
        # TOOL — unambiguous imperative at the start
        ("send an email to Bob", RouteStrategy.TOOL),
        ("schedule a meeting for Friday", RouteStrategy.TOOL),
        # TOOL — ambiguous verb, but object-qualified
        ("create a ticket for the outage", RouteStrategy.TOOL),
        ("open an issue about the crash", RouteStrategy.TOOL),
        # SEARCH — bare term / lookup imperative
        ("FAISS", RouteStrategy.SEARCH),
        ("find the Q3 revenue report", RouteStrategy.SEARCH),
        ("look up the release notes", RouteStrategy.SEARCH),
        # CHAT — question / explain / generative / trailing '?'
        ("What is FAISS?", RouteStrategy.CHAT),
        ("explain how to send an email", RouteStrategy.CHAT),
        ("write a haiku about the sea", RouteStrategy.CHAT),
        ("is this thing on?", RouteStrategy.CHAT),
        # None — currency conflict on a chat-form question → defer to LLM
        ("what is the latest price of NVDA", None),
        # None — no confident signal → defer to LLM
        ("the procurement approval flow", None),
        ("", None),
        # SEARCH — a lookup verb wins over both the trailing '?' and the
        # currency cue (the currency short-circuit only applies to the CHAT
        # branch, not SEARCH).
        ("find the latest report?", RouteStrategy.SEARCH),
    ],
)
def test_regex_route(query, expected):
    assert _regex_route(query) is expected


def test_regex_route_tool_verb_needs_object_when_ambiguous():
    # A bare ambiguous verb must NOT misfire to TOOL without an object.
    assert _regex_route("open source models") is RouteStrategy.SEARCH
    assert _regex_route("post office hours") is RouteStrategy.SEARCH


def test_regex_route_polysemous_verbs_are_not_bare_tool_actions():
    # These verbs are common leading nouns/adjectives, not just imperatives,
    # so they must NOT short-circuit to TOOL without an object qualifier.
    assert (
        _regex_route("book recommendations for machine learning")
        is not RouteStrategy.TOOL
    )
    assert _regex_route("email templates for onboarding") is not RouteStrategy.TOOL
    assert _regex_route("schedule of the world cup") is not RouteStrategy.TOOL
    assert _regex_route("cancel culture explained") is not RouteStrategy.TOOL
    assert _regex_route("trigger warnings in modern media") is not RouteStrategy.TOOL
    # True positives must be preserved: unambiguous bare verb, and
    # object-qualified polysemous verb.
    assert _regex_route("send an email to Bob") is RouteStrategy.TOOL
    assert _regex_route("schedule a meeting for Friday") is RouteStrategy.TOOL


# --- recognition uses _regex_route before the LLM ---


def test_recognize_intent_confident_regex_skips_llm():
    # A confident chat-form question routes deterministically; the LLM classifier
    # is never consulted (previously this misrouted via the classifier).
    llm = _FakeLLM("search")  # would say search if consulted
    strategy = _strategy_for("What is FAISS?", llm=llm, explicit_source=False)
    assert strategy is RouteStrategy.CHAT
    assert llm.calls == []  # regex decided; classifier not consulted


def test_recognize_intent_ambiguous_falls_through_to_llm():
    # No confident regex match → the LLM classifier decides.
    llm = _FakeLLM("chat")
    strategy = _strategy_for(
        "the procurement approval flow",
        llm=llm,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.CHAT
    assert llm.calls  # classifier consulted


def test_recognize_intent_currency_conflict_defers_to_llm():
    # A chat-form question with a currency cue is NOT decided by regex.
    llm = _FakeLLM("search")
    strategy = _strategy_for(
        "what is the latest price of NVDA",
        llm=llm,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.SEARCH
    assert llm.calls  # deferred to the classifier


class _SpyLLM:
    def __init__(self):
        self.called = False

    def complete(self, *_a, **_k):
        self.called = True
        return "chat"


def test_recognize_intent_uses_model_when_confident(monkeypatch):
    settings = AppSettings()
    observed_settings = []

    def _predict(_query, *, settings=None):
        observed_settings.append(settings)
        return IntentModelDecision(
            strategy=RouteStrategy.SEARCH,
            confidence=0.9,
            latency_ms=1.25,
        )

    monkeypatch.setattr(similarity, "predict_route", _predict)
    llm = _SpyLLM()
    strategy = _strategy_for(
        "the vendor contract renewal terms",
        llm=llm,
        explicit_source=False,
        settings=settings,
    )
    assert strategy is RouteStrategy.SEARCH
    assert llm.called is False  # model replaced the LLM step
    assert observed_settings == [settings]


def test_recognize_intent_defers_to_llm_when_the_model_abstains(monkeypatch):
    """Abstention now arrives on the decision rather than as low confidence.

    This asserted a confidence-below-threshold deferral until that gate was
    removed for changing 3 decisions in 416. The property under test is
    unchanged -- an abstaining model must reach the LLM classifier -- only the
    way abstention is signalled has moved.
    """
    monkeypatch.setattr(
        similarity,
        "predict_route",
        lambda q, *, settings=None: IntentModelDecision(
            strategy=RouteStrategy.SEARCH,
            confidence=0.3,
            latency_ms=1.25,
            abstain_reason="margin_below_threshold",
        ),
    )
    llm = _SpyLLM()
    strategy = _strategy_for(
        "the vendor contract renewal terms",
        llm=llm,
        explicit_source=False,
    )
    assert llm.called is True  # abstained -> LLM fallback
    assert strategy is RouteStrategy.CHAT  # spy LLM returns "chat"


def test_recognize_intent_no_model_is_unchanged(monkeypatch):
    monkeypatch.setattr(similarity, "predict_route", lambda q, *, settings=None: None)
    llm = _SpyLLM()
    strategy = _strategy_for(
        "the vendor contract renewal terms",
        llm=llm,
        explicit_source=False,
    )
    assert llm.called is True  # no model -> today's behavior
    assert strategy is RouteStrategy.CHAT


def test_regex_still_wins_over_model(monkeypatch):
    called = {"model": False}

    def _spy(_q, *, settings=None):
        called["model"] = True
        return IntentModelDecision(
            strategy=RouteStrategy.CHAT,
            confidence=0.99,
            latency_ms=1.25,
        )

    monkeypatch.setattr(similarity, "predict_route", _spy)
    # "find X" matches the anchored search regex -> returns before predict_route.
    strategy = _strategy_for(
        "find the Q3 revenue report",
        llm=None,
        explicit_source=False,
    )
    assert strategy is RouteStrategy.SEARCH
    assert called["model"] is False


def test_recognize_intent_reports_mechanism_outside_debug_captures(monkeypatch):
    """Route telemetry must survive without the dev-only request capture.

    Request captures only run under AGENTIC_SEARCH_DEBUG_PANELS, so a caller
    needs the deciding mechanism handed back directly to persist it and recycle
    model errors into training.
    """
    telemetry: dict = {}
    decision = recognize_intent(
        "find the onboarding doc", llm=None, explicit_source=False
    )
    strategy = decision.strategy
    telemetry.update(decision.metadata)

    assert strategy is RouteStrategy.SEARCH
    assert telemetry["route_mechanism"] == "rules"

    monkeypatch.setattr(
        similarity,
        "predict_route",
        lambda query, settings=None: IntentModelDecision(
            strategy=RouteStrategy.TOOL,
            confidence=0.91,
            latency_ms=1.5,
        ),
    )
    telemetry = {}
    decision = recognize_intent(
        "vendor renewal terms archive and notes",
        llm=None,
        explicit_source=False,
    )
    strategy = decision.strategy
    telemetry.update(decision.metadata)

    assert strategy is RouteStrategy.TOOL
    assert telemetry["route_mechanism"] == "model"
    assert telemetry["route_confidence"] == 0.91
    assert telemetry["route_abstained"] is False


# --- recognize_intent: guess-site detection ---


def test_recognize_intent_clarifies_when_heuristic_has_no_signal():
    decision = recognize_intent(
        "Review the vendor renewal terms", llm=None, explicit_source=False
    )

    assert decision.strategy is RouteStrategy.CHAT  # legacy answer preserved
    assert decision.clarification is not None
    assert [option.route for option in decision.clarification.options] == [
        "chat",
        "search",
        "tool",
    ]


def test_recognize_intent_does_not_clarify_when_heuristic_matches_a_cue():
    decision = recognize_intent(
        "email the quarterly report to legal", llm=None, explicit_source=False
    )

    assert decision.strategy is RouteStrategy.TOOL
    assert decision.clarification is None


def test_recognize_intent_clarifies_on_unusable_llm_output():
    class _GarbageLLM:
        def complete(self, messages, **kwargs):
            return "I'm not sure what you mean"

    decision = recognize_intent(
        "Review the vendor renewal terms",
        llm=_GarbageLLM(),
        explicit_source=False,
    )

    assert decision.strategy is RouteStrategy.CHAT  # today's classifier default
    assert decision.clarification is not None


def test_recognize_intent_never_clarifies_on_a_deterministic_rule():
    for query in ("find the onboarding checklist", "Explain how FAISS works"):
        decision = recognize_intent(query, llm=None, explicit_source=False)
        assert decision.clarification is None


def test_recognize_intent_does_not_clarify_on_a_usable_llm_label():
    class _UsableLLM:
        def complete(self, messages, **kwargs):
            return "search"

    decision = recognize_intent(
        "Review the vendor renewal terms", llm=_UsableLLM(), explicit_source=False
    )

    assert decision.strategy is RouteStrategy.SEARCH
    assert decision.clarification is None


def test_recognize_intent_falls_through_to_heuristic_when_llm_raises():
    class _BrokenLLM:
        def complete(self, messages, **kwargs):
            raise RuntimeError("provider down")

    decision = recognize_intent(
        "email the quarterly report to legal",
        llm=_BrokenLLM(),
        explicit_source=False,
    )

    assert decision.strategy is RouteStrategy.TOOL
    assert decision.clarification is None


def test_recognize_intent_never_clarifies_on_a_confident_model(monkeypatch):
    monkeypatch.setattr(
        similarity,
        "predict_route",
        lambda query, settings=None: IntentModelDecision(
            strategy=RouteStrategy.TOOL,
            confidence=0.91,
            latency_ms=1.5,
        ),
    )

    decision = recognize_intent(
        "Review the vendor renewal terms", llm=None, explicit_source=False
    )

    assert decision.strategy is RouteStrategy.TOOL
    assert decision.clarification is None


def test_recognize_intent_never_clarifies_for_an_explicit_source():
    decision = recognize_intent(
        "Review the vendor renewal terms", llm=None, explicit_source=True
    )

    assert decision.strategy is RouteStrategy.SEARCH
    assert decision.clarification is None


def test_recognize_intent_returns_chat_without_clarifying_for_empty_query():
    decision = recognize_intent("   ", llm=None, explicit_source=False)

    assert decision.strategy is RouteStrategy.CHAT
    assert decision.clarification is None


def test_recognize_intent_strategy_is_unchanged_at_every_guess_site():
    class _GarbageLLM:
        def complete(self, messages, **kwargs):
            return ""

    assert (
        _strategy_for(
            "Review the vendor renewal terms", llm=None, explicit_source=False
        )
        is RouteStrategy.CHAT
    )
    assert (
        _strategy_for(
            "Review the vendor renewal terms",
            llm=_GarbageLLM(),
            explicit_source=False,
        )
        is RouteStrategy.CHAT
    )
    assert (
        _strategy_for(
            "email the quarterly report to legal", llm=None, explicit_source=False
        )
        is RouteStrategy.TOOL
    )


def test_route_mechanisms_use_the_documented_vocabulary():
    telemetry: dict = {}
    decision = recognize_intent(
        "find the onboarding checklist",
        llm=None,
        explicit_source=False,
    )
    telemetry.update(decision.metadata)
    assert telemetry["route_mechanism"] == "rules"

    telemetry = {}
    decision = recognize_intent(
        "email the quarterly report to legal",
        llm=None,
        explicit_source=False,
    )
    telemetry.update(decision.metadata)
    assert telemetry["route_mechanism"] == "heuristic_default"

    telemetry = {}
    decision = recognize_intent(
        "Review the vendor renewal terms",
        llm=None,
        explicit_source=False,
    )
    telemetry.update(decision.metadata)
    assert telemetry["route_mechanism"] == "clarify"


def test_recognize_intent_honors_the_clarification_setting():
    settings = AppSettings(route_clarification=False)
    decision = recognize_intent(
        "Review the vendor renewal terms",
        llm=None,
        explicit_source=False,
        settings=settings,
    )

    assert decision.strategy is RouteStrategy.CHAT
    assert decision.clarification is None
