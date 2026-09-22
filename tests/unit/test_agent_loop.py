"""Unit tests for src.agent_loop."""

import asyncio
import dataclasses

import pytest

from src import (
    AgentLoopBase,
    AgentLoopConfig,
    AgentContext,
    PlainGenerationLoop,
    PlainGenerationLoopConfig,
    RolloutStep,
    SearchEvaluationConfig,
    SearchAgentLoop,
    SearchAgentLoopConfig,
    SearchResultEvaluator,
    SearchContext,
    SearchResult,
    SingleTurnAgentLoop,
    SingleTurnAgentLoopConfig,
    ToolAgentLoopConfig,
    get_registered_agent_loop,
    list_registered_agent_loops,
    register,
)
from src.agents.components.loop_controller import LoopController
from src.agents.core.base import (
    AgentLoopOutput,
    resolve_agent_name,
    CANONICAL_AGENT_NAMES,
)


class DummyTokenizerWithTemplate:
    chat_template = "dummy"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        assert tokenize is False
        return "ABC"

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class DummyTokenizerWithEncode:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class DummyServerManager:
    def __init__(self, response_ids):
        self.response_ids = response_ids
        self.calls = []
        self.index = 0

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": prompt_ids,
                "sampling_params": sampling_params,
            }
        )
        if self.response_ids and isinstance(self.response_ids[0], list):
            response = self.response_ids[self.index]
            self.index += 1
            return list(response)
        return list(self.response_ids)


class DummySyncServerManager:
    def __init__(self, response_ids):
        self.response_ids = response_ids

    def generate(self, request_id, prompt_ids, sampling_params):
        del request_id, prompt_ids, sampling_params
        return list(self.response_ids)


class ConcreteAgentLoop(AgentLoopBase):
    async def run(self, messages, sampling_params):
        del messages, sampling_params
        raise NotImplementedError


def test_agent_loop_output_defaults_to_empty_control_flow_trace() -> None:
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=0,
        metrics={},
        request_id="req",
    )

    assert output.control_flow_trace == []


def test_build_prompt_ids_uses_chat_template_when_available():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithTemplate(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=3, response_length=5),
    )
    prompt_ids = asyncio.run(
        loop.build_prompt_ids([{"role": "user", "content": "hello"}])
    )
    assert prompt_ids == [65, 66, 67]  # ord("ABC"), prompt_length=3


def test_resolve_cli_aliases():
    assert resolve_agent_name("single") == "plain_generation"
    assert resolve_agent_name("search") == "search_agent"
    assert resolve_agent_name("tool") == "tool_agent"


def test_resolve_canonical_names_passthrough():
    for name in ("plain_generation", "single_turn_agent", "search_agent", "tool_agent"):
        assert resolve_agent_name(name) == name


def test_resolve_rejects_non_registry_modes():
    for mode in ("chat_loop", "search_tool", "hybrid_search", "chat_once", "nope"):
        with pytest.raises(KeyError):
            resolve_agent_name(mode)


def test_canonical_names_are_registered():
    from src.agents.core.base import list_registered_agent_loops

    registered = set(list_registered_agent_loops())
    assert CANONICAL_AGENT_NAMES <= registered


def test_resolver_is_exported_from_public_package():
    # The CLI and web app import these from the public ``src`` package, so the
    # __init__ export path must work — not just the submodule path.
    from src import resolve_agent_name as public_resolve
    from src import CANONICAL_AGENT_NAMES as public_canonical

    assert public_resolve("search") == "search_agent"
    assert public_canonical == CANONICAL_AGENT_NAMES


def test_build_prompt_ids_falls_back_to_encode():
    """Over budget, whole messages are dropped rather than the text sliced.

    This previously asserted ``len(prompt_ids) == 4`` -- the budget filled
    exactly by slicing "abc\nde" mid-message. Dropping the older message whole
    leaves 2 tokens: under budget, and a prompt with no severed message in it.
    """
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=4, response_length=5),
    )
    prompt_ids = asyncio.run(
        loop.build_prompt_ids(
            [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "de"}]
        )
    )
    assert len(prompt_ids) <= 4
    assert prompt_ids == [ord("d"), ord("e")]  # the newest message, intact
    assert loop.prompt_messages_dropped == 1
    assert loop.prompt_hard_truncated is False


def test_build_prompt_ids_sync_with_chat_template_returns_int_list():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithTemplate(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=10, response_length=5),
    )
    result = loop._build_prompt_ids_sync([{"role": "user", "content": "hello"}])
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
    assert result == [65, 66, 67]  # encode("ABC") = [ord(c) for c in "ABC"]


def test_build_prompt_ids_sync_fallback_when_chat_template_is_none():
    class NoChatTemplateTokenizer:
        chat_template = None

        def encode(self, text):
            return [ord(c) for c in text]

    loop = ConcreteAgentLoop(
        tokenizer=NoChatTemplateTokenizer(),
        server_manager=DummyServerManager([]),
        config=AgentLoopConfig(prompt_length=5, response_length=5),
    )
    result = loop._build_prompt_ids_sync([{"role": "user", "content": "hi"}])
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
    assert result == [ord("h"), ord("i")]


def test_plain_generation_loop_runs_one_model_generation():
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(tokenizer.encode("plain answer"))
    loop = PlainGenerationLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=PlainGenerationLoopConfig(response_length=64),
    )

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "What is FAISS?"}],
            {"temperature": 0.0},
        )
    )

    assert output.num_turns == 1
    assert output.final_answer == "plain answer"
    assert output.context is None
    assert len(server_manager.calls) == 1


def test_search_agent_default_prompt_includes_training_template_boundaries():
    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
    )
    prompt = loop.search_config.system_prompt
    assert prompt is not None
    assert "<think>" in prompt
    assert "<search>" in prompt
    assert "<information>" in prompt
    assert "<answer>" in prompt
    assert "Never write or fabricate this block" in prompt


def test_generate_response_ids_truncates_to_response_length():
    server_manager = DummyServerManager([1, 2, 3, 4, 5])
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=server_manager,
        config=AgentLoopConfig(prompt_length=10, response_length=3),
    )
    response_ids = asyncio.run(
        loop.generate_response_ids([9, 9], {"temperature": 0.1}, request_id="req-1")
    )
    assert response_ids == [1, 2, 3]
    assert server_manager.calls[0]["request_id"] == "req-1"


# ── force_search=True (classic pre-retrieval RAG, backward-compat) ───────────


def test_single_turn_agent_loop_force_search_returns_expected_output():
    """force_search=True: retrieve unconditionally, then generate once."""
    server_manager = DummyServerManager([21, 22, 23, 24])
    loop = SingleTurnAgentLoop(
        tokenizer=DummyTokenizerWithTemplate(),
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(
            prompt_length=4, response_length=2, force_search=True
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("hello",): [
                [SearchResult(contents='"Greeting"\nHello world evidence')],
            ],
        }
    )
    output = asyncio.run(
        loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.7})
    )
    assert output.response_ids == [21, 22]
    assert output.response_mask == [1, 1]
    assert output.num_turns == 1
    assert output.context.num_rounds == 1
    assert output.context.queries == ["hello"]
    assert output.final_answer is not None
    assert output.trajectory_messages[-1]["role"] == "assistant"
    assert "retrieve" in output.metrics
    assert "generate_sequences" in output.metrics
    assert output.request_id is not None


def test_single_turn_agent_loop_force_search_injects_evidence_into_prompt():
    """force_search=True: retrieved evidence must appear in the prompt."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager([tokenizer.encode("Answer with evidence")])
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(response_length=64, force_search=True),
    )
    loop._search_client = FakeSearchClient(
        {
            ("what happened",): [
                [SearchResult(contents='"Doc A"\nEvidence body')],
            ],
        }
    )

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "what happened"}],
            {"temperature": 0.0},
        )
    )

    prompt_text = tokenizer.decode(output.prompt_ids, skip_special_tokens=False)
    assert "<information>" in prompt_text
    assert "Evidence body" in prompt_text
    assert output.context.num_results == 1


def test_single_turn_agent_loop_force_search_disabled_retrieval():
    """force_search=True but use_retrieval=False: no retrieval, one generation."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager([tokenizer.encode("Direct answer")])
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(
            response_length=64, force_search=True, use_retrieval=False
        ),
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.0})
    )

    assert loop._search_client is None
    assert output.context.num_rounds == 0
    assert output.num_turns == 1


# ── default mode: tool-augmented one-shot ─────────────────────────────────────


def test_single_turn_agent_loop_tool_augmented_search_then_answer():
    """Default mode: model emits <search>, retrieves, then emits <answer>."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            # First generation: model decides to search
            tokenizer.encode("<search>Nobel Prize Physics 2024</search>"),
            # Second generation: model answers with evidence
            tokenizer.encode("<answer>Hopfield and Hinton</answer>"),
        ]
    )
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(response_length=64),
    )
    loop._search_client = FakeSearchClient(
        {
            ("Nobel Prize Physics 2024",): [
                [SearchResult(contents='"Nobel 2024"\nHopfield and Hinton won')],
            ],
        }
    )

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "Who won the Nobel Prize in Physics 2024?"}],
            {"temperature": 0.8},
        )
    )

    assert output.num_turns == 2
    assert output.context.num_rounds == 1
    assert output.context.queries == ["Nobel Prize Physics 2024"]
    assert output.final_answer == "Hopfield and Hinton"
    assert "retrieve" in output.metrics
    assert "generate_sequences" in output.metrics
    assert "generate_sequences_2" in output.metrics
    assert output.metrics["search_rounds"] == 1.0
    # Trajectory must include: original user msg, assistant <search>, user <information>, assistant <answer>
    roles = [m["role"] for m in output.trajectory_messages]
    assert roles.count("assistant") >= 1


def test_single_turn_agent_loop_tool_augmented_direct_answer():
    """Default mode: model emits <answer> directly — no retrieval call."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [tokenizer.encode("<answer>Paris is the capital of France.</answer>")]
    )
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(response_length=64),
    )
    fake_client = FakeSearchClient({})
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "What is the capital of France?"}],
            {"temperature": 0.0},
        )
    )

    assert fake_client.calls == []  # no retrieval
    assert output.num_turns == 1  # only one generation step
    assert output.context.num_rounds == 0
    assert output.final_answer == "Paris is the capital of France."
    assert output.metrics["search_rounds"] == 0.0


def test_single_turn_agent_loop_tool_augmented_search_tag_no_retrieval():
    """Default mode: model emits <search> but use_retrieval=False — falls through to direct answer."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [tokenizer.encode("<search>some query</search>")]
    )
    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        config=SingleTurnAgentLoopConfig(response_length=64, use_retrieval=False),
    )
    fake_client = FakeSearchClient({})
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "hello"}], {"temperature": 0.0})
    )

    assert fake_client.calls == []  # retrieval skipped
    assert output.num_turns == 1  # no second generation step
    assert output.context.num_rounds == 0


def test_single_turn_agent_loop_tool_augmented_observation_in_second_prompt():
    """Default mode: after <search>, the second prompt contains <information>."""
    tokenizer = DummyTokenizerWithEncode()
    second_gen_prompt_ids: list[list[int]] = []

    class CapturingServerManager:
        call_count = 0

        async def generate(self, request_id, prompt_ids, sampling_params):
            self.call_count += 1
            if self.call_count == 1:
                return tokenizer.encode("<search>query</search>")
            second_gen_prompt_ids.append(list(prompt_ids))
            return tokenizer.encode("<answer>found it</answer>")

    loop = SingleTurnAgentLoop(
        tokenizer=tokenizer,
        server_manager=CapturingServerManager(),
        config=SingleTurnAgentLoopConfig(response_length=128),
    )
    loop._search_client = FakeSearchClient(
        {
            ("query",): [
                [SearchResult(contents='"Source"\nKey evidence here')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "question"}], {"temperature": 0.7})
    )

    assert output.num_turns == 2
    # The second generation prompt must include the retrieved evidence
    second_prompt_text = tokenizer.decode(
        second_gen_prompt_ids[0], skip_special_tokens=False
    )
    assert "<information>" in second_prompt_text
    assert "Key evidence here" in second_prompt_text


def test_register_stores_class_by_name():
    @register("test_agent_loop")
    class RegisteredLoop(ConcreteAgentLoop):
        pass

    assert RegisteredLoop.__name__ == "RegisteredLoop"


def test_get_registered_agent_loop_returns_single_turn_loop():
    registered = get_registered_agent_loop("single_turn_agent")
    assert registered is SingleTurnAgentLoop


def test_get_registered_agent_loop_returns_plain_generation_loop():
    registered = get_registered_agent_loop("plain_generation")
    assert registered is PlainGenerationLoop


def test_list_registered_agent_loops_includes_single_turn():
    assert "single_turn_agent" in list_registered_agent_loops()
    assert "plain_generation" in list_registered_agent_loops()


def test_tool_agent_loop_defaults_to_generic_json_parser():
    assert ToolAgentLoopConfig().tool_parser_format == "json"


def test_get_registered_agent_loop_raises_for_unknown_name():
    with pytest.raises(KeyError, match="Unknown agent loop"):
        get_registered_agent_loop("missing_loop")


def test_generate_response_ids_supports_sync_server_manager():
    loop = ConcreteAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummySyncServerManager([7, 8, 9, 10]),
        config=AgentLoopConfig(prompt_length=10, response_length=2),
    )
    response_ids = asyncio.run(
        loop.generate_response_ids([1, 2], {"temperature": 0.3}, request_id="req-sync")
    )
    assert response_ids == [7, 8]


def test_search_result_information_block_supports_citation_prefix():
    ctx = SearchContext(
        query="alpha",
        results=[SearchResult(contents='"Alpha"\nBeta', url="https://example.com")],
    )
    assert (
        ctx.to_information_block(citation_prefix="R1Q1D")
        == "[R1Q1D1] (Title: Alpha) Beta URL: https://example.com"
    )


def test_search_result_evaluator_marks_weak_results_as_insufficient():
    evaluator = SearchResultEvaluator(
        SearchEvaluationConfig(
            min_results_per_query=2,
            min_total_results=3,
            min_top_score=0.8,
            min_avg_score=0.7,
        )
    )
    evaluation = evaluator.evaluate_round(
        [
            SearchContext(
                query="alpha",
                results=[SearchResult(contents='"Alpha"\nbody', score=0.4)],
            )
        ]
    )

    assert evaluation.is_sufficient is False
    assert evaluation.total_results == 1
    assert "below minimum" in evaluation.to_feedback_block()


def test_search_result_evaluator_marks_strong_results_as_sufficient():
    evaluator = SearchResultEvaluator(
        SearchEvaluationConfig(
            min_results_per_query=1,
            min_total_results=2,
            min_top_score=0.8,
            min_avg_score=0.7,
        )
    )
    evaluation = evaluator.evaluate_round(
        [
            SearchContext(
                query="alpha",
                results=[
                    SearchResult(contents='"Alpha"\nbody', score=0.9),
                    SearchResult(contents='"Alpha 2"\nbody', score=0.8),
                ],
            )
        ]
    )

    assert evaluation.is_sufficient is True
    assert "Verdict: SUFFICIENT" in evaluation.to_feedback_block()


class FakeSearchClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.fetch_calls = []
        self.fetch_responses = {}
        self.closed = False

    async def retrieve(self, queries, topk=None, filters=None):
        del topk
        self.calls.append(list(queries))
        return self.responses[tuple(queries)]

    async def fetch_urls(self, urls):
        self.fetch_calls.append(list(urls))
        return self.fetch_responses[tuple(urls)]

    async def aclose(self):
        self.closed = True


def test_search_agent_loop_supports_plan_parallel_search_and_research_rounds():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<plan>Compare two sources and validate with a follow-up search.</plan>"
        ),
        tokenizer.encode("<searches>\n- first query\n- second query\n</searches>"),
        tokenizer.encode("<searches><query>refined query</query></searches>"),
        tokenizer.encode("<answer>Final report [R1Q1D1] [R2Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query", "second query"): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
                [SearchResult(contents='"Doc B"\nBeta body')],
            ],
            ("refined query",): [
                [SearchResult(contents='"Doc C"\nGamma body')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert loop._search_client.calls == [
        ["first query", "second query"],
        ["refined query"],
    ]
    assert output.context is not None
    assert output.context.num_rounds == 2
    assert output.context.num_searches == 3
    assert output.context.queries == ["first query", "second query", "refined query"]
    assert output.num_turns == 4


def test_search_agent_loop_auto_searches_when_model_emits_no_action():
    """The real guarantee: a tag-less dead-end still fires retrieval once.

    The model here is *genuinely* tag-less — it never emits a parseable tag on
    any turn, including the forced-answer turn. The auto-search therefore makes
    retrieval run (search_rounds == 1) instead of dead-ending with zero
    evidence. It does NOT manufacture an answer: with no <answer> tag and only a
    refusal-style generation, ``final_answer`` stays None (the no-fabricate
    invariant, see test_forced_turn_emitting_no_answer_returns_none). The
    feature's honest contract is "retrieval fires at least once", not "produces
    a non-empty answer" — a capable model that emits <answer> when prompted is
    covered by test_deadend_forces_answer_from_evidence.
    """
    tokenizer = DummyTokenizerWithEncode()
    # Never emits any recognized tag, on every turn (incl. the forced turn).
    responses = [tokenizer.encode("I cannot follow the required format.")] * 6
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            # First tag-less turn is the dead-end, so the auto-search fires
            # immediately rather than after re-prompts.
            max_consecutive_format_errors=1,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("What is FAISS?",): [
                [SearchResult(contents='"FAISS"\nFacebook AI Similarity Search')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "What is FAISS?"}], {"temperature": 0.0})
    )

    # Auto-search fired on the user's question despite no <search> tag: retrieval
    # ran exactly once, evidence was collected, and the event was recorded.
    assert loop._search_client.calls == [["What is FAISS?"]]
    assert output.context is not None
    assert output.context.num_rounds == 1
    assert output.metrics["search_rounds"] == 1.0
    assert any(e.action == "auto_search" for e in output.control_flow_trace)
    # No <answer> tag ever emitted → no fabricated answer.
    assert output.final_answer is None


def test_search_agent_loop_auto_search_disabled_preserves_format_recovery():
    """With auto_search_on_deadend=False, a tag-less run never triggers search."""
    tokenizer = DummyTokenizerWithEncode()
    responses = [tokenizer.encode("just rambling, no tags here")] * 5
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            auto_search_on_deadend=False,
            force_answer_on_deadend=False,
        ),
    )
    loop._search_client = FakeSearchClient({})

    output = asyncio.run(
        loop.run([{"role": "user", "content": "What is FAISS?"}], {"temperature": 0.0})
    )

    assert loop._search_client.calls == []  # retrieval never invoked
    assert output.metrics["search_rounds"] == 0.0


def _two_round_plateau_loop(evidence_plateau_min_gain):
    """A 2-round search loop where round 2's evidence equals round 1 (a plateau)."""
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nfirst query\n</searches>"),
        tokenizer.encode("<searches>\nsecond query\n</searches>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            require_sufficient_evidence_before_answer=False,
            evidence_plateau_min_gain=evidence_plateau_min_gain,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    # Both rounds return one equally-strong doc → identical per-round evidence
    # score → round 2's marginal gain is 0 (a plateau).
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body', score=0.5)]
            ],
            ("second query",): [
                [SearchResult(contents='"Doc B"\nBeta body', score=0.5)]
            ],
        }
    )
    return loop


def test_loop_emits_early_stops_when_plateau_configured():
    loop = _two_round_plateau_loop(evidence_plateau_min_gain=0.05)

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.metrics["search_rounds"] == 2.0
    assert output.metrics["early_stops"] >= 1.0


def test_loop_early_stops_zero_by_default():
    loop = _two_round_plateau_loop(evidence_plateau_min_gain=None)

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.metrics["search_rounds"] == 2.0
    assert output.metrics["early_stops"] == 0.0


def _two_round_loop_at_budget(max_search_limit):
    """The plateau harness, but with the round-2 search budget configurable.

    Round 2 has zero marginal evidence gain, so the plateau condition holds. With
    ``max_search_limit=2`` round 2 is also *at* the budget, which is the case the
    two stop reasons overlap on.
    """
    loop = _two_round_plateau_loop(evidence_plateau_min_gain=0.05)
    loop.search_config = dataclasses.replace(
        loop.search_config, max_search_limit=max_search_limit
    )
    loop._loop_controller = LoopController(loop.search_config)
    return loop


def _information_blocks(output):
    return [
        m["content"]
        for m in output.trajectory_messages
        if m["role"] == "user" and "<information>" in m["content"]
    ]


def test_the_final_round_within_budget_still_reaches_the_model():
    """A round at the search budget must still have its evidence injected.

    LoopController checks the budget before the plateau, and SearchAgentLoop acts
    only on PLATEAU -- so at the budget the round proceeds normally and its
    <information> is appended. That ordering is load-bearing: drop the budget arm
    and this round instead early-stops on the plateau, and the last round's
    evidence never reaches the model that has to answer from it.
    """
    loop = _two_round_loop_at_budget(max_search_limit=2)

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.metrics["search_rounds"] == 2.0
    blocks = _information_blocks(output)
    assert any("Beta body" in block for block in blocks), (
        f"round 2's evidence never reached the model; information blocks seen: {blocks}"
    )
    assert output.metrics["plateau_early_stop"] == 0.0


def test_a_plateau_below_budget_does_early_stop():
    """The contrast case: with budget to spare, the same plateau stops the search."""
    loop = _two_round_loop_at_budget(max_search_limit=6)

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.metrics["plateau_early_stop"] == 1.0
    assert not any("Beta body" in block for block in _information_blocks(output))


def test_search_agent_loop_injects_search_evaluation_feedback():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nfirst query\n</searches>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=2,
                min_total_results=2,
                min_top_score=0.8,
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body', score=0.5)],
            ],
        }
    )

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<search_evaluation>" in second_prompt
    assert "Verdict: INSUFFICIENT" in second_prompt
    assert "keep searching" in second_prompt


def test_search_agent_loop_closes_search_client_after_run():
    tokenizer = DummyTokenizerWithEncode()
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager([tokenizer.encode("<answer>Done</answer>")]),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            require_sufficient_evidence_before_answer=False,
        ),
    )
    fake_client = FakeSearchClient({})
    loop._search_client = fake_client

    asyncio.run(
        loop.run([{"role": "user", "content": "answer directly"}], {"temperature": 0.0})
    )

    assert fake_client.closed is True


def test_search_agent_loop_rejects_answer_before_any_search():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<answer>Done too early</answer>"),
        tokenizer.encode("<searches>\nfirst query\n</searches>"),
        tokenizer.encode("<answer>Done after search</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<answer_feedback>" in second_prompt
    # Targeted feedback: no search has happened yet, so the builder emits the
    # "Search first" message rather than the generic controller constant.
    assert "Search first" in second_prompt
    assert output.num_turns == 3
    assert output.context.num_rounds == 1


def test_search_agent_loop_rejects_answer_when_latest_evidence_is_insufficient():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nfirst query\n</searches>"),
        tokenizer.encode("<answer>Done too early</answer>"),
        tokenizer.encode("<searches>\nrefined query\n</searches>"),
        tokenizer.encode("<answer>Done after refinement</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=2,
                min_total_results=2,
                min_top_score=0.8,
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body', score=0.5)],
            ],
            ("refined query",): [
                [
                    SearchResult(contents='"Doc B"\nBeta body', score=0.95),
                    SearchResult(contents='"Doc C"\nGamma body', score=0.85),
                ],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert "<answer_feedback>" in third_prompt
    # Targeted feedback: a search happened but evidence was insufficient, so the
    # builder emits the "latest search evaluation was insufficient" message.
    assert "latest search evaluation was insufficient" in third_prompt
    assert output.num_turns == 4
    assert output.context.num_rounds == 2


def test_search_agent_loop_handles_plan_and_searches_in_same_response():
    """When a model emits <plan> and <searches> in a single response, both are
    processed in one turn — no wasted round-trip for the plan acknowledgement."""
    tokenizer = DummyTokenizerWithEncode()
    combined = "<plan>Quick plan.</plan><searches>\nalpha\nbeta\n</searches>"
    responses = [
        tokenizer.encode(combined),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(max_turns=4),
    )
    loop._search_client = FakeSearchClient(
        {
            ("alpha", "beta"): [
                [SearchResult(contents='"A"\nbody a')],
                [SearchResult(contents='"B"\nbody b')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "go"}], {"temperature": 0.0})
    )

    assert loop._search_client.calls == [["alpha", "beta"]]
    assert output.context.num_rounds == 1
    assert output.context.num_searches == 2
    assert output.num_turns == 2


def test_search_agent_loop_can_fetch_full_page_content():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<fetch>https://example.com/a</fetch>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(
                contents="Full page body", title="Doc A", url="https://example.com/a"
            ),
        ],
    }
    loop._search_client = fake_client

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert fake_client.fetch_calls == [["https://example.com/a"]]
    assert "<full_page>" in third_prompt
    assert "Full page body" in third_prompt


def test_search_agent_loop_deduplicates_queries_and_urls_and_tracks_metrics():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nalpha\nalpha\n</searches>"),
        tokenizer.encode("<fetch>https://example.com/a, https://example.com/a</fetch>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(
                contents="Full page body", title="Doc A", url="https://example.com/a"
            ),
        ],
    }
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert fake_client.calls == [["alpha"]]
    assert fake_client.fetch_calls == [["https://example.com/a"]]
    assert output.metrics["search_rounds"] == 1.0
    assert output.metrics["search_queries"] == 1.0
    assert output.metrics["fetched_pages"] == 1.0
    assert len(output.context.fetched_pages) == 1
    assert output.metrics["useful_fetched_pages"] == 1.0
    assert output.metrics["unnecessary_fetch_count"] == 0.0


def test_search_agent_loop_deduplicates_duplicate_evidence_across_queries():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nalpha\nbeta\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q1D1] [R1Q2D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=2
            ),
        ),
    )
    duplicate = SearchResult(
        contents='"Shared"\nDuplicate body',
        score=0.4,
        url="https://example.com/shared",
    )
    loop._search_client = FakeSearchClient(
        {
            ("alpha", "beta"): [
                [
                    duplicate,
                    SearchResult(
                        contents='"Shared"\nDuplicate body',
                        score=0.9,
                        url="https://example.com/shared",
                    ),
                    SearchResult(
                        contents='"Alpha"\nAlpha body',
                        url="https://example.com/alpha",
                    ),
                ],
                [
                    SearchResult(
                        contents='"Shared"\nDuplicate body',
                        url="https://example.com/shared#section",
                    ),
                    SearchResult(
                        contents='"Beta"\nBeta body',
                        url="https://example.com/beta",
                    ),
                ],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    first_query_results = output.context.rounds[0][0].results
    second_query_results = output.context.rounds[0][1].results
    assert [result.url for result in first_query_results] == [
        "https://example.com/shared",
        "https://example.com/alpha",
    ]
    assert first_query_results[0].score == 0.9
    assert [result.url for result in second_query_results] == [
        "https://example.com/beta"
    ]
    assert output.metrics["duplicate_search_results_removed"] == 2.0
    assert output.metrics["citation_count"] == 2.0
    assert output.metrics["cited_search_contexts"] == 2.0


def test_search_agent_loop_registers_subquestions_and_tracks_task_searches():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<subquestions>\nT1: identify the voice actor\nT2: identify the developer\n</subquestions>"
            "<searches>\n[T1] Alice David Lara Croft voice\n[T2] Lara Croft game developer\n</searches>"
        ),
        tokenizer.encode("<answer>Done [R1Q1D1] [R1Q2D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=2
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("Alice David Lara Croft voice", "Lara Croft game developer"): [
                [
                    SearchResult(
                        contents='"Voice"\nAlice David', url="https://example.com/voice"
                    )
                ],
                [
                    SearchResult(
                        contents='"Developer"\nCrystal Dynamics',
                        url="https://example.com/dev",
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.context.tasks == {
        "T1": "identify the voice actor",
        "T2": "identify the developer",
    }
    assert output.context.turns[0].task_id == "T1"
    assert output.context.turns[1].task_id == "T2"
    assert output.metrics["active_subquestions"] == 2.0
    assert output.metrics["subquestions_covered"] == 2.0
    assert output.metrics["subquestion_coverage_ratio"] == 1.0
    assert output.metrics["citation_count"] == 2.0
    assert output.metrics["cited_task_coverage_ratio"] == 1.0


def test_search_agent_loop_auto_registers_task_prefixed_research_tracks():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<searches>\n"
            "[T1] Alice David Lara Croft voice\n"
            "[T2] Lara Croft game developer\n"
            "</searches>"
        ),
        tokenizer.encode("<answer>Done [R1Q1D1] [R1Q2D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=2
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("Alice David Lara Croft voice", "Lara Croft game developer"): [
                [SearchResult(contents='"Voice"\nAlice David')],
                [SearchResult(contents='"Developer"\nCrystal Dynamics')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.context.tasks == {
        "T1": "Alice David Lara Croft voice",
        "T2": "Lara Croft game developer",
    }
    assert output.context.turns[0].task_id == "T1"
    assert output.context.turns[1].task_id == "T2"
    assert output.metrics["implicit_subquestions"] == 2.0
    assert output.metrics["active_subquestions"] == 2.0
    assert output.metrics["subquestion_coverage_ratio"] == 1.0


def test_search_agent_loop_tracks_research_followup_queries():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\n[T1] company funding\n</searches>"),
        tokenizer.encode("<searches>\n[T1] company funding 2024 amount\n</searches>"),
        tokenizer.encode("<answer>Done [R2Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=2, min_total_results=2
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("company funding",): [
                [SearchResult(contents='"Funding"\nSeed round')],
            ],
            ("company funding 2024 amount",): [
                [
                    SearchResult(contents='"Funding 2024"\nSeries A was $10M.'),
                    SearchResult(contents='"Investor"\nLead investor confirmed.'),
                ],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "T1: company funding (searches: 1)" in second_prompt
    assert output.metrics["research_followup_queries"] == 1.0
    assert output.metrics["research_tasks_with_followup"] == 1.0
    assert output.metrics["subquestion_coverage_ratio"] == 1.0


def test_agent_context_reports_cited_task_coverage():
    ctx = SearchContext(
        query="voice actor",
        task_id="T1",
        task_description="identify the voice actor",
        results=[
            SearchResult(contents='"Voice"\nAlice David'),
            SearchResult(contents='"Bio"\nBiography'),
        ],
    )
    agent_ctx = AgentContext(tasks={"T1": "identify the voice actor"})
    agent_ctx.rounds.append([ctx])
    agent_ctx.turns.append(ctx)

    assert agent_ctx.cited_result_ids("Use the voice source [R1Q1D2]") == frozenset(
        {"R1Q1D2"}
    )
    assert agent_ctx.cited_search_contexts("Use the voice source [R1Q1D2]") == [ctx]
    assert agent_ctx.cited_task_ids("Use the voice source [R1Q1D2]") == frozenset(
        {"T1"}
    )


def test_search_agent_loop_rejects_answer_when_a_subquestion_is_unresolved():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<subquestions>\nT1: identify the voice actor\nT2: identify the developer\n</subquestions>"
            "<searches>\n[T1] Alice David Lara Croft voice\n</searches>"
        ),
        tokenizer.encode("<answer>Done too early</answer>"),
        tokenizer.encode("<searches>\n[T2] Lara Croft game developer\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q1D1] [R2Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("Alice David Lara Croft voice",): [
                [
                    SearchResult(
                        contents='"Voice"\nAlice David', url="https://example.com/voice"
                    )
                ],
            ],
            ("Lara Croft game developer",): [
                [
                    SearchResult(
                        contents='"Developer"\nCrystal Dynamics',
                        url="https://example.com/dev",
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert "T2: identify the developer" in third_prompt
    assert "<answer_feedback>" in third_prompt
    assert output.context.num_rounds == 2


def test_search_agent_loop_reports_subquestion_coverage_feedback():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<subquestions>\nT1: find founding year\nT2: find headquarters\n</subquestions>"
            "<searches>\n[T1] company founding year\n</searches>"
        ),
        tokenizer.encode("<answer>Too early</answer>"),
        tokenizer.encode("<searches>\n[T2] company headquarters\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q1D1] [R2Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=6,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("company founding year",): [
                [SearchResult(contents='"Founded"\n1999')],
            ],
            ("company headquarters",): [
                [SearchResult(contents='"HQ"\nNew York')],
            ],
        }
    )

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<subquestions_feedback>" in second_prompt
    assert "Covered:" in second_prompt
    assert "Needs more evidence:" in second_prompt


def test_search_agent_loop_skips_repeated_queries_with_feedback():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert fake_client.calls == [["alpha query"]]
    assert "Repeated search skipped" in third_prompt
    assert output.metrics["repeated_search_queries"] == 1.0


def test_search_agent_loop_skips_repeated_queries_after_whitespace_normalization():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha   query</search>"),
        tokenizer.encode("<search> alpha query </search>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha   query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert fake_client.calls == [["alpha   query"]]
    assert "Repeated search skipped" in third_prompt
    assert output.metrics["repeated_search_queries"] == 1.0


def test_search_agent_loop_enforces_search_limit():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<search>beta query</search>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            max_search_limit=1,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    third_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[2]["prompt_ids"]
    )
    assert fake_client.calls == [["alpha query"]]
    assert "Search limit reached" in third_prompt
    assert output.metrics["search_limit_hits"] == 1.0


def test_search_agent_loop_tracks_budget_exhausted_without_answer():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<search>beta query</search>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            max_search_limit=1,
            force_answer_on_deadend=False,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.final_answer is None
    assert output.metrics["search_budget_exhausted_without_answer"] == 1.0


def test_search_agent_loop_records_answered_exit_metric():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=3,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("alpha query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.metrics["exit_answered"] == 1.0
    assert output.metrics["exit_max_turns"] == 0.0
    assert [
        (event.component, event.action, event.status)
        for event in output.control_flow_trace
    ] == [
        ("planner", "search_planned", "decided"),
        ("search_tool", "vector_db_search", "completed"),
        ("evidence_judge", "evidence_evaluated", "completed"),
        ("loop_controller", "search_continued", "decided"),
        ("planner", "answer_planned", "decided"),
        ("loop_controller", "answer_accepted", "decided"),
        ("answer_generator", "citations_resolved", "completed"),
    ]


def test_search_agent_loop_records_max_turns_exit_metric():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<think>planning</think>"),
        tokenizer.encode("<think>still planning</think>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(max_turns=2),
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.final_answer is None
    assert output.metrics["exit_max_turns"] == 1.0
    assert output.metrics["exit_answered"] == 0.0


def test_search_agent_loop_stops_after_repeated_no_action_turns():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("plain text"),
        tokenizer.encode("still plain text"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            max_consecutive_format_errors=2,
            # Exercise the legacy format-error stop path, not the auto-search.
            auto_search_on_deadend=False,
        ),
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.num_turns == 2
    assert output.metrics["format_error_turns"] == 2.0
    assert output.metrics["exit_format_error_limit"] == 1.0
    assert output.metrics["exit_no_action"] == 1.0
    assert [(event.component, event.action) for event in output.control_flow_trace] == [
        ("planner", "format_recovery"),
        ("planner", "format_recovery"),
    ]


def test_search_agent_loop_allows_direct_answer_before_search_when_enabled():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<search_decision>answer</search_decision><answer>Paris</answer>"
        ),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=3,
            allow_internal_knowledge_answer=True,
        ),
    )

    output = asyncio.run(
        loop.run(
            [{"role": "user", "content": "What is the capital of France?"}],
            {"temperature": 0.0},
        )
    )

    assert output.num_turns == 1
    assert output.context.num_rounds == 0
    assert output.metrics["direct_answers"] == 1.0
    assert output.metrics["answer_allowed"] == 1.0


def test_search_agent_loop_requests_search_after_search_decision():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<search_decision>search</search_decision>"),
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )
    loop._search_client = fake_client

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<decision_feedback>" in second_prompt
    assert "Issue a <search> or <searches> action next" in second_prompt


def test_search_agent_loop_prompts_for_decision_when_no_action_before_search():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("I am thinking but have not decided yet."),
        tokenizer.encode("<search_decision>search</search_decision>"),
        tokenizer.encode("<search>alpha query</search>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [SearchResult(contents='"Doc A"\nAlpha body')],
            ],
        }
    )
    loop._search_client = fake_client

    asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    second_prompt = "".join(
        chr(token) for token in loop.server_manager.calls[1]["prompt_ids"]
    )
    assert "<answer_feedback>" in second_prompt
    assert "Use <search_decision>answer</search_decision>" in second_prompt


def test_search_agent_loop_emits_search_quality_metrics():
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nweak query\nstrong query\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q2D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1,
                min_total_results=2,
                min_top_score=0.8,
                require_scores=True,
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("weak query", "strong query"): [
                [SearchResult(contents='"Weak"\nbody', score=0.2)],
                [
                    SearchResult(contents='"Strong"\nbody', score=0.95),
                    SearchResult(contents='"Strong 2"\nbody', score=0.9),
                ],
            ],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.metrics["search_quality_score"] == pytest.approx(0.5, abs=0.001)
    assert output.metrics["evidence_insufficient_rounds"] == 1.0
    assert output.metrics["final_evidence_sufficient"] == 0.0
    assert output.metrics["answer_when_evidence_insufficient"] == 1.0


def test_search_agent_loop_blocks_direct_answer_when_internal_knowledge_disabled():
    """allow_internal_knowledge_answer=False prevents bypassing the search gate."""
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode(
            "<search_decision>answer</search_decision><answer>Paris</answer>"
        ),
        tokenizer.encode("<searches>\nalpha\n</searches>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            allow_internal_knowledge_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {("alpha",): [[SearchResult(contents='"Doc A"\nAlpha body')]]}
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q"}], {"temperature": 0.0})
    )

    assert output.metrics["direct_answers"] == 0.0
    assert output.context.num_rounds == 1


def test_search_agent_loop_search_decision_with_searches_fires_search_not_decision_feedback():
    """<search_decision>search</search_decision> alongside <searches> should execute
    the search without injecting a decision_feedback observation."""
    tokenizer = DummyTokenizerWithEncode()
    combined = "<search_decision>search</search_decision><searches>\nalpha\n</searches>"
    responses = [
        tokenizer.encode(combined),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {("alpha",): [[SearchResult(contents='"Doc A"\nAlpha body')]]}
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "go"}], {"temperature": 0.0})
    )

    second_prompt = "".join(chr(t) for t in loop.server_manager.calls[1]["prompt_ids"])
    assert "<decision_feedback>" not in second_prompt
    assert "<information>" in second_prompt
    assert output.context.num_rounds == 1
    assert output.num_turns == 2


def test_search_client_config_derives_fetch_url_from_retrieve_url():
    from src import SearchClientConfig

    cases = [
        ("http://localhost:8000/retrieve", "http://localhost:8000/fetch"),
        ("http://localhost:8000/retrieve/", "http://localhost:8000/fetch"),
        ("http://host:9000/api/retrieve", "http://host:9000/api/fetch"),
        ("http://host/other", "http://host/other/fetch"),
    ]
    for url, expected in cases:
        assert SearchClientConfig(url=url).get_fetch_url() == expected, (
            f"Failed for {url!r}"
        )


def test_search_agent_loop_processes_search_and_fetch_in_same_turn():
    """When the model emits <searches> and <fetch> in the same turn, both are
    executed and their results appear in a single observation message."""
    tokenizer = DummyTokenizerWithEncode()
    combined = (
        "<searches>\nalpha query\n</searches><fetch>https://example.com/a</fetch>"
    )
    responses = [
        tokenizer.encode(combined),
        tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    fake_client = FakeSearchClient(
        {
            ("alpha query",): [
                [
                    SearchResult(
                        contents='"Doc A"\nAlpha body', url="https://example.com/a"
                    )
                ]
            ]
        }
    )
    fake_client.fetch_responses = {
        ("https://example.com/a",): [
            SearchResult(
                contents="Full page body", title="Doc A", url="https://example.com/a"
            )
        ]
    }
    loop._search_client = fake_client

    output = asyncio.run(
        loop.run([{"role": "user", "content": "go"}], {"temperature": 0.0})
    )

    # Both search and fetch fired in turn 0.
    assert fake_client.calls == [["alpha query"]]
    assert fake_client.fetch_calls == [["https://example.com/a"]]
    # Both observations are in the same injected user message (turn 1 prompt).
    second_prompt = "".join(chr(t) for t in loop.server_manager.calls[1]["prompt_ids"])
    assert "<information>" in second_prompt
    assert "<full_page>" in second_prompt
    assert output.num_turns == 2


# ---------------------------------------------------------------------------
# generate_rollout_step — explicit RL step: state → action → terminal/continue
# ---------------------------------------------------------------------------


def test_generate_rollout_step_classifies_search_action():
    tokenizer = DummyTokenizerWithEncode()
    loop = ConcreteAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            tokenizer.encode("<search>who invented radar</search>")
        ),
        config=AgentLoopConfig(prompt_length=16, response_length=64),
    )
    step = asyncio.run(
        loop.generate_rollout_step(
            prompt_ids=[1, 2, 3],
            sampling_params={"temperature": 0.7},
        )
    )
    assert isinstance(step, RolloutStep)
    assert step.action_type == "search"
    assert step.action_content == "who invented radar"
    assert step.is_terminal is False
    assert step.prompt_ids == [1, 2, 3]
    assert step.response_mask == [1] * len(step.response_ids)


def test_generate_rollout_step_classifies_answer_as_terminal():
    tokenizer = DummyTokenizerWithEncode()
    loop = ConcreteAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(
            tokenizer.encode("<answer>Watson and Watt</answer>")
        ),
        config=AgentLoopConfig(prompt_length=16, response_length=64),
    )
    step = asyncio.run(
        loop.generate_rollout_step(
            prompt_ids=[4, 5],
            sampling_params={"temperature": 0.0},
        )
    )
    assert step.action_type == "answer"
    assert step.action_content == "Watson and Watt"
    assert step.is_terminal is True


def test_generate_rollout_step_marks_no_action_as_terminal():
    tokenizer = DummyTokenizerWithEncode()
    loop = ConcreteAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(tokenizer.encode("I have no idea.")),
        config=AgentLoopConfig(prompt_length=16, response_length=64),
    )
    step = asyncio.run(
        loop.generate_rollout_step(
            prompt_ids=[1],
            sampling_params={"temperature": 0.5},
        )
    )
    assert step.action_type is None
    assert step.action_content == ""
    assert step.is_terminal is True


def test_generate_rollout_step_accepts_custom_action_re_and_terminal_actions():
    import re

    tokenizer = DummyTokenizerWithEncode()
    loop = ConcreteAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(tokenizer.encode("<tool>calculator</tool>")),
        config=AgentLoopConfig(prompt_length=16, response_length=64),
    )
    step = asyncio.run(
        loop.generate_rollout_step(
            prompt_ids=[7, 8],
            sampling_params={"temperature": 0.0},
            action_re=re.compile(r"<(tool)>(.*?)</\1>", re.DOTALL),
            terminal_actions=frozenset({"tool"}),
        )
    )
    assert step.action_type == "tool"
    assert step.action_content == "calculator"
    assert step.is_terminal is True


def test_search_agent_loop_routes_round_to_web_retriever():
    """`<search retriever="web">` sends the round to the web client, not the VDB."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search retriever="web">latest ai news</search>'),
            tokenizer.encode("<answer>Done [R1Q1D1]</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            web_search_url="http://web",
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    vdb = FakeSearchClient({})
    web = FakeSearchClient(
        {("latest ai news",): [[SearchResult(contents='"News"\nbig story today')]]}
    )
    loop._search_client = vdb
    loop._web_search_client = web

    output = asyncio.run(
        loop.run([{"role": "user", "content": "news?"}], {"temperature": 0.0})
    )

    assert web.calls == [["latest ai news"]]
    assert vdb.calls == []
    assert output.context.num_rounds == 1


def test_search_agent_loop_web_request_degrades_to_vdb_when_unconfigured():
    """A web request with no web client configured falls back to the VDB client."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search retriever="web">q</search>'),
            tokenizer.encode("<answer>a [R1Q1D1]</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {("q",): [[SearchResult(contents='"T"\nbody content here')]]}
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q?"}], {"temperature": 0.0})
    )

    assert loop._search_client.calls == [["q"]]
    assert output.context.num_rounds == 1


def test_search_agent_loop_surfaces_action_metrics_for_reward():
    """web/vdb counts and evidence_score/gain are surfaced for the Phase C reward."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search retriever="web">q</search>'),
            tokenizer.encode("<answer>done [R1Q1D1]</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            web_search_url="http://web",
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient({})
    loop._web_search_client = FakeSearchClient(
        {("q",): [[SearchResult(contents='"T"\nstrong body content', score=5.0)]]}
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q?"}], {"temperature": 0.0})
    )

    assert output.metrics["web_searches"] == 1.0
    assert output.metrics["vdb_searches"] == 0.0
    assert output.metrics["evidence_score_final"] > 0.0
    assert output.metrics["evidence_gain_total"] > 0.0


def test_search_agent_loop_reranks_round_when_requested():
    """`<search rerank="true">` reorders the round's results and counts a rerank call."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search rerank="true">q</search>'),
            tokenizer.encode("<answer>done [R1Q1D1]</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("q",): [
                [
                    SearchResult(contents="low relevance body", score=0.1, title="lo"),
                    SearchResult(contents="high relevance body", score=0.9, title="hi"),
                ]
            ]
        }
    )
    # Inject a simple reranker: sort by score descending.
    loop._reranker = lambda query, docs: sorted(
        docs, key=lambda d: d.score, reverse=True
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q?"}], {"temperature": 0.0})
    )

    assert output.metrics["rerank_calls"] == 1.0
    # The reranked (high-first) order is what landed in the round.
    first_round = output.context.rounds[0][0]
    assert [r.title for r in first_round.results] == ["hi", "lo"]


def test_search_agent_loop_rerank_request_is_noop_without_reranker():
    """rerank requested but none configured: no crash, no rerank counted."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search rerank="true">q</search>'),
            tokenizer.encode("<answer>a [R1Q1D1]</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {("q",): [[SearchResult(contents="body content here", score=0.5)]]}
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q?"}], {"temperature": 0.0})
    )

    assert output.metrics["rerank_calls"] == 0.0


def test_search_agent_loop_skips_rerank_for_single_result_round():
    """A one-document round cannot benefit from rerank, so the reranker is not called."""
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search rerank="true">q</search>'),
            tokenizer.encode("<answer>a [R1Q1D1]</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=4,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {("q",): [[SearchResult(contents="single body content", score=0.5)]]}
    )
    calls: list[str] = []

    def reranker(query: str, docs: list[SearchResult]) -> list[SearchResult]:
        calls.append(query)
        return docs

    loop._reranker = reranker

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q?"}], {"temperature": 0.0})
    )

    assert calls == []
    assert output.metrics["rerank_requested"] == 1.0
    assert output.metrics["rerank_calls"] == 0.0
    assert output.metrics["rerank_skipped"] == 1.0


def test_search_agent_loop_counts_empty_rerank_request_as_skipped():
    tokenizer = DummyTokenizerWithEncode()
    server_manager = DummyServerManager(
        [
            tokenizer.encode('<search rerank="true">q</search>'),
            tokenizer.encode("<answer>a</answer>"),
        ]
    )
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=server_manager,
        search_config=SearchAgentLoopConfig(
            max_turns=3,
            require_sufficient_evidence_before_answer=False,
        ),
    )
    loop._search_client = FakeSearchClient({("q",): [[]]})
    calls: list[str] = []
    loop._reranker = lambda query, docs: calls.append(query) or docs

    output = asyncio.run(
        loop.run([{"role": "user", "content": "q?"}], {"temperature": 0.0})
    )

    assert calls == []
    assert output.metrics["rerank_requested"] == 1.0
    assert output.metrics["rerank_calls"] == 0.0
    assert output.metrics["rerank_skipped"] == 1.0


def test_loop_controller_config_defaults():
    cfg = SearchAgentLoopConfig()
    assert cfg.evidence_plateau_min_gain == 0.05
    assert cfg.plateau_requires_sufficient is True
    assert cfg.search_budget_per_subquestion == 1
    assert cfg.max_search_limit_cap == 10
    assert cfg.force_answer_on_deadend is True


def test_adaptive_budget_raises_limit_for_multiple_subquestions():
    """With 3 subquestions and budget_per_subquestion=1: effective_limit = base 2 + (3-1) = 4."""
    tokenizer = DummyTokenizerWithEncode()
    # Script: declare 3 subquestions + first search in same turn, then 3 more searches, then answer.
    responses = [
        tokenizer.encode(
            "<subquestions>\nT1: first aspect\nT2: second aspect\nT3: third aspect\n</subquestions>"
            "<searches>\n[T1] first query\n</searches>"
        ),
        tokenizer.encode("<searches>\n[T2] second query\n</searches>"),
        tokenizer.encode("<searches>\n[T3] third query\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q1D1] [R2Q1D1] [R3Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=8,
            max_search_limit=2,
            search_budget_per_subquestion=1,
            max_search_limit_cap=10,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [[SearchResult(contents='"Doc A"\nAlpha body')]],
            ("second query",): [[SearchResult(contents='"Doc B"\nBeta body')]],
            ("third query",): [[SearchResult(contents='"Doc C"\nGamma body')]],
        }
    )

    output = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert output.metrics["effective_search_limit"] == 4.0  # base 2 + (3-1)*1
    assert output.metrics["adaptive_budget_bonus"] == 2.0


def test_deadend_forces_answer_from_evidence():
    """Dead-end after one good search round: forced-answer is emitted from evidence."""
    tokenizer = DummyTokenizerWithEncode()
    # Turn 1: search round (evidence collected)
    # Turn 2: no recognised action → dead-end (no_action exit)
    # Forced turn: model produces <answer> tag
    responses = [
        tokenizer.encode("<search>what is FAISS</search>"),
        tokenizer.encode("I have no idea what to do next."),
        # forced-answer turn
        tokenizer.encode(
            "<answer>FAISS is a library for similarity search [R1Q1D1]</answer>"
        ),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            max_search_limit=5,
            force_answer_on_deadend=True,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("what is FAISS",): [
                [
                    SearchResult(
                        contents='"FAISS"\nFacebook AI Similarity Search', score=0.9
                    )
                ],
            ],
        }
    )

    out = asyncio.run(
        loop.run([{"role": "user", "content": "What is FAISS?"}], {"temperature": 0.0})
    )

    assert out.final_answer is not None
    assert out.metrics["forced_final_answer"] == 1.0
    assert out.metrics["search_budget_exhausted_without_answer"] == 0.0
    assert out.metrics["answer_when_evidence_insufficient"] == 0.0


def test_deadend_with_no_evidence_does_not_fabricate():
    """Dead-end with no search rounds: forced-answer opt-out (never fabricate)."""
    tokenizer = DummyTokenizerWithEncode()
    # Immediate format errors, no search → agent_ctx.num_rounds == 0
    responses = [
        tokenizer.encode("plain text no tags"),
        tokenizer.encode("still no tags"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            max_consecutive_format_errors=2,
            force_answer_on_deadend=True,
            # Validate the no-evidence dead-end opt-out, not the auto-search.
            auto_search_on_deadend=False,
        ),
    )
    loop._search_client = FakeSearchClient({})

    out = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert out.metrics["forced_final_answer"] == 0.0


def test_budget_exhausted_forces_answer():
    """When max_turns exhausts without an answer, _force_final_answer is called.

    Script:
    - Turn 1: search (evidence collected)
    - Turn 2: another search (search limit reached after this)
    - Loop exits via max_turns with final_answer=None
    - Post-loop hook calls _force_final_answer → model emits <answer>

    Asserts:
    - out.final_answer is not None
    - forced_final_answer == 1.0
    - search_budget_exhausted_without_answer == 0.0
    """
    tokenizer = DummyTokenizerWithEncode()
    # Turn 1: search round (evidence collected)
    # Turn 2: search round (hits max_search_limit=1; loop observes limit but continues)
    # Turn 3 (max_turns=2, so loop exits after turn 2 without getting here)
    # Forced turn: model produces <answer>
    responses = [
        tokenizer.encode("<searches>\nwhat is FAISS\n</searches>"),
        tokenizer.encode("<searches>\nmore about FAISS\n</searches>"),
        # forced-answer turn (post-loop hook)
        tokenizer.encode(
            "<answer>FAISS is a library for similarity search [R1Q1D1]</answer>"
        ),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=2,
            max_search_limit=2,
            force_answer_on_deadend=True,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("what is FAISS",): [
                [
                    SearchResult(
                        contents='"FAISS"\nFacebook AI Similarity Search', score=0.9
                    )
                ],
            ],
            ("more about FAISS",): [
                [
                    SearchResult(
                        contents='"FAISS index"\nEfficient similarity indexing',
                        score=0.8,
                    )
                ],
            ],
        }
    )

    out = asyncio.run(
        loop.run([{"role": "user", "content": "What is FAISS?"}], {"temperature": 0.0})
    )

    assert out.final_answer is not None
    assert out.metrics["forced_final_answer"] == 1.0
    assert out.metrics["search_budget_exhausted_without_answer"] == 0.0


def test_plateau_stops_searching_when_sufficient():
    """Plateau early-stop fires when evidence is sufficient and gain stalls.

    Script: round 1 yields sufficient evidence; round 2 returns the same
    score (gain = 0 < 0.05) and evidence is still sufficient → PLATEAU fires
    after round 2, appending the search-limit observation and skipping further
    searches. The model answers on the next turn.
    """
    tokenizer = DummyTokenizerWithEncode()
    # responses: search round 1, search round 2, answer
    responses = [
        tokenizer.encode("<searches>\nfirst query\n</searches>"),
        tokenizer.encode("<searches>\nsecond query\n</searches>"),
        tokenizer.encode("<answer>Done</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=8,
            max_search_limit=5,
            evidence_plateau_min_gain=0.05,
            plateau_requires_sufficient=True,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1,
                min_total_results=1,
                min_top_score=0.5,
            ),
        ),
    )
    # Both rounds return identically-scored sufficient docs → round 2 gain = 0.
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [
                [SearchResult(contents='"Doc A"\nAlpha body', score=0.9)]
            ],
            ("second query",): [
                [SearchResult(contents='"Doc B"\nBeta body', score=0.9)]
            ],
        }
    )

    out = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    assert out.metrics["plateau_early_stop"] == 1.0
    assert out.metrics["rounds_used"] < 5.0


def test_answer_gate_forces_after_max_rejections():
    """After max_answer_rejections rejections the gate should FORCE-accept the answer
    and set metrics["forced_final_answer"] == 1.0.

    Script:
    - Turn 1: search round (one weak result — insufficient evidence)
    - Turn 2: answer emitted with insufficient evidence → rejected (rejection 1)
    - Turn 3: answer emitted again (no new search) → rejected (rejection 2)
    - Turn 4: answer emitted again → consecutive_rejections == max_answer_rejections (2)
              → FORCE path: accept answer, set forced_final_answer=1.0
    """
    tokenizer = DummyTokenizerWithEncode()
    responses = [
        tokenizer.encode("<searches>\nwhat is FAISS\n</searches>"),
        tokenizer.encode("<answer>FAISS is a library [R1Q1D1]</answer>"),
        tokenizer.encode("<answer>FAISS is a library [R1Q1D1]</answer>"),
        tokenizer.encode("<answer>FAISS is a library [R1Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=8,
            max_answer_rejections=2,
            require_sufficient_evidence_before_answer=True,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=2,
                min_total_results=2,
                min_top_score=0.8,
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("what is FAISS",): [
                # Only one weak result — fails min_results_per_query=2 and min_top_score=0.8
                [
                    SearchResult(
                        contents='"FAISS"\nFacebook AI Similarity Search', score=0.5
                    )
                ],
            ],
        }
    )

    out = asyncio.run(
        loop.run([{"role": "user", "content": "What is FAISS?"}], {"temperature": 0.0})
    )

    assert out.final_answer is not None  # forced through after cap
    assert out.metrics["forced_final_answer"] == 1.0


def test_budget_metric_uses_effective_limit_for_multi_subquestions():
    """With 3 subquestions and base=2 the effective limit is 4.
    A run that searches 4 rounds (exhausting the effective limit) then emits
    an answer must NOT fire search_budget_exhausted_without_answer.
    """
    tokenizer = DummyTokenizerWithEncode()
    # Script: declare 3 subquestions + first search in turn 1, then 3 more searches,
    # then answer — 4 rounds total (effective limit with base=2, 3 subquestions, per_sub=1).
    responses = [
        tokenizer.encode(
            "<subquestions>\nT1: first aspect\nT2: second aspect\nT3: third aspect\n</subquestions>"
            "<searches>\n[T1] first query\n</searches>"
        ),
        tokenizer.encode("<searches>\n[T2] second query\n</searches>"),
        tokenizer.encode("<searches>\n[T3] third query\n</searches>"),
        tokenizer.encode("<searches>\n[T1] refined query\n</searches>"),
        tokenizer.encode("<answer>Done [R1Q1D1] [R2Q1D1] [R3Q1D1]</answer>"),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=10,
            max_search_limit=2,
            search_budget_per_subquestion=1,
            max_search_limit_cap=10,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("first query",): [[SearchResult(contents='"Doc A"\nAlpha body')]],
            ("second query",): [[SearchResult(contents='"Doc B"\nBeta body')]],
            ("third query",): [[SearchResult(contents='"Doc C"\nGamma body')]],
            ("refined query",): [[SearchResult(contents='"Doc D"\nDelta body')]],
        }
    )

    out = asyncio.run(
        loop.run([{"role": "user", "content": "research this"}], {"temperature": 0.0})
    )

    # effective limit = 2 + (3-1)*1 = 4; 4 rounds used; answer produced
    assert out.metrics["effective_search_limit"] == 4.0
    assert out.metrics["rounds_used"] == 4.0
    assert out.final_answer is not None
    # Must NOT fire even though rounds_used == base limit (2) — effective limit was 4
    assert out.metrics["search_budget_exhausted_without_answer"] == 0.0


def test_forced_turn_emitting_no_answer_returns_none():
    """When the forced final-answer turn itself emits no <answer> tag and there
    is no prior tentative answer, final_answer is None and forced_final_answer == 0.0.
    """
    tokenizer = DummyTokenizerWithEncode()
    # Turn 1: one good search round (evidence collected)
    # Turn 2: no recognised action → dead-end (no_action exit)
    # Forced turn: model responds with plain text, NO <answer> tag
    responses = [
        tokenizer.encode("<searches>\nwhat is FAISS\n</searches>"),
        tokenizer.encode("I give up."),
        # forced-answer turn — intentionally no <answer> tag
        tokenizer.encode("I still cannot provide an answer."),
    ]
    loop = SearchAgentLoop(
        tokenizer=tokenizer,
        server_manager=DummyServerManager(responses),
        search_config=SearchAgentLoopConfig(
            max_turns=5,
            max_search_limit=5,
            force_answer_on_deadend=True,
            require_sufficient_evidence_before_answer=False,
            evaluation_config=SearchEvaluationConfig(
                min_results_per_query=1, min_total_results=1
            ),
        ),
    )
    loop._search_client = FakeSearchClient(
        {
            ("what is FAISS",): [
                [
                    SearchResult(
                        contents='"FAISS"\nFacebook AI Similarity Search', score=0.9
                    )
                ]
            ],
        }
    )

    out = asyncio.run(
        loop.run([{"role": "user", "content": "What is FAISS?"}], {"temperature": 0.0})
    )

    assert out.final_answer is None
    assert out.metrics["forced_final_answer"] == 0.0


def test_finalize_run_metrics_computes_derived_keys():
    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(max_search_limit=3),
    )
    metrics = loop._initial_metrics()
    metrics["search_queries"] = 2.0
    metrics["repeated_search_queries"] = 0.0
    loop._finalize_run_metrics(
        metrics,
        rounds_used=0,
        task_statuses={},
        task_search_counts={},
        active_tasks={},
        agent_ctx=AgentContext(),
        final_answer=None,
        latest_evaluation=None,
        exit_status="answered",
    )
    # No subquestions → coverage ratio defaults to 1.0
    assert metrics["subquestion_coverage_ratio"] == 1.0
    # No answer → answer_allowed stays 0.0
    assert metrics["answer_allowed"] == 0.0
    # rounds_used surfaced as float
    assert metrics["rounds_used"] == 0.0
    # exit fixup recorded
    assert metrics["exit_answered"] == 1.0
    # seeded search_queries=2, repeated=0 → ratio 0.0 (no divide-by-zero)
    assert metrics["repeated_query_ratio"] == 0.0


def _gate_loop():
    from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig

    return SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(max_answer_rejections=3),
    )


def test_apply_answer_gate_rejects_insufficient_evidence():
    import asyncio
    from src.agents.search import TurnControl

    loop = _gate_loop()
    metrics = loop._initial_metrics()
    d = asyncio.run(
        loop._apply_answer_gate(
            on_turn=None,
            num_turns=1,
            rounds_used=1,
            active_tasks={},
            task_statuses={},
            latest_evaluation=None,
            latest_search_decision=None,
            consecutive_rejections=0,
            final_answer="draft",
            metrics=metrics,
            working_messages=[],
        )
    )
    assert d.control is TurnControl.CONTINUE
    assert d.final_answer is None
    assert d.consecutive_rejections == 1
    assert metrics["answer_rejections"] == 1.0


def test_apply_answer_gate_accepts_with_internal_knowledge():
    import asyncio
    from src.agents.search import TurnControl, SearchAgentLoopConfig, SearchAgentLoop

    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(allow_internal_knowledge_answer=True),
    )
    metrics = loop._initial_metrics()
    d = asyncio.run(
        loop._apply_answer_gate(
            on_turn=None,
            num_turns=1,
            rounds_used=0,
            active_tasks={},
            task_statuses={},
            latest_evaluation=None,
            latest_search_decision="answer",
            consecutive_rejections=0,
            final_answer="ans",
            metrics=metrics,
            working_messages=[],
        )
    )
    assert d.control is TurnControl.BREAK
    assert d.exit_status == "answered"
    assert metrics["direct_answers"] == 1.0


def test_handle_no_action_format_error_limit_breaks():
    from src.agents.search import (
        SearchAgentLoop,
        SearchAgentLoopConfig,
        TurnControl,
    )
    from src.context.search import AgentContext

    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(
            max_consecutive_format_errors=1, force_answer_on_deadend=False
        ),
    )
    metrics = loop._initial_metrics()
    d = asyncio.run(
        loop._handle_no_action(
            working_messages=[],
            agent_ctx=AgentContext(),
            request_id="r",
            sampling_params={},
            metrics=metrics,
            latest_evaluation=None,
            task_statuses={},
            active_tasks={},
            rounds_used=1,
            consecutive_format_errors=0,
            consecutive_rejections=0,
            forced_answer_attempted=False,
            final_answer=None,
            num_turns=1,
        )
    )
    assert d.control is TurnControl.BREAK
    assert d.exit_status == "format_error_limit"
    assert d.consecutive_format_errors == 1
    assert metrics["format_error_turns"] == 1.0


def test_handle_no_action_below_limit_reprompts_continue():
    from src.agents.search import (
        SearchAgentLoop,
        SearchAgentLoopConfig,
        TurnControl,
    )
    from src.context.search import AgentContext

    loop = SearchAgentLoop(
        tokenizer=DummyTokenizerWithEncode(),
        server_manager=DummyServerManager([]),
        search_config=SearchAgentLoopConfig(
            max_consecutive_format_errors=5,
            require_sufficient_evidence_before_answer=True,
            max_answer_rejections=3,
        ),
    )
    metrics = loop._initial_metrics()
    msgs = []
    d = asyncio.run(
        loop._handle_no_action(
            working_messages=msgs,
            agent_ctx=AgentContext(),
            request_id="r",
            sampling_params={},
            metrics=metrics,
            latest_evaluation=None,
            task_statuses={},
            active_tasks={},
            rounds_used=0,
            consecutive_format_errors=0,
            consecutive_rejections=0,
            forced_answer_attempted=False,
            final_answer=None,
            num_turns=1,
        )
    )
    assert d.control is TurnControl.CONTINUE
    assert d.consecutive_rejections == 1
    assert len(msgs) == 1  # a re-prompt was appended
