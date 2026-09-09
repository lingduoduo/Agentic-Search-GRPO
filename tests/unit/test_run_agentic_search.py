from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import AgentLoopBase
from examples.run_agentic_search import (
    IntentPrediction,
    _build_sampling_params,
    _has_accelerate,
    _friendly_model_load_error,
    _handle_local_cli_value_error,
    _load_intent_prediction,
    _parse_major_minor,
    _resolve_model_route,
    _resolve_local_device,
    _validate_local_runtime_device,
    _validate_local_runtime_stack,
    _validate_local_generation_config,
    run_single_turn,
)


class _DummyLoop(AgentLoopBase):
    async def run(self, messages, sampling_params):
        raise NotImplementedError


class _TokenizerWithoutChatTemplate:
    chat_template = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        raise AssertionError(
            "apply_chat_template should not be used when chat_template is missing"
        )

    def encode(self, text: str) -> list[int]:
        return [len(text)]


class _TokenizerWithIds:
    pad_token_id = None
    eos_token_id = 99


class _PlainGenerationTokenizer:
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(i) for i in ids)


class _RecordingServerManager:
    def __init__(self, response_text: str):
        self.response_ids = [ord(ch) for ch in response_text]
        self.calls: list[dict[str, object]] = []

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
            }
        )
        return self.response_ids


def test_build_prompt_ids_falls_back_to_encode_without_chat_template():
    loop = _DummyLoop(
        tokenizer=_TokenizerWithoutChatTemplate(),
        server_manager=object(),
    )
    prompt_ids = loop._build_prompt_ids_sync(
        [{"role": "user", "content": "What is FAISS?"}]
    )
    assert prompt_ids == [14]


def test_run_single_turn_is_plain_model_generation(capsys):
    tokenizer = _PlainGenerationTokenizer()
    server_manager = _RecordingServerManager("plain answer")

    asyncio.run(
        run_single_turn(
            tokenizer=tokenizer,
            server_manager=server_manager,
            question="What is FAISS?",
            sampling_params={"temperature": 0.0, "max_tokens": 16},
            max_tokens=16,
            search_url="http://should-not-be-used/retrieve",
            topk=99,
        )
    )

    captured = capsys.readouterr()
    assert "plain answer" in captured.out
    assert len(server_manager.calls) == 1
    assert server_manager.calls[0]["sampling_params"]["max_tokens"] == 16


def test_run_agentic_search_has_no_minisweagent_dependency():
    """Scan the whole CLI, not just its entry file.

    The parser and routing modules were split out of run_agentic_search.py; a
    check that reads only the entry file would stop covering them and let the
    dependency return through a submodule.
    """
    sources = [Path("examples/run_agentic_search.py")]
    sources += sorted(Path("examples/agentic_search").glob("*.py"))
    assert len(sources) >= 4, sources

    for path in sources:
        source = path.read_text()
        assert "minisweagent" not in source, path
        assert "class DefaultAgent" not in source, path


def test_validate_local_generation_config_rejects_encoder_only_model():
    config = SimpleNamespace(
        model_type="bert",
        is_encoder_decoder=False,
        architectures=["BertModel"],
    )

    with pytest.raises(
        ValueError, match="Local generation mode requires a generative language model"
    ):
        _validate_local_generation_config("BAAI/bge-base-en-v1.5", config)


def test_validate_local_generation_config_allows_causal_lm():
    config = SimpleNamespace(
        model_type="llama",
        is_encoder_decoder=False,
        architectures=["LlamaForCausalLM"],
    )

    _validate_local_generation_config("meta-llama/Llama-3.1-8B-Instruct", config)


def test_friendly_model_load_error_formats_gated_repo_failure():
    exc = OSError(
        "You are trying to access a gated repo. 401 Client Error. Cannot access gated repo"
    )
    message = _friendly_model_load_error("meta-llama/Llama-3.1-8B-Instruct", exc)
    assert message is not None
    assert "gated on Hugging Face" in message
    assert "huggingface-cli login" in message


def test_friendly_model_load_error_formats_missing_repo_failure():
    exc = OSError("404 Client Error. Repository Not Found for url")
    message = _friendly_model_load_error("missing/model", exc)
    assert message is not None
    assert "could not find that repo" in message


def test_friendly_model_load_error_formats_cache_only_miss():
    exc = OSError(
        "Couldn't connect to the Hub and cannot find the requested files in the disk cache"
    )
    message = _friendly_model_load_error("Qwen/Qwen2.5-1.5B-Instruct", exc)
    assert message is not None
    assert "local Hugging Face cache" in message
    assert "--allow_remote_model_downloads" in message


def test_build_sampling_params_uses_cli_namespace():
    args = SimpleNamespace(temperature=0.2, max_tokens=64, top_p=0.9)
    assert _build_sampling_params(args) == {
        "temperature": 0.2,
        "max_tokens": 64,
        "top_p": 0.9,
    }


def test_resolve_model_route_uses_base_model_when_disabled():
    args = SimpleNamespace(
        model_routing="off",
        model="base-model",
        fast_model="fast-model",
        balanced_model="balanced-model",
        reasoning_model="reasoning-model",
        model_routing_min_confidence=0.7,
    )

    decision = _resolve_model_route(
        args,
        IntentPrediction(intent="search", confidence=0.99),
    )

    assert decision.model == "base-model"
    assert decision.route == "base"
    assert decision.metadata["model_routing"] == "off"


def test_resolve_model_route_maps_intents_to_model_tiers():
    args = SimpleNamespace(
        model_routing="intent",
        model="base-model",
        fast_model="fast-model",
        balanced_model="balanced-model",
        reasoning_model="reasoning-model",
        model_routing_min_confidence=0.7,
    )

    assert (
        _resolve_model_route(
            args, IntentPrediction(intent="search", confidence=0.8)
        ).model
        == "fast-model"
    )
    assert (
        _resolve_model_route(
            args, IntentPrediction(intent="chat", confidence=0.8)
        ).model
        == "balanced-model"
    )
    assert (
        _resolve_model_route(
            args, IntentPrediction(intent="tool", confidence=0.8)
        ).model
        == "reasoning-model"
    )


def test_resolve_model_route_keeps_base_model_for_low_confidence():
    args = SimpleNamespace(
        model_routing="intent",
        model="base-model",
        fast_model="fast-model",
        balanced_model="balanced-model",
        reasoning_model="reasoning-model",
        model_routing_min_confidence=0.7,
    )

    decision = _resolve_model_route(
        args,
        IntentPrediction(intent="search", confidence=0.2),
    )

    assert decision.model == "base-model"
    assert decision.metadata["model_routing_applied"] is False


def _canonical_index(tmp_path: Path):
    """A three-route index on the basis axes, so every cosine is exact."""
    import numpy as np

    from src.model.pre_training.intents.model import DEFAULT_ENCODER
    from src.model.pre_training.intents.model import (
        INDEX_FILENAME,
        CanonicalExample,
        IntentIndex,
    )

    axes = {"search": 0, "chat": 1, "tool": 2}
    modules = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}
    examples, rows = [], []
    for route, axis in axes.items():
        for position in range(3):
            examples.append(
                CanonicalExample(
                    id=f"{route}-{position}",
                    text=f"{route} example {position}",
                    route=route,
                    modules=(modules[route],),
                )
            )
            rows.append(np.eye(3, dtype=np.float32)[axis])
    index = IntentIndex(
        examples=examples,
        vectors=np.stack(rows),
        encoder=DEFAULT_ENCODER,
        fingerprint="sha256:test",
    )
    index.save(tmp_path / INDEX_FILENAME)
    return tmp_path


def _stub_encoder(monkeypatch, vector):
    import numpy as np

    from src.model.pre_training.intents import model as encoder

    monkeypatch.setattr(
        encoder,
        "encode_texts",
        lambda texts, **kwargs: np.array([vector] * len(texts), dtype=np.float32),
    )


def test_load_intent_prediction_returns_the_nearest_canonical_route(
    tmp_path, monkeypatch
):
    directory = _canonical_index(tmp_path)
    _stub_encoder(monkeypatch, [0.0, 0.0, 1.0])
    prediction = _load_intent_prediction(str(directory), "book the room")

    assert prediction is not None
    assert prediction.intent == "tool"
    assert prediction.confidence == pytest.approx(1.0)


def test_load_intent_prediction_returns_none_when_the_index_abstains(
    tmp_path, monkeypatch
):
    """Two routes fit equally well, so there is nothing to route on.

    This used to force abstention with a `min_confidence` above every possible
    score. That gate was removed after measuring at 3 changed decisions in 416,
    so abstention is provoked the way it now actually happens: an equidistant
    query whose top two routes tie, failing the margin.
    """
    directory = _canonical_index(tmp_path)
    _stub_encoder(monkeypatch, [0.577, 0.577, 0.577])

    assert _load_intent_prediction(str(directory), "book the room") is None


def test_load_intent_prediction_rejects_an_index_built_with_a_different_encoder(
    tmp_path, monkeypatch
):
    """e5-small is also 384-d, so a stale index would otherwise score silently.

    The fixture must be genuinely 384-dimensional -- the same width as e5's
    real output -- not the 3-dim toy vectors ``_canonical_index`` uses
    elsewhere. With 3-dim rows, ``index.decide()`` would raise a numpy shape
    mismatch against the real-width query vector regardless of any
    encoder-name check, which would make this assertion hold even with the
    guard removed. A structurally valid, same-width index, paired with a
    query vector that would score an unambiguous, confident decision if
    scoring were ever reached, leaves the encoder-name check in
    ``_load_intent_prediction`` as the only thing that can make this raise.
    """
    import numpy as np

    from src.model.pre_training.intents.model import (
        INDEX_FILENAME,
        CanonicalExample,
        IntentIndex,
    )

    axes = {"search": 0, "chat": 1, "tool": 2}
    modules = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}
    basis = np.eye(384, dtype=np.float32)
    examples, rows = [], []
    for route, axis in axes.items():
        for position in range(3):
            examples.append(
                CanonicalExample(
                    id=f"{route}-{position}",
                    text=f"{route} example {position}",
                    route=route,
                    modules=(modules[route],),
                )
            )
            rows.append(basis[axis])
    directory = tmp_path / "stale"
    IntentIndex(
        examples=examples,
        vectors=np.stack(rows),
        encoder="sentence-transformers/all-MiniLM-L6-v2",
        fingerprint="sha256:test",
    ).save(directory / INDEX_FILENAME)

    # An exact match on the "search" axis: if the guard did not short-circuit
    # before this vector is ever used, decide() would return a maximally
    # confident, unambiguous decision, well clear of every default threshold.
    _stub_encoder(monkeypatch, list(basis[axes["search"]]))

    with pytest.raises(
        ValueError, match="all-MiniLM-L6-v2.*e5-small-v2|e5-small-v2.*all-MiniLM-L6-v2"
    ):
        _load_intent_prediction(str(directory), "book the room")


def test_resolve_local_device_returns_explicit_choice():
    assert _resolve_local_device("cpu") == "cpu"


def test_has_accelerate_returns_boolean():
    assert isinstance(_has_accelerate(), bool)


def test_parse_major_minor_handles_build_suffix():
    assert _parse_major_minor("2.2.2+cpu") == (2, 2)


def test_validate_local_runtime_device_rejects_mps_without_override():
    with pytest.raises(ValueError, match="Apple MPS is disabled by default"):
        _validate_local_runtime_device("mps", allow_unsafe_mps=False)


def test_validate_local_runtime_device_allows_mps_with_override():
    _validate_local_runtime_device("mps", allow_unsafe_mps=True)


def test_validate_local_runtime_stack_rejects_old_macos_mps_stack():
    with pytest.raises(ValueError, match="older runtime stack"):
        _validate_local_runtime_stack(
            "mps",
            platform_system="Darwin",
            torch_version="2.2.2",
            transformers_version="4.39.3",
        )


def test_validate_local_runtime_stack_allows_cpu_on_old_macos_stack():
    # CPU is safe even on old torch — only MPS can segfault.
    _validate_local_runtime_stack(
        "cpu",
        platform_system="Darwin",
        torch_version="2.2.2",
        transformers_version="4.39.3",
    )


def test_validate_local_runtime_stack_allows_newer_macos_mps_stack():
    _validate_local_runtime_stack(
        "mps",
        platform_system="Darwin",
        torch_version="2.5.1",
        transformers_version="4.46.0",
    )


def test_handle_local_cli_value_error_returns_true_for_known_error(capsys):
    handled = _handle_local_cli_value_error(
        ValueError(
            "Local generation on Apple MPS is disabled by default in this CLI because it can segfault"
        )
    )
    captured = capsys.readouterr()
    assert handled is True
    assert "Retry with `--device cpu`" in captured.out


def test_local_generate_sync_adds_attention_mask_for_greedy_decode():
    torch = pytest.importorskip("torch")
    from examples.run_agentic_search import LocalServerManager

    captured: dict[str, object] = {}

    class _Model:
        generation_config = SimpleNamespace(
            pad_token_id=None,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
        )

        def generate(self, inputs, **kwargs):
            captured["inputs"] = inputs
            captured["kwargs"] = kwargs
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    manager = LocalServerManager(
        model_path="dummy",
        device="cpu",
        generation_timeout_seconds=5.0,
        generation_heartbeat_seconds=999.0,
    )
    manager._tokenizer = _TokenizerWithIds()
    manager._model = _Model()

    result = manager._generate_sync([1, 2], {"max_tokens": 2, "temperature": 0})
    assert result == [3, 4]
    kwargs = captured["kwargs"]
    generation_config = kwargs["generation_config"]
    assert generation_config.do_sample is False
    assert generation_config.temperature == 1.0
    assert generation_config.top_p == 1.0
    assert generation_config.top_k == 50
    assert generation_config.pad_token_id == 99
    # max_time replaced by a wall-clock StoppingCriteria; verify it is present
    assert "stopping_criteria" in kwargs
    assert kwargs["attention_mask"].tolist() == [[1, 1]]


def test_cli_mode_resolves_to_registry_class():
    from src import get_registered_agent_loop, resolve_agent_name
    from src.agents.search import SearchAgentLoop
    from src.agents.tool import ToolAgentLoop
    from src.agents.generation import PlainGenerationLoop

    assert get_registered_agent_loop(resolve_agent_name("search")) is SearchAgentLoop
    assert get_registered_agent_loop(resolve_agent_name("tool")) is ToolAgentLoop
    assert (
        get_registered_agent_loop(resolve_agent_name("single")) is PlainGenerationLoop
    )


@pytest.mark.parametrize(
    ("top_k", "expected_route", "expected_confidence"),
    [("1", "search", 1.0), ("4", "chat", 0.8), (None, "chat", 0.8)],
)
def test_cli_and_web_intent_scoring_parity(
    tmp_path, monkeypatch, top_k, expected_route, expected_confidence
):
    """A CLI request must honor the same configured neighbors as web scoring."""
    import numpy as np

    from src.internal import configs
    from src.internal.servers.web.intent import similarity
    from src.model.pre_training.intents.model import (
        DEFAULT_ENCODER,
        INDEX_FILENAME,
        CanonicalExample,
        IntentIndex,
    )

    examples, rows = [], []
    for route, module, vectors in (
        ("search", "lookup_fact", [[1, 0, 0]] + [[0, 0, 1]] * 11),
        ("chat", "explain", [[0.8, 0.6, 0]] * 12),
        ("tool", "schedule", [[0, 0, 1]] * 12),
    ):
        for i, vector in enumerate(vectors):
            examples.append(
                CanonicalExample(f"{route}-{i}", "anchor", route, (module,))
            )
            rows.append(vector)
    IntentIndex(
        examples, np.array(rows, dtype=np.float32), DEFAULT_ENCODER, "sha256:parity"
    ).save(tmp_path / INDEX_FILENAME)
    env = {"AGENTIC_SEARCH_INTENT_INDEX_PATH": str(tmp_path)}
    if top_k is not None:
        env["AGENTIC_SEARCH_INTENT_TOP_K"] = top_k
    settings = configs.load_app_settings(env)
    monkeypatch.setattr(configs, "load_app_settings", lambda: settings)
    monkeypatch.setattr(similarity, "_INTENT_INDEXES", {})
    _stub_encoder(monkeypatch, [1.0, 0.0, 0.0])

    cli = _load_intent_prediction(str(tmp_path), "ambiguous request")
    web = similarity.predict_route("ambiguous request", settings=settings)

    assert cli is not None and web is not None
    assert web.abstain_reason is None
    assert web.strategy.value == expected_route
    assert cli.intent == expected_route
    assert cli.confidence == pytest.approx(expected_confidence)
    assert web.confidence == pytest.approx(expected_confidence)
