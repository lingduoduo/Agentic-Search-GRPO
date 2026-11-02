import pytest

from src.model.serving import ServerManager


class _Conformer:
    async def generate(self, request_id, prompt_ids, sampling_params):
        return [1, 2, 3]


class _NonConformer:
    def something_else(self): ...


def test_protocol_is_runtime_checkable():
    assert isinstance(_Conformer(), ServerManager)
    assert not isinstance(_NonConformer(), ServerManager)


def test_managers_importable_from_serving():
    from src.model.serving import OpenAIServerManager, LocalServerManager

    assert OpenAIServerManager is not None and LocalServerManager is not None


def test_managers_still_importable_from_examples_shim():
    # Back-compat: existing call sites import from the examples module.
    from examples.run_agentic_search import OpenAIServerManager as A
    from src.model.serving import OpenAIServerManager as B

    assert A is B  # same class object, not a copy


def test_concrete_managers_conform_to_protocol():
    from src.model.serving import OpenAIServerManager, LocalServerManager

    # Do not instantiate LocalServerManager (it loads a real HF model).
    # Assert the async generate method exists on each class.
    assert callable(getattr(OpenAIServerManager, "generate", None))
    assert callable(getattr(LocalServerManager, "generate", None))


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class _Tok:
    """Minimal tokenizer stand-in for factory tests."""

    pad_token_id = 0
    eos_token_id = 0

    def encode(self, s):
        return [1]

    def decode(self, ids, **k):
        return "x"


def test_factory_selects_openai_when_server_url():
    from src.model.serving import build_server_manager, OpenAIServerManager

    mgr = build_server_manager(_Tok(), server_url="http://localhost:8080", model="m")
    assert isinstance(mgr, OpenAIServerManager)


def test_factory_raises_when_nothing_configured():
    from src.model.serving import build_server_manager

    with pytest.raises(ValueError):
        build_server_manager(_Tok())


# ---------------------------------------------------------------------------
# aclose is uniform across managers
# ---------------------------------------------------------------------------


def test_local_manager_can_be_closed():
    """Every manager the factory returns must close the same way.

    Callers hold a ``ServerManager`` without knowing which concrete class the
    factory picked; when only one of them defines ``aclose`` the call site has
    to guard, and the two bamboogle example scripts did not.
    """
    import asyncio

    from src.model.serving import LocalServerManager

    # __init__ loads nothing (the model is lazy), so this touches no weights.
    manager = LocalServerManager(model_path="unused/model", device="cpu")
    assert asyncio.run(manager.aclose()) is None


def test_factory_built_managers_all_close(monkeypatch):
    import asyncio

    from src.model.serving import build_server_manager

    monkeypatch.setattr("src.model.serving._resolve_local_device", lambda d: "cpu")
    remote = build_server_manager(_Tok(), server_url="http://localhost:8080", model="m")
    local = build_server_manager(_Tok(), model="unused/model", device="cpu")
    for manager in (remote, local):
        assert asyncio.run(manager.aclose()) is None


# ---------------------------------------------------------------------------
# One manager, several event loops
# ---------------------------------------------------------------------------


def test_openai_manager_generates_from_several_event_loops():
    """``evaluate_bamboogle`` runs ``agent.invoke`` in a thread pool and the
    example agent calls ``asyncio.run`` per question, so one manager instance is
    driven from one event loop per worker thread.  An ``aiohttp`` session is
    bound to the loop that created it, so a session cached across loops fails
    every call but the first.
    """
    import asyncio
    import http.server
    import json
    import socketserver
    import threading

    from src.model.serving import OpenAIServerManager

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"choices": [{"text": "ok"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence the stdlib access log
            pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    class _EchoTok:
        pad_token_id = 0
        eos_token_id = 0

        def decode(self, ids, **kwargs):
            return "prompt"

        def encode(self, text, **kwargs):
            return [7]

    manager = OpenAIServerManager(
        tokenizer=_EchoTok(), base_url=f"http://127.0.0.1:{port}", model="m"
    )
    results: dict[int, object] = {}

    def worker(index: int) -> None:
        try:
            results[index] = asyncio.run(
                manager.generate(f"r{index}", [1, 2, 3], {"max_tokens": 4})
            )
        except Exception as exc:  # recorded so the assert names the failure
            results[index] = f"{type(exc).__name__}: {exc}"

    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        server.shutdown()

    assert results == {i: [7] for i in range(4)}


def test_closing_in_one_loop_leaves_another_loops_session_open():
    """``--concurrency N`` closes the manager once per finished question.

    Each worker thread runs its own event loop, so the thread that finishes
    first must not tear down the session another thread is still reading from.
    The interleaving below is the one the thread pool produces: worker A opens
    its session, worker B opens its own, then A finishes and closes.
    """
    import asyncio
    import threading

    from src.model.serving import OpenAIServerManager

    manager = OpenAIServerManager(
        tokenizer=_Tok(), base_url="http://127.0.0.1:1", model="m"
    )
    a_opened = threading.Event()
    b_opened = threading.Event()
    a_closed = threading.Event()
    result: dict[str, object] = {}

    def wait(event: threading.Event):
        """Block on *event* without holding this thread's event loop."""
        return asyncio.get_running_loop().run_in_executor(None, event.wait)

    def worker_a() -> None:
        async def body() -> None:
            manager._get_session()
            a_opened.set()
            await wait(b_opened)
            await manager.aclose()
            a_closed.set()

        asyncio.run(body())

    def worker_b() -> None:
        async def body() -> None:
            await wait(a_opened)
            session = manager._get_session()
            b_opened.set()
            await wait(a_closed)
            result["b_session_closed"] = session.closed
            await manager.aclose()

        asyncio.run(body())

    threads = [threading.Thread(target=worker_a), threading.Thread(target=worker_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert result["b_session_closed"] is False


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5", "qwen3_5_text"])
def test_new_qwen_generation_configs_are_accepted(model_type):
    from types import SimpleNamespace
    from src.model.serving import _validate_local_generation_config

    _validate_local_generation_config(
        "Qwen/new-model", SimpleNamespace(model_type=model_type)
    )


@pytest.mark.parametrize(
    "model_type,expected",
    [("qwen3_5", "multimodal"), ("qwen3_5_text", "causal"), ("qwen2", "causal")],
)
def test_local_loader_matches_checkpoint_architecture(
    monkeypatch, model_type, expected
):
    import sys
    from types import SimpleNamespace
    from src.model import serving

    causal, multimodal = object(), object()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=causal,
            AutoModelForImageTextToText=multimodal,
        ),
    )
    loader = serving._local_model_loader(SimpleNamespace(model_type=model_type))
    assert loader is {"causal": causal, "multimodal": multimodal}[expected]
