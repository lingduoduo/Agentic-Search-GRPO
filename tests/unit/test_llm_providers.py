"""Tests for OpenAICompatibleLLM and get_llm_for_persona."""

from __future__ import annotations

import json
import pytest
import requests
from unittest.mock import MagicMock, patch


from src.context.models import LLMTimeoutError, ModelUnavailableError
from src.context.structured_output import StructuredOutputRequest
from src.internal.llm.interfaces import LLMConfig, ToolChoiceOptions
from src.internal.llm.providers import (
    OpenAICompatibleLLM,
    _parse_sse_chunk,
)


# ---------------------------------------------------------------------------
# _parse_sse_chunk
# ---------------------------------------------------------------------------


def test_parse_sse_chunk_content():
    line = 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'
    chunk = _parse_sse_chunk(line)
    assert chunk is not None
    assert chunk.choice.delta.content == "hello"
    assert chunk.choice.finish_reason is None
    assert chunk.usage is None


def test_parse_sse_chunk_done():
    assert _parse_sse_chunk("data: [DONE]") is None


def test_parse_sse_chunk_non_data():
    assert _parse_sse_chunk(": keep-alive") is None


def test_parse_sse_chunk_finish_reason():
    line = 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
    chunk = _parse_sse_chunk(line)
    assert chunk is not None
    assert chunk.choice.finish_reason == "stop"


def test_parse_sse_chunk_usage():
    line = json.dumps(
        {
            "choices": [{"delta": {"content": "hi"}, "finish_reason": None}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
    )
    chunk = _parse_sse_chunk(f"data: {line}")
    assert chunk is not None
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 5


def test_parse_sse_chunk_tool_call():
    line = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    chunk = _parse_sse_chunk(f"data: {line}")
    assert chunk is not None
    tc = chunk.choice.delta.tool_calls[0]
    assert tc.function.name == "search"
    assert tc.id == "call_abc"


# ---------------------------------------------------------------------------
# OpenAICompatibleLLM.stream
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    return LLMConfig(
        model_provider="openai",
        model_name="gpt-4o-mini",
        api_key="test-key",
        **kwargs,
    )


def _mock_response(*data_lines: str):
    """Build a mock requests.Response that streams the given SSE lines."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = iter(data_lines)
    return resp


def test_stream_yields_content_chunks():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    lines = [
        'data: {"choices":[{"delta":{"content":"hello "},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    with patch("requests.Session.post", return_value=_mock_response(*lines)):
        chunks = list(llm.stream(prompt="hi"))
    assert len(chunks) == 2
    assert chunks[0].choice.delta.content == "hello "
    assert chunks[1].choice.finish_reason == "stop"


def test_stream_normalises_string_prompt():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt="hello"))
    body = mock_post.call_args.kwargs["json"]
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_stream_normalises_dict_messages():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt=msgs))
    body = mock_post.call_args.kwargs["json"]
    assert body["messages"] == msgs


def test_stream_includes_tools():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt="hi", tools=tools, tool_choice=ToolChoiceOptions.AUTO))
    body = mock_post.call_args.kwargs["json"]
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def test_stream_no_tools_omits_tool_fields():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt="hi"))
    body = mock_post.call_args.kwargs["json"]
    assert "tools" not in body
    assert "tool_choice" not in body


def test_stream_puts_the_json_schema_in_the_request_body():
    """structured_response_format was accepted and silently dropped."""
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    lines = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}',
        "data: [DONE]",
    ]
    schema_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer_draft"},
    }
    with patch(
        "requests.Session.post", return_value=_mock_response(*lines)
    ) as mock_post:
        list(llm.stream(prompt="hi", structured_response_format=schema_format))
    body = mock_post.call_args.kwargs["json"]
    assert body["response_format"]["json_schema"]["name"] == "answer_draft"


def test_stream_complete_yields_text_deltas_and_applies_the_schema():
    config = _make_config()
    llm = OpenAICompatibleLLM(config)
    lines = [
        'data: {"choices":[{"delta":{"content":"{\\"abstain\\""}}]}',
        'data: {"choices":[{"delta":{"content":": false}"}}]}',
        "data: [DONE]",
    ]
    request = StructuredOutputRequest(name="answer_draft", schema={"type": "object"})
    with patch(
        "requests.Session.post", return_value=_mock_response(*lines)
    ) as mock_post:
        chunks = list(
            llm.stream_complete(
                [{"role": "user", "content": "q"}], structured_output=request
            )
        )
    body = mock_post.call_args.kwargs["json"]
    assert "".join(chunks) == '{"abstain": false}'
    assert body["response_format"]["json_schema"]["name"] == "answer_draft"
    assert body["stream"] is True


def test_stream_complete_omits_the_schema_when_the_provider_cannot_enforce_it():
    """PROMPT_ONLY providers must not receive a response_format they will reject."""
    config = LLMConfig(model_provider="ollama", model_name="llama3")
    llm = OpenAICompatibleLLM(config)
    with patch(
        "requests.Session.post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(
            llm.stream_complete(
                [{"role": "user", "content": "q"}],
                structured_output=StructuredOutputRequest(name="d", schema={}),
            )
        )
    body = mock_post.call_args.kwargs["json"]
    assert "response_format" not in body


def test_stream_custom_base_url():
    config = LLMConfig(
        model_provider="ollama_chat",
        model_name="llama3",
        api_base="http://localhost:11434/v1",
    )
    llm = OpenAICompatibleLLM(config)
    assert llm._endpoint == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in llm._headers


# ---------------------------------------------------------------------------
# get_llm_for_persona
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# complete() — non-streaming path
# ---------------------------------------------------------------------------


def test_complete_returns_message_content():
    config = LLMConfig(
        model_provider="openai", model_name="gpt-4o-mini", api_key="sk-test"
    )
    llm = OpenAICompatibleLLM(config)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Dense retrieval uses vectors."}}]
    }
    mock_resp.raise_for_status.return_value = None

    with patch.object(llm._session, "post", return_value=mock_resp):
        result = llm.complete([{"role": "user", "content": "What is dense retrieval?"}])

    assert result == "Dense retrieval uses vectors."


def test_complete_passes_messages_to_api():
    config = LLMConfig(
        model_provider="openai", model_name="gpt-4o-mini", api_key="sk-test"
    )
    llm = OpenAICompatibleLLM(config)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_resp.raise_for_status.return_value = None

    with patch.object(llm._session, "post", return_value=mock_resp) as mock_post:
        llm.complete([{"role": "user", "content": "hello"}])

    body = mock_post.call_args[1]["json"]
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_complete_forwards_temperature():
    # A caller that pins temperature (e.g. the route classifier) must have it
    # forwarded to the API so decoding is deterministic instead of relying on
    # the server default (typically 1.0).
    config = LLMConfig(
        model_provider="openai", model_name="gpt-4o-mini", api_key="sk-test"
    )
    llm = OpenAICompatibleLLM(config)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_resp.raise_for_status.return_value = None

    with patch.object(llm._session, "post", return_value=mock_resp) as mock_post:
        llm.complete([{"role": "user", "content": "hello"}], temperature=0.0)

    body = mock_post.call_args[1]["json"]
    assert body["temperature"] == 0.0


# ---------------------------------------------------------------------------
# LLM timeout handling
# ---------------------------------------------------------------------------


def _timeout_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider="openai",
            model_name="test-model",
            api_key="test-key",
        )
    )


def test_read_timeout_is_normalized_to_llm_timeout_error():
    llm = _timeout_llm()
    with patch.object(
        llm._session,
        "post",
        side_effect=requests.ReadTimeout(
            "HTTPSConnectionPool(host='secret-endpoint.invalid', port=443): Read timed out."
        ),
    ):
        with pytest.raises(LLMTimeoutError):
            llm.complete([{"role": "user", "content": "hi"}])


def test_connect_timeout_is_normalized_to_llm_timeout_error():
    # ConnectTimeout subclasses BOTH Timeout and ConnectionError. It is a timeout,
    # so it must be handled — the ConnectionError exclusion does not reach it.
    llm = _timeout_llm()
    with patch.object(
        llm._session, "post", side_effect=requests.ConnectTimeout("connect timed out")
    ):
        with pytest.raises(LLMTimeoutError):
            llm.complete([{"role": "user", "content": "hi"}])


def test_llm_timeout_error_does_not_leak_endpoint_or_original_text():
    llm = _timeout_llm()
    with patch.object(
        llm._session,
        "post",
        side_effect=requests.ReadTimeout(
            "HTTPSConnectionPool(host='secret-endpoint.invalid', port=443): Read timed out."
        ),
    ):
        with pytest.raises(LLMTimeoutError) as caught:
            llm.complete([{"role": "user", "content": "hi"}])
    assert "secret-endpoint.invalid" not in str(caught.value)
    assert "HTTPSConnectionPool" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_plain_connection_error_is_model_unavailable():
    # A host that is down or refusing means the model is unavailable: the web
    # dispatch degrades to search-only on this type instead of a 502.
    llm = _timeout_llm()
    with patch.object(
        llm._session,
        "post",
        side_effect=requests.ConnectionError("connection refused"),
    ):
        with pytest.raises(ModelUnavailableError):
            llm.complete([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# stream() / stream_complete() timeout handling — Findings 1 and 3
# ---------------------------------------------------------------------------


def test_stream_default_timeout_matches_complete():
    """complete() defaults to 30s; stream() must match, not 120s (Finding 3)."""
    llm = _timeout_llm()
    with patch.object(
        llm._session, "post", return_value=_mock_response("data: [DONE]")
    ) as mock_post:
        list(llm.stream(prompt="hi"))
    assert mock_post.call_args.kwargs["timeout"] == 30


def test_stream_timeout_at_call_time_is_normalized_to_llm_timeout_error():
    """A timeout raised while opening the connection must become LLMTimeoutError,
    exactly as complete() already does."""
    llm = _timeout_llm()
    with patch.object(
        llm._session,
        "post",
        side_effect=requests.ReadTimeout("read timed out"),
    ):
        with pytest.raises(LLMTimeoutError):
            list(llm.stream(prompt="hi"))


def test_stream_timeout_during_iteration_is_normalized_to_llm_timeout_error():
    """Critical placement check: stream() is a generator, so its body — the
    initial POST included — only runs on first iteration. A timeout raised
    later, while reading subsequent chunks, must ALSO become LLMTimeoutError,
    not just one raised by the initial call."""

    def _lines_then_timeout():
        yield 'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}'
        raise requests.ReadTimeout("read timed out mid-stream")

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = _lines_then_timeout()

    llm = _timeout_llm()
    with patch.object(llm._session, "post", return_value=resp):
        with pytest.raises(LLMTimeoutError):
            list(llm.stream(prompt="hi"))


def test_stream_complete_timeout_during_iteration_is_normalized_to_llm_timeout_error():
    """The pipeline calls stream_complete, not stream, directly — pin that the
    conversion is visible through that entry point too."""

    def _lines_then_timeout():
        yield 'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}'
        raise requests.ReadTimeout("read timed out mid-stream")

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = _lines_then_timeout()

    llm = _timeout_llm()
    with patch.object(llm._session, "post", return_value=resp):
        with pytest.raises(LLMTimeoutError):
            list(llm.stream_complete([{"role": "user", "content": "hi"}]))
