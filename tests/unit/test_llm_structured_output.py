from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.context.models import LLMResponse, LLMTimeoutError, ModelUnavailableError
from src.context.structured_output import (
    SchemaUnsupportedError,
    StructuredOutputCapability,
    StructuredOutputRequest,
)
from src.internal.llm.constants import LlmProviderNames
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.multi_llm import LitellmLLM
from src.internal.llm.models import UserMessage
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.servers.web import request_capture as rc


MESSAGES = [{"role": "user", "content": "secret prompt"}]


@pytest.fixture
def schema_request() -> StructuredOutputRequest:
    return StructuredOutputRequest(
        name="answer",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )


def configured_llm(
    provider: str = "openai", custom_config: dict[str, str] | None = None
) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=provider,
            model_name="test-model",
            api_key="test-key",
            custom_config=custom_config,
        )
    )


def configured_litellm(
    provider: str, custom_config: dict[str, str] | None = None
) -> LitellmLLM:
    return LitellmLLM(
        api_key="test-key",
        model_provider=provider,
        model_name="test-model",
        max_input_tokens=1024,
        custom_config=custom_config,
        extra_body=None,
    )


def successful_response(
    content: str = '{"answer":"ok"}',
    *,
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, "refusal": refusal},
            }
        ]
    }
    return response


def http_error(status: int, body: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    response.url = "https://provider.invalid/chat/completions"
    return requests.HTTPError(f"{status} provider body: {body}", response=response)


def test_native_openai_complete_sends_strict_json_schema(schema_request):
    llm = configured_llm()
    with patch.object(llm._session, "post", return_value=successful_response()) as post:
        llm.complete(MESSAGES, structured_output=schema_request)

    assert post.call_args.kwargs["json"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": schema_request.name,
            "strict": True,
            "schema": schema_request.schema,
        },
    }


def test_generic_compatible_endpoint_is_prompt_only():
    assert (
        configured_llm("openai_compatible").structured_output_capability
        is StructuredOutputCapability.PROMPT_ONLY
    )


@pytest.mark.parametrize("value", ["true", "TRUE", "TrUe"])
def test_generic_compatible_endpoint_can_explicitly_enable_schema(value):
    llm = configured_llm("openai_compatible", {"supports_json_schema": value})
    assert llm.structured_output_capability is StructuredOutputCapability.JSON_SCHEMA


@pytest.mark.parametrize("value", ["false", "1", " yes ", " true "])
def test_generic_compatible_endpoint_requires_exact_true_value(value):
    llm = configured_llm("openai_compatible", {"supports_json_schema": value})
    assert llm.structured_output_capability is StructuredOutputCapability.PROMPT_ONLY


def test_non_compatible_provider_cannot_enable_schema_with_generic_flag():
    llm = configured_llm("anthropic", {"supports_json_schema": "true"})
    assert llm.structured_output_capability is StructuredOutputCapability.PROMPT_ONLY


def test_prompt_only_complete_does_not_send_response_format(schema_request):
    llm = configured_llm("openai_compatible")
    with patch.object(llm._session, "post", return_value=successful_response()) as post:
        result = llm.complete(MESSAGES, structured_output=schema_request)
    assert "response_format" not in post.call_args.kwargs["json"]
    assert isinstance(result, LLMResponse)
    assert result.structured.requested is True
    assert result.structured.applied is False


def test_structured_complete_retains_refusal(schema_request):
    llm = configured_llm()
    with patch.object(
        llm._session,
        "post",
        return_value=successful_response("", refusal="cannot comply"),
    ):
        result = llm.complete(MESSAGES, structured_output=schema_request)
    assert isinstance(result, LLMResponse)
    assert result.structured.refused is True
    assert result.raw == {"refusal": "cannot comply", "finish_reason": "stop"}


def test_structured_complete_retains_length_incomplete_reason(schema_request):
    llm = configured_llm()
    with patch.object(
        llm._session,
        "post",
        return_value=successful_response(finish_reason="length"),
    ):
        result = llm.complete(MESSAGES, structured_output=schema_request)
    assert isinstance(result, LLMResponse)
    assert result.structured.incomplete_reason == "length"


def test_explicit_unsupported_schema_400_is_classified_without_raw_body(schema_request):
    llm = configured_llm()
    error = http_error(400, "unknown parameter: response_format json_schema SECRET")
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(SchemaUnsupportedError) as caught:
            llm.complete(MESSAGES, structured_output=schema_request)
    assert "SECRET" not in str(caught.value)
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "body", ["response_format is not supported", "json_schema not supported"]
)
def test_explicit_not_supported_schema_400_is_classified(schema_request, body):
    llm = configured_llm()
    error = http_error(400, body)
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(SchemaUnsupportedError):
            llm.complete(MESSAGES, structured_output=schema_request)


@pytest.mark.parametrize(
    ("status", "body"),
    [(400, "invalid API key")],
)
def test_other_http_errors_propagate_unchanged(schema_request, status, body):
    llm = configured_llm()
    error = http_error(status, body)
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(requests.HTTPError) as caught:
            llm.complete(MESSAGES, structured_output=schema_request)
    assert caught.value is error


@pytest.mark.parametrize(
    ("status", "body"),
    [(429, "response_format is unsupported"), (500, "json_schema failed")],
)
def test_unavailable_statuses_become_model_unavailable(schema_request, status, body):
    llm = configured_llm()
    error = http_error(status, body)
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(ModelUnavailableError) as caught:
            llm.complete(MESSAGES, structured_output=schema_request)
    assert caught.value.__cause__ is error


def test_timeout_is_normalized_to_llm_timeout_error(schema_request):
    llm = configured_llm()
    error = requests.Timeout("timeout")
    with patch.object(llm._session, "post", side_effect=error):
        with pytest.raises(LLMTimeoutError):
            llm.complete(MESSAGES, structured_output=schema_request)


def test_connection_error_becomes_model_unavailable(schema_request):
    llm = configured_llm()
    error = requests.ConnectionError("down")
    with patch.object(llm._session, "post", side_effect=error):
        with pytest.raises(ModelUnavailableError) as caught:
            llm.complete(MESSAGES, structured_output=schema_request)
    assert caught.value.__cause__ is error


def test_schema_error_is_not_classified_without_structured_request():
    llm = configured_llm()
    error = http_error(400, "response_format is unsupported")
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(requests.HTTPError) as caught:
            llm.complete(MESSAGES)
    assert caught.value is error


def test_prompt_only_schema_400_propagates_unchanged(schema_request):
    llm = configured_llm("openai_compatible")
    error = http_error(400, "unknown parameter: response_format")
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(requests.HTTPError) as caught:
            llm.complete(MESSAGES, structured_output=schema_request)
    assert caught.value is error


def test_incidental_schema_error_substrings_propagate_unchanged(schema_request):
    llm = configured_llm()
    error = http_error(
        400, "unknown account option; documentation mentions response_format"
    )
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(requests.HTTPError) as caught:
            llm.complete(MESSAGES, structured_output=schema_request)
    assert caught.value is error


def test_stream_complete_downgrades_on_explicit_unsupported_schema_400(schema_request):
    """Finding 2: a provider that rejects response_format only while
    streaming must trigger the same downgrade `complete` gets, reusing the
    shared detection helper rather than a second copy of the regex."""
    llm = configured_llm()
    error = http_error(400, "unknown parameter: response_format json_schema SECRET")
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(SchemaUnsupportedError) as caught:
            list(llm.stream_complete(MESSAGES, structured_output=schema_request))
    assert "SECRET" not in str(caught.value)


def test_stream_complete_other_http_errors_propagate_unchanged(schema_request):
    llm = configured_llm()
    error = http_error(400, "invalid API key")
    response = MagicMock()
    response.raise_for_status.side_effect = error
    with patch.object(llm._session, "post", return_value=response):
        with pytest.raises(requests.HTTPError) as caught:
            list(llm.stream_complete(MESSAGES, structured_output=schema_request))
    assert caught.value is error


@pytest.mark.parametrize(
    ("provider", "custom_config", "expected_forwarded"),
    [
        (LlmProviderNames.OPENAI_COMPATIBLE, None, False),
        (LlmProviderNames.OPENAI, None, True),
        (
            LlmProviderNames.OPENAI_COMPATIBLE,
            {"supports_json_schema": "true"},
            True,
        ),
    ],
)
def test_litellm_wrapper_forwarding_respects_capability(
    schema_request, provider, custom_config, expected_forwarded, monkeypatch
):
    llm = configured_litellm(provider, custom_config)
    completion = MagicMock(return_value=MagicMock())
    fake_litellm = ModuleType("litellm")
    fake_litellm.completion = completion
    fake_litellm.model_cost = {}
    fake_litellm.supports_reasoning = MagicMock(return_value=False)
    fake_exceptions = ModuleType("litellm.exceptions")
    fake_exceptions.RateLimitError = type("RateLimitError", (Exception,), {})
    fake_exceptions.Timeout = type("Timeout", (Exception,), {})
    fake_litellm.exceptions = fake_exceptions
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setitem(sys.modules, "litellm.exceptions", fake_exceptions)
    fake_singleton = ModuleType("src.internal.llm.litellm_singleton")
    fake_singleton.litellm = fake_litellm
    monkeypatch.setitem(
        sys.modules, "src.internal.llm.litellm_singleton", fake_singleton
    )

    llm._completion(
        UserMessage(content="secret prompt"),
        tools=None,
        tool_choice=None,
        stream=False,
        parallel_tool_calls=False,
        structured_response_format=schema_request,
    )

    if expected_forwarded:
        assert completion.call_args.kwargs["response_format"]["type"] == "json_schema"
    else:
        assert "response_format" not in completion.call_args.kwargs


def test_structured_capture_contains_only_safe_metadata(schema_request):
    llm = configured_llm()
    token = rc.start_capture("request", "query")
    try:
        with patch.object(llm._session, "post", return_value=successful_response()):
            llm.complete(MESSAGES, structured_output=schema_request)
        payload = [s.payload for s in rc.active().stages if s.stage == "llm"][0]
    finally:
        rc.reset_capture(token)

    assert payload == {
        "model": "test-model",
        "structured": {
            "requested": True,
            "applied": True,
            "downgraded": False,
            "refused": False,
            "incomplete_reason": None,
        },
    }


def test_legacy_complete_still_returns_string(schema_request):
    llm = configured_llm()
    with patch.object(llm._session, "post", return_value=successful_response("plain")):
        assert llm.complete(MESSAGES) == "plain"
