"""ModelUnavailableError: the one typed "model is unavailable" error."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.context.models import LLMTimeoutError, ModelUnavailableError
from src.context.structured_output import (
    SchemaUnsupportedError,
    StructuredOutputRequest,
)
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM


def test_llm_timeout_is_a_model_unavailable_error():
    assert issubclass(ModelUnavailableError, RuntimeError)
    assert issubclass(LLMTimeoutError, ModelUnavailableError)


def test_model_unavailable_error_is_exported_with_llm_timeout_error():
    import src.context as context

    assert context.ModelUnavailableError is context.models.ModelUnavailableError
    assert "ModelUnavailableError" in context.__all__


# ---------------------------------------------------------------------------
# OpenAICompatibleLLM: which provider failures mean "the model is unavailable"
# ---------------------------------------------------------------------------

MESSAGES = [{"role": "user", "content": "hi"}]


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        LLMConfig(model_provider="openai", model_name="m", api_key="k")
    )


def _http_error(status: int, body: str = "provider says no") -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    response.url = "https://provider.invalid/chat/completions"
    return requests.HTTPError(f"{status} {body}", response=response)


def _failing_response(error: requests.HTTPError) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.side_effect = error
    return response


def _complete(llm, **kwargs):
    return llm.complete(MESSAGES, **kwargs)


def _stream_complete(llm, **kwargs):
    return list(llm.stream_complete(MESSAGES, **kwargs))


def _stream(llm, **kwargs):
    return list(llm.stream(prompt="hi"))


PATHS = [_complete, _stream_complete, _stream]


@pytest.mark.parametrize("call", PATHS)
def test_connection_error_is_model_unavailable(call):
    llm = _llm()
    cause = requests.ConnectionError("connection refused")
    with patch.object(llm._session, "post", side_effect=cause):
        with pytest.raises(ModelUnavailableError) as caught:
            call(llm)
    assert caught.value.__cause__ is cause


@pytest.mark.parametrize("call", PATHS)
@pytest.mark.parametrize("status", [500, 503, 429])
def test_5xx_and_429_are_model_unavailable(call, status):
    llm = _llm()
    error = _http_error(status)
    with patch.object(llm._session, "post", return_value=_failing_response(error)):
        with pytest.raises(ModelUnavailableError) as caught:
            call(llm)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("call", PATHS)
def test_4xx_reraises_http_error(call):
    llm = _llm()
    error = _http_error(400, "context length exceeded")
    with patch.object(llm._session, "post", return_value=_failing_response(error)):
        with pytest.raises(requests.HTTPError) as caught:
            call(llm)
    assert caught.value is error
    assert not isinstance(caught.value, ModelUnavailableError)


@pytest.mark.parametrize("call", [_complete, _stream_complete])
def test_schema_unsupported_400_still_raises_schema_unsupported(call):
    llm = _llm()
    request = StructuredOutputRequest(
        name="answer",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    error = _http_error(400, "unknown parameter: response_format")
    with patch.object(llm._session, "post", return_value=_failing_response(error)):
        with pytest.raises(SchemaUnsupportedError):
            call(llm, structured_output=request)
