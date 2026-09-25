"""ModelUnavailableError: the one typed "model is unavailable" error."""

from __future__ import annotations


def test_llm_timeout_is_a_model_unavailable_error():
    from src.context.models import LLMTimeoutError, ModelUnavailableError

    assert issubclass(ModelUnavailableError, RuntimeError)
    assert issubclass(LLMTimeoutError, ModelUnavailableError)


def test_model_unavailable_error_is_exported_with_llm_timeout_error():
    import src.context as context

    assert context.ModelUnavailableError is context.models.ModelUnavailableError
    assert "ModelUnavailableError" in context.__all__
