"""Tool-call parsers decode model output before extracting calls."""

from __future__ import annotations

import pytest

from src.internal.tools.parsers import FunctionCall, ToolParser


@pytest.mark.parametrize("raw", ["{broken", "[]", '"text"', "42", "null", "true"])
def test_function_call_rejects_invalid_argument_objects(raw):
    with pytest.raises(ValueError, match="JSON"):
        FunctionCall(name="tool", arguments=raw).parsed_arguments()


@pytest.mark.parametrize(
    "raw,expected", [("", {}), ("{}", {}), ('{"value": 1}', {"value": 1})]
)
def test_function_call_keeps_valid_objects(raw, expected):
    assert FunctionCall(name="tool", arguments=raw).parsed_arguments() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["json", "hermes", "llama3"])
@pytest.mark.parametrize("arguments", [{"value": 1}, '{"value": 1}'])
async def test_parser_preserves_object_argument_formats(fmt, arguments):
    import json

    text = json.dumps({"name": "tool", "arguments": arguments})
    if fmt == "hermes":
        text = f"<tool_call>{text}</tool_call>"
    elif fmt == "llama3":
        text = f"<|python_tag|>{text}<|eom_id|>"
    parser = ToolParser.get_tool_parser(fmt, _RecordingTokenizer(text))
    _, calls = await parser.extract_tool_calls([1])
    assert len(calls) == 1
    assert calls[0].parsed_arguments() == {"value": 1}


class _RecordingTokenizer:
    """Decodes to a fixed string and records how ``decode`` was called."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def decode(self, ids, **kwargs):
        self.calls.append(kwargs)
        return self.text


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["hermes", "json"])
async def test_parsers_strip_special_tokens_from_content(fmt):
    # Without skip_special_tokens the EOS marker lands in the user-visible answer.
    tokenizer = _RecordingTokenizer("FAISS is a similarity search library.")
    parser = ToolParser.get_tool_parser(fmt, tokenizer)

    content, calls = await parser.extract_tool_calls([1, 2, 3])

    assert calls == []
    assert content == "FAISS is a similarity search library."
    assert tokenizer.calls == [{"skip_special_tokens": True}]


@pytest.mark.asyncio
async def test_llama3_parser_keeps_special_tokens():
    # <|python_tag|>/<|eom_id|> are special tokens in the Llama 3 tokenizer;
    # stripping them would erase the very markers this parser matches on.
    tokenizer = _RecordingTokenizer(
        '<|python_tag|>{"name": "search", "parameters": {"query": "faiss"}}<|eom_id|>'
    )
    parser = ToolParser.get_tool_parser("llama3", tokenizer)

    content, calls = await parser.extract_tool_calls([1, 2, 3])

    assert tokenizer.calls == [{"skip_special_tokens": False}]
    assert [c.name for c in calls] == ["search"]
    assert content == ""


@pytest.mark.asyncio
async def test_json_parser_strips_leftover_tool_call_markup():
    # Qwen emits Hermes-style <tool_call> wrappers around the JSON. The generic
    # JSON parser removes the object but used to leave the tags behind, so a run
    # that ended on a tool-calling turn surfaced "<tool_call></tool_call>" as
    # its answer.
    tokenizer = _RecordingTokenizer(
        '<tool_call>\n{"name": "search", "arguments": {"query": "x"}}\n</tool_call>'
    )
    parser = ToolParser.get_tool_parser("json", tokenizer)

    content, calls = await parser.extract_tool_calls([1, 2, 3])

    assert [c.name for c in calls] == ["search"]
    assert content == ""


@pytest.mark.asyncio
async def test_json_parser_keeps_prose_around_a_tool_call():
    tokenizer = _RecordingTokenizer(
        'Let me look that up.\n<tool_call>{"name": "search", "arguments": {}}</tool_call>'
    )
    parser = ToolParser.get_tool_parser("json", tokenizer)

    content, calls = await parser.extract_tool_calls([1])

    assert [c.name for c in calls] == ["search"]
    assert content == "Let me look that up."
