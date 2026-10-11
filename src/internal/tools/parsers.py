"""Tool-call parsers for LLM responses.

Each parser decodes a list of token IDs produced by the model and extracts
zero or more FunctionCall objects.  Select a parser by format name:

    parser = ToolParser.get_tool_parser("hermes", tokenizer)
    text, calls = await parser.extract_tool_calls(response_ids)

Supported formats
-----------------
hermes  — NousResearch Hermes 2.5 / 3:
              <tool_call>{"name": "...", "arguments": {...}}</tool_call>

llama3  — Meta Llama 3.1 / 3.2:
              <|python_tag|>{"name": "...", "parameters": {...}}<|eom_id|>
          or a bare JSON array of call objects.

json    — Generic fallback: any JSON object containing a ``name`` key and
          ``arguments`` / ``parameters`` found anywhere in the decoded text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FunctionCall:
    """A single tool invocation parsed from an LLM response."""

    name: str
    arguments: str  # raw JSON string; use parsed_arguments() for a dict
    call_id: str = field(default_factory=lambda: uuid4().hex[:8])

    def parsed_arguments(self) -> dict[str, Any]:
        """Decode an argument object; reject malformed JSON and nonobjects."""
        try:
            arguments = json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments must be valid JSON: " + exc.msg) from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return arguments


class ToolParser(ABC):
    """Base parser — register subclasses with ``@ToolParser.register("name")``."""

    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    async def extract_tool_calls(
        self, response_ids: list[int]
    ) -> tuple[str, list[FunctionCall]]:
        """Decode *response_ids* and return ``(content_text, tool_calls)``.

        *content_text* is the decoded text with tool-call markup stripped.
        """
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a ToolParser subclass under *name*."""

        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer: Any) -> "ToolParser":
        """Instantiate the parser registered under *name*."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown tool parser {name!r}. Registered: {sorted(cls._registry)}"
            )
        return cls._registry[name](tokenizer)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    async def _decode(
        self, response_ids: list[int], *, skip_special_tokens: bool = True
    ) -> str:
        """Decode *response_ids*, dropping special tokens by default.

        The decoded text doubles as the user-visible answer, so the template's
        EOS marker (``<|im_end|>``, ``<|eot_id|>``) must not survive into it.
        Parsers whose call markers are themselves special tokens opt out.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                response_ids, skip_special_tokens=skip_special_tokens
            ),
        )

    @staticmethod
    def _normalise(data: dict[str, Any]) -> FunctionCall | None:
        """Convert a parsed dict to a FunctionCall, accepting name/args variants."""
        name = data.get("name") or data.get("function")
        if not name:
            return None
        args = data.get("arguments", data.get("parameters", {}))
        return FunctionCall(
            name=name,
            arguments=json.dumps(args, ensure_ascii=False)
            if isinstance(args, dict)
            else str(args),
        )


@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    """Parses NousResearch Hermes 2.5 / 3 tool-call blocks.

    Expected format::

        <tool_call>
        {"name": "get_weather", "arguments": {"location": "NYC"}}
        </tool_call>
    """

    _START = "<tool_call>"
    _END = "</tool_call>"
    _BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    async def extract_tool_calls(
        self, response_ids: list[int]
    ) -> tuple[str, list[FunctionCall]]:
        text = await self._decode(response_ids)

        if self._START not in text or self._END not in text:
            return text, []

        calls: list[FunctionCall] = []
        for raw in self._BLOCK_RE.findall(text):
            try:
                data = json.loads(raw)
                call = self._normalise(data)
                if call:
                    calls.append(call)
            except (json.JSONDecodeError, ValueError):
                logger.debug("HermesToolParser: failed to parse block %r", raw[:120])

        content = self._BLOCK_RE.sub("", text).strip()
        return content, calls


@ToolParser.register("llama3")
class Llama3ToolParser(ToolParser):
    """Parses Meta Llama 3.1 / 3.2 tool-call formats.

    Variant A — python-tag block (Llama 3.1)::

        <|python_tag|>{"name": "func", "parameters": {...}}<|eom_id|>

    Variant B — bare JSON array (Llama 3.2)::

        [{"name": "func", "arguments": {...}}]
    """

    _PYTHON_TAG_RE = re.compile(
        r"<\|python_tag\|>(.*?)(?:<\|eom_id\|>|<\|eot_id\|>|$)", re.DOTALL
    )
    _JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)

    async def extract_tool_calls(
        self, response_ids: list[int]
    ) -> tuple[str, list[FunctionCall]]:
        # <|python_tag|> and <|eom_id|> are special tokens in the Llama 3
        # tokenizer, so this parser must decode with them intact.
        text = await self._decode(response_ids, skip_special_tokens=False)

        # Variant A: <|python_tag|> blocks
        python_blocks = self._PYTHON_TAG_RE.findall(text)
        if python_blocks:
            calls: list[FunctionCall] = []
            for raw in python_blocks:
                calls.extend(self._parse_block(raw.strip()))
            content = self._PYTHON_TAG_RE.sub("", text).strip()
            return content, calls

        # Variant B: bare JSON array
        for m in self._JSON_ARRAY_RE.finditer(text):
            try:
                items = json.loads(m.group(0))
                if (
                    isinstance(items, list)
                    and items
                    and isinstance(items[0], dict)
                    and "name" in items[0]
                ):
                    calls = [
                        c
                        for c in (
                            self._normalise(i) for i in items if isinstance(i, dict)
                        )
                        if c
                    ]
                    content = text[: m.start()] + text[m.end() :]
                    return content.strip(), calls
            except (json.JSONDecodeError, TypeError):
                continue

        return text, []

    def _parse_block(self, raw: str) -> list[FunctionCall]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [
                c
                for c in (self._normalise(i) for i in data if isinstance(i, dict))
                if c
            ]
        if isinstance(data, dict):
            call = self._normalise(data)
            return [call] if call else []
        return []


@ToolParser.register("json")
class JSONToolParser(ToolParser):
    """Generic fallback — scans for any JSON object with a ``name`` key.

    Works as a best-effort parser for models without a strict tool-call format.
    """

    # Matches JSON objects up to one level of nesting.
    _OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)
    # Models that wrap calls Hermes-style still route through this parser when
    # it is the configured one. Removing the JSON object alone leaves the empty
    # <tool_call></tool_call> shell behind, which then reads as the answer.
    _WRAPPER_RE = re.compile(r"</?tool_call>", re.IGNORECASE)

    async def extract_tool_calls(
        self, response_ids: list[int]
    ) -> tuple[str, list[FunctionCall]]:
        text = await self._decode(response_ids)

        calls: list[FunctionCall] = []
        consumed_spans: list[tuple[int, int]] = []

        for m in self._OBJ_RE.finditer(text):
            try:
                data = json.loads(m.group(0))
                if not isinstance(data, dict) or "name" not in data:
                    continue
                call = self._normalise(data)
                if call:
                    calls.append(call)
                    consumed_spans.append((m.start(), m.end()))
            except (json.JSONDecodeError, ValueError):
                continue

        # Strip matched spans from content (reverse order to preserve offsets)
        content = text
        for start, end in reversed(consumed_spans):
            content = content[:start] + content[end:]

        if calls:
            content = self._WRAPPER_RE.sub("", content)

        return content.strip(), calls
