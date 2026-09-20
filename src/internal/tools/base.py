"""Generic function/tool abstractions for ``ToolAgentLoop``.

Tools are plain callable capabilities exposed through JSON schemas. They are
not search-agent XML actions; search can be one tool among many, but the tool
loop itself is intentionally domain-agnostic.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ToolEffect(str, Enum):
    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"
    UNSPECIFIED = "unspecified"


@dataclass(slots=True)
class ToolSchema:
    """JSON Schema description of one generic function-calling tool."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Tool(ABC):
    """Abstract base for a callable tool used by ToolAgentLoop.

    Lifecycle per tool call:
        instance_id = await tool.create()
        response, raw, meta = await tool.execute(instance_id, arguments)
        await tool.release(instance_id)
    """

    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.UNSPECIFIED

    @property
    def citeable(self) -> bool:
        """True if this tool produces citable documents."""
        return False

    @property
    def stopping(self) -> bool:
        """True if the loop should stop after this tool runs."""
        return False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def schema(self) -> ToolSchema: ...

    async def create(self) -> str:
        """Allocate a tool instance. Returns an opaque instance_id."""
        return "default"

    @abstractmethod
    async def execute(
        self, instance_id: str, arguments: dict[str, Any]
    ) -> tuple[str, Any, Any]:
        """Execute the tool. Returns (response_text, raw_output, metadata)."""
        ...

    async def release(self, instance_id: str) -> None:
        """Free any resources held by *instance_id*."""


class FunctionTool(Tool):
    """Wrap a plain Python callable (sync or async) as a Tool.

    Example::

        @FunctionTool.from_fn(
            description="Search Wikipedia",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        async def search_wiki(query: str) -> str:
            ...
    """

    def __init__(
        self,
        fn: Callable,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        effect: ToolEffect = ToolEffect.UNSPECIFIED,
        citeable: bool = False,
        stopping: bool = False,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._name = name or fn.__name__
        self._effect = effect
        self._citeable = citeable
        self._stopping = stopping
        self._schema = ToolSchema(
            name=self._name,
            description=description or (fn.__doc__ or "").strip(),
            parameters=parameters or {},
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    @property
    def effect(self) -> ToolEffect:
        return self._effect

    @property
    def citeable(self) -> bool:
        return self._citeable

    @property
    def stopping(self) -> bool:
        return self._stopping

    async def execute(
        self, instance_id: str, arguments: dict[str, Any]
    ) -> tuple[str, Any, Any]:
        if inspect.iscoroutinefunction(self._fn):
            result = await self._fn(**arguments)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self._fn(**arguments))
        # JSON, not str(): the source-card builder json.loads this back into
        # structure and _fit_json_array trims JSON arrays by whole items. A
        # Python repr defeats both silently. A str passes through untouched,
        # since json.dumps would only add quotes. See #596.
        if isinstance(result, str):
            response = result
        else:
            try:
                response = json.dumps(result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                response = str(result)
        return response, result, {}

    @classmethod
    def from_fn(
        cls,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        name: str | None = None,
        effect: ToolEffect = ToolEffect.UNSPECIFIED,
        citeable: bool = False,
        stopping: bool = False,
    ) -> Callable:
        """Decorator factory that wraps a function as a FunctionTool."""

        def decorator(fn: Callable) -> "FunctionTool":
            return cls(
                fn,
                name=name,
                description=description,
                parameters=parameters,
                effect=effect,
                citeable=citeable,
                stopping=stopping,
            )

        return decorator
