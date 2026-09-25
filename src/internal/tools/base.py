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


class ResultKind(str, Enum):
    """What a tool's successful response text is."""

    DOCUMENTS = "documents"  # JSON array of {title, content, url} (string values)
    JSON = "json"
    TEXT = "text"


class InvalidToolInput(ValueError):
    """A tool argument is wrong; the model should correct it and call again."""


class FailureCategory(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ToolFailure:
    """Why a tool call failed, in a form the recovery policy can act on.

    ``message`` is a short fixed-vocabulary description safe to show a user;
    raw exception text and provider bodies stay in the logs.
    """

    category: FailureCategory
    message: str
    retry_after: float | None = None
    provider_attempts: int = 0
    is_timeout: bool = False


def is_timeout_exception(exc: BaseException) -> bool:
    """Recognize timeout types, retaining Python 3.10 asyncio compatibility."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, httpx.TimeoutException)


class ToolErrorText(str):
    """Error text a tool returns, carrying the typed failure behind it.

    A ``str``, so every caller that only reads text sees exactly what it did
    before; the agent loop reads ``.failure``.
    """

    failure: ToolFailure

    def __new__(cls, text: str, failure: ToolFailure) -> "ToolErrorText":
        obj = super().__new__(cls, text)
        obj.failure = failure
        return obj


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
    def result_kind(self) -> "ResultKind | None":
        """What a successful response is. None = undeclared (a strict registry rejects it)."""
        return None

    @property
    def retries_internally(self) -> bool:
        """True when the provider already retries transient failures itself."""
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
        result_kind: "ResultKind | None" = None,
        retries_internally: bool = False,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._name = name or fn.__name__
        self._effect = effect
        self._citeable = citeable
        self._result_kind = result_kind
        self._retries_internally = retries_internally
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
    def result_kind(self) -> "ResultKind | None":
        return self._result_kind

    @property
    def retries_internally(self) -> bool:
        return self._retries_internally

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
        meta = {"failure": result.failure} if isinstance(result, ToolErrorText) else {}
        return response, result, meta

    @classmethod
    def from_fn(
        cls,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        name: str | None = None,
        effect: ToolEffect = ToolEffect.UNSPECIFIED,
        citeable: bool = False,
        result_kind: "ResultKind | None" = None,
        retries_internally: bool = False,
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
                result_kind=result_kind,
                retries_internally=retries_internally,
            )

        return decorator
