"""Global tool registry for Agentic Search.

Provides a singleton registry that:
  - Wraps Python callables as tools via ``@tool`` (schema auto-inferred from type hints)
  - Registers OpenAPI operations as tools via ``register_from_openapi()``
  - Validates arguments against the declared JSON schema before execution
  - Acts as the single source of truth for both the REST management API and
    the MCP server bridge

Usage::

    from src.internal.tools.registry import tool, tool_registry

    @tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    result = await tool_registry.invoke("add", {"a": 1, "b": 2})

    # Register an OpenAPI spec (all operations become tools)
    ids = tool_registry.register_from_openapi(
        openapi_json_str, name="my_api", headers={"Authorization": "Bearer ..."}
    )
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

from .api import ApiToolRegistry, ApiToolNotFoundError
from .base import (
    FailureCategory,
    FunctionTool,
    InvalidToolInput,
    ResultKind,
    Tool,
    ToolEffect,
    ToolFailure,
)
from .validation import validate_arguments

if TYPE_CHECKING:
    from src.internal.tools.openapi_schema import OpenAPISchema

logger = logging.getLogger(__name__)

_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    bytes: "string",
}


def _params_from_signature(fn: Callable) -> dict[str, Any]:
    """Auto-build a JSON Schema ``parameters`` object from a function's type hints."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        raw_hint = hints.get(pname)

        # Unwrap Optional[X] → X
        origin = get_origin(raw_hint)
        if origin is Union:
            inner = [a for a in get_args(raw_hint) if a is not type(None)]
            raw_hint = inner[0] if inner else None

        json_type = _PY_TO_JSON_TYPE.get(raw_hint, "string")  # type: ignore[arg-type]
        prop: dict[str, Any] = {"type": json_type}

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)

        properties[pname] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


@dataclass
class ToolEntry:
    """Metadata stored alongside each registered tool."""

    tool: Tool
    source: str  # "function" | "openapi" | "mcp"
    provider_id: str | None  # set for OpenAPI- and MCP-registered tools
    # May an agent loop be offered this tool? False for tools that generate a
    # whole answer, or that re-enter an agent (which would let it call itself).
    # They stay registered and directly invocable; only agents are denied them.
    agent_callable: bool = True
    # Scoped to a specific user (per-user storage such as memory). Offered only
    # when a user is present; with none, an anonymous write would land in a
    # shared bucket and pool unrelated people's data.
    user_scoped: bool = False


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    response: str
    raw: Any
    errors: list[str]
    failure: ToolFailure | None


def validate_tool_contract(tool: Tool, *, source: str) -> list[str]:
    """The ways *tool* breaks the standard tool contract (empty = conforms)."""
    problems: list[str] = []
    if tool.result_kind is None:
        problems.append("result_kind must be declared")
    if tool.effect is ToolEffect.UNSPECIFIED and source != "mcp":
        problems.append(
            "effect must be declared (UNSPECIFIED is only for source='mcp')"
        )
    if tool.citeable and tool.result_kind is not ResultKind.DOCUMENTS:
        problems.append("citeable tools must declare result_kind DOCUMENTS")
    return problems


_INPUT_STATUSES = frozenset({400, 404, 422})


def _failure_from_exception(exc: Exception) -> ToolFailure:
    if isinstance(exc, InvalidToolInput):
        return ToolFailure(FailureCategory.INVALID_INPUT, str(exc))
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - aiohttp is a declared dependency
        aiohttp = None
    if aiohttp is not None and isinstance(exc, aiohttp.ClientResponseError):
        status = exc.status
        if status in _INPUT_STATUSES:
            category = FailureCategory.INVALID_INPUT
        elif status == 429 or status >= 500:
            category = FailureCategory.TRANSIENT
        else:
            category = FailureCategory.PERMANENT
        return ToolFailure(category, type(exc).__name__)
    transient = isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError))
    if aiohttp is not None:
        transient = transient or isinstance(exc, aiohttp.ClientError)
    try:
        import httpx

        transient = transient or isinstance(exc, httpx.TransportError)
    except ImportError:
        pass
    category = FailureCategory.TRANSIENT if transient else FailureCategory.UNKNOWN
    return ToolFailure(category, type(exc).__name__)


class ToolRegistry:
    """Singleton registry for all tools exposed via MCP and the REST API."""

    def __init__(self, *, strict: bool = False) -> None:
        self._entries: dict[str, ToolEntry] = {}
        self._api_registry = ApiToolRegistry()
        self._strict = strict

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        tool: Tool,
        *,
        source: str = "function",
        provider_id: str | None = None,
        agent_callable: bool = True,
        user_scoped: bool = False,
    ) -> None:
        """Add a tool to the registry (replaces any existing tool with the same name)."""
        if self._strict:
            problems = validate_tool_contract(tool, source=source)
            if problems:
                raise ValueError(f"tool {tool.name}: " + "; ".join(problems))
        self._entries[tool.name] = ToolEntry(
            tool=tool,
            source=source,
            provider_id=provider_id,
            agent_callable=agent_callable,
            user_scoped=user_scoped,
        )
        logger.debug("Tool registered: %s (source=%s)", tool.name, source)

    def tool(
        self,
        fn: Callable | None = None,
        *,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        effect: ToolEffect = ToolEffect.UNSPECIFIED,
        citeable: bool = False,
        stopping: bool = False,
        result_kind: ResultKind | None = None,
        retries_internally: bool = False,
    ) -> Any:
        """Decorator that registers a Python function as a tool.

        Schema is auto-inferred from type hints when *parameters* is omitted.

        Can be used with or without arguments::

            @tool_registry.tool
            def greet(name: str) -> str: ...

            @tool_registry.tool(description="Greet someone")
            def greet(name: str) -> str: ...
        """

        def _decorate(f: Callable) -> Callable:
            resolved_params = parameters or _params_from_signature(f)
            resolved_desc = description or (f.__doc__ or "").strip()
            t = FunctionTool(
                f,
                name=name or f.__name__,
                description=resolved_desc,
                parameters=resolved_params,
                effect=effect,
                citeable=citeable,
                stopping=stopping,
                result_kind=result_kind,
                retries_internally=retries_internally,
            )
            self.register(t, source="function")
            return (
                f  # return original fn, not FunctionTool, so callers can still call it
            )

        if fn is not None:
            return _decorate(fn)
        return _decorate

    def register_from_openapi(
        self,
        openapi_json: str,
        *,
        name: str,
        headers: dict[str, str] | None = None,
        icon: str | None = None,
    ) -> list[str]:
        """Parse an OpenAPI JSON string and register all operations as tools.

        Returns the list of tool names that were registered.
        """
        provider = self._api_registry.create_provider(
            name=name,
            openapi_schema=openapi_json,
            headers=headers,
            icon=icon,
        )
        tool_names: list[str] = []
        for api_tool in self._api_registry.build_tools(provider.id):
            self.register(api_tool, source="openapi", provider_id=str(provider.id))
            tool_names.append(api_tool.name)
        logger.info(
            "Registered %d tool(s) from OpenAPI provider %r (id=%s)",
            len(tool_names),
            name,
            provider.id,
        )
        return tool_names

    def register_from_schema(
        self,
        schema: "OpenAPISchema",
        *,
        name: str,
        headers: dict[str, str] | None = None,
        icon: str | None = None,
    ) -> list[str]:
        """Validate a simplified OpenAPISchema and register its operations as tools.

        Accepts the Pydantic ``OpenAPISchema`` from
        ``src.internal.tools.openapi_schema`` (already validated), converts it
        to a standard OpenAPI JSON string, then delegates to
        ``register_from_openapi()``.  This is the preferred entry point when the
        schema originates from user input — validation happens before registration.

        Returns the list of tool names registered.
        """
        return self.register_from_openapi(
            schema.to_openapi_json(),
            name=name,
            headers=headers,
            icon=icon,
        )

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        entry = self._entries.pop(name, None)
        return entry is not None

    def unregister_provider(self, provider_id: str) -> list[str]:
        """Remove all tools from an OpenAPI provider. Returns removed tool names."""
        removed = [n for n, e in self._entries.items() if e.provider_id == provider_id]
        for n in removed:
            del self._entries[n]
        try:
            self._api_registry.delete_provider(UUID(provider_id))
        except (ApiToolNotFoundError, ValueError):
            pass
        return removed

    # ------------------------------------------------------------------
    # Lookup & invocation
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        entry = self._entries.get(name)
        return entry.tool if entry else None

    def list(self) -> list[ToolEntry]:
        return list(self._entries.values())

    def list_tools(self) -> list[Tool]:
        return [e.tool for e in self._entries.values()]

    def agent_tools(self, *, user_present: bool = True) -> list[Tool]:
        """Tools an agent loop may be offered for this caller.

        ``user_present=False`` also withholds user-scoped tools; see
        ``ToolEntry.user_scoped``.
        """
        return [
            e.tool
            for e in self._entries.values()
            if e.agent_callable and (user_present or not e.user_scoped)
        ]

    def _resolve(
        self, name: str, arguments: dict[str, Any], validate: bool
    ) -> tuple[Tool | None, list[str]]:
        entry = self._entries.get(name)
        if entry is None:
            return None, [f"Tool {name!r} not found."]
        if validate and entry.tool.schema.parameters:
            return entry.tool, validate_arguments(
                entry.tool.schema.parameters, arguments
            )
        return entry.tool, []

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        validate: bool = True,
    ) -> tuple[str, Any, list[str]]:
        """Execute a tool by name.

        Returns ``(response_text, raw_output, validation_errors)``.
        *validate=True* (default) checks arguments against the declared schema
        and returns errors without executing if invalid.
        """
        tool, errors = self._resolve(name, arguments, validate)
        if tool is None or errors:
            return "", None, errors

        instance_id = await tool.create()
        try:
            response, raw, _meta = await tool.execute(instance_id, arguments)
        finally:
            await tool.release(instance_id)

        return response, raw, []

    async def invoke_detailed(
        self, name: str, arguments: dict[str, Any], *, validate: bool = True
    ) -> ToolInvocation:
        """Like ``invoke``, but every failure comes back typed instead of raised."""
        tool, errors = self._resolve(name, arguments, validate)
        if tool is None:
            return ToolInvocation(
                "", None, errors, ToolFailure(FailureCategory.NOT_FOUND, errors[0])
            )
        if errors:
            return ToolInvocation(
                "",
                None,
                errors,
                ToolFailure(FailureCategory.INVALID_INPUT, "; ".join(errors)),
            )
        instance_id = await tool.create()
        try:
            try:
                response, raw, meta = await tool.execute(instance_id, arguments)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("tool %r raised", name, exc_info=True)
                return ToolInvocation("", None, [], _failure_from_exception(exc))
        finally:
            await tool.release(instance_id)
        failure = meta.get("failure") if isinstance(meta, dict) else None
        return ToolInvocation(response, raw, [], failure)

    # ------------------------------------------------------------------
    # Summary for the REST API
    # ------------------------------------------------------------------

    @staticmethod
    def _summary(entry: ToolEntry) -> dict[str, Any]:
        t = entry.tool
        return {
            "name": t.name,
            "description": t.schema.description,
            "parameters": t.schema.parameters,
            "source": entry.source,
            "provider_id": entry.provider_id,
            "agent_callable": entry.agent_callable,
            "user_scoped": entry.user_scoped,
            "effect": t.effect.value,
            "result_kind": t.result_kind.value if t.result_kind is not None else None,
            "citeable": t.citeable,
            "retries_internally": t.retries_internally,
        }

    def tool_summary(self, name: str) -> dict[str, Any] | None:
        entry = self._entries.get(name)
        if not entry:
            return None
        return self._summary(entry)

    def all_summaries(self) -> list[dict[str, Any]]:
        return [self._summary(e) for e in self._entries.values()]


# ---------------------------------------------------------------------------
# Module-level singleton + convenience decorator
# ---------------------------------------------------------------------------

tool_registry = ToolRegistry()


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    effect: ToolEffect = ToolEffect.UNSPECIFIED,
    citeable: bool = False,
    result_kind: ResultKind | None = None,
    retries_internally: bool = False,
) -> Any:
    """Module-level shorthand for ``tool_registry.tool(...)``."""
    return tool_registry.tool(
        fn,
        name=name,
        description=description,
        parameters=parameters,
        effect=effect,
        citeable=citeable,
        result_kind=result_kind,
        retries_internally=retries_internally,
    )


__all__ = [
    "ToolRegistry",
    "ToolEntry",
    "ToolInvocation",
    "tool_registry",
    "tool",
    "validate_tool_contract",
]
