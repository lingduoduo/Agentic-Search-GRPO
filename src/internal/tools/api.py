"""OpenAPI-backed tools for ``ToolAgentLoop``.

This module provides a small in-memory service for turning OpenAPI operations
into repo-native ``Tool`` instances. It borrows the useful shape of a provider
service without adding database or dependency-injection requirements.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from ...context.retrieval.client import aiohttp
from ..configs.timeouts import get_timeout_policies
from .base import ResultKind, Tool, ToolEffect, ToolSchema

HTTP_METHODS = {"get", "head", "options", "post", "put", "patch", "delete"}
_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")
_PATH_PARAM_RE = re.compile(r"\{([^{}]+)\}")


class ApiToolError(ValueError):
    """Raised when an API tool provider or schema is invalid."""


class ApiToolNotFoundError(LookupError):
    """Raised when an API tool provider or operation cannot be found."""


@dataclass(frozen=True)
class OpenAPISchema:
    """Parsed subset of an OpenAPI schema needed to build API tools."""

    title: str
    description: str
    server: str
    paths: dict[str, dict[str, dict[str, Any]]]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ApiToolSpec:
    """One API operation exposed as a callable tool."""

    name: str
    description: str
    url: str
    method: str
    parameters: dict[str, Any]
    openapi_parameters: list[dict[str, Any]] = field(default_factory=list)
    request_body_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class ApiToolProviderSpec:
    """A named OpenAPI provider plus the tools derived from it."""

    id: UUID
    name: str
    description: str
    openapi_schema: str
    headers: dict[str, str] = field(default_factory=dict)
    icon: str | None = None
    tools: dict[str, ApiToolSpec] = field(default_factory=dict)


class ApiRequestTool(Tool):
    """Tool implementation that invokes one registered OpenAPI operation."""

    def __init__(
        self, registry: "ApiToolRegistry", provider_id: UUID, spec: ApiToolSpec
    ):
        self._registry = registry
        self._provider_id = provider_id
        self._spec = spec
        self._schema = ToolSchema(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
        )

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    @property
    def effect(self) -> ToolEffect:
        if self._spec.method in {"get", "head", "options"}:
            return ToolEffect.READ_ONLY
        return ToolEffect.SIDE_EFFECTING

    @property
    def result_kind(self) -> ResultKind:
        return ResultKind.JSON

    async def execute(
        self, instance_id: str, arguments: dict[str, Any]
    ) -> tuple[str, Any, Any]:
        del instance_id
        raw = await self._registry.invoke_tool(
            self._provider_id,
            self._spec.name,
            arguments,
        )
        return _format_tool_response(raw), raw, {"provider_id": str(self._provider_id)}


class ApiToolRegistry:
    """In-memory provider registry for OpenAPI-backed tools."""

    def __init__(self) -> None:
        self._providers: dict[UUID, ApiToolProviderSpec] = {}

    def create_provider(
        self,
        *,
        name: str,
        openapi_schema: str,
        headers: dict[str, str] | None = None,
        icon: str | None = None,
    ) -> ApiToolProviderSpec:
        """Register a provider from an OpenAPI JSON string."""

        self._ensure_unique_name(name)
        parsed = parse_openapi_schema(openapi_schema)
        provider_id = uuid4()
        tools = _build_tool_specs(parsed)
        provider = ApiToolProviderSpec(
            id=provider_id,
            name=name,
            description=parsed.description,
            icon=icon,
            headers=headers or {},
            openapi_schema=openapi_schema,
            tools={tool.name: tool for tool in tools},
        )
        self._providers[provider_id] = provider
        return provider

    def update_provider(
        self,
        provider_id: UUID,
        *,
        name: str | None = None,
        openapi_schema: str | None = None,
        headers: dict[str, str] | None = None,
        icon: str | None = None,
    ) -> ApiToolProviderSpec:
        """Update a provider and rebuild derived tools when the schema changes."""

        provider = self.get_provider(provider_id)
        next_name = name or provider.name
        self._ensure_unique_name(next_name, exclude_provider_id=provider_id)

        schema_text = openapi_schema or provider.openapi_schema
        parsed = parse_openapi_schema(schema_text)
        tools = _build_tool_specs(parsed)
        updated = ApiToolProviderSpec(
            id=provider.id,
            name=next_name,
            description=parsed.description,
            icon=icon if icon is not None else provider.icon,
            headers=headers if headers is not None else provider.headers,
            openapi_schema=schema_text,
            tools={tool.name: tool for tool in tools},
        )
        self._providers[provider_id] = updated
        return updated

    def delete_provider(self, provider_id: UUID) -> None:
        """Remove a provider and all tools derived from it."""

        if provider_id not in self._providers:
            raise ApiToolNotFoundError("Provider does not exist.")
        del self._providers[provider_id]

    def list_providers(
        self, search_word: str | None = None
    ) -> list[ApiToolProviderSpec]:
        """Return providers, optionally filtered by provider name."""

        providers = list(self._providers.values())
        if search_word:
            needle = search_word.lower()
            providers = [p for p in providers if needle in p.name.lower()]
        return sorted(providers, key=lambda provider: provider.name)

    def get_provider(self, provider_id: UUID) -> ApiToolProviderSpec:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise ApiToolNotFoundError("Provider does not exist.") from exc

    def get_tool(self, provider_id: UUID, tool_name: str) -> ApiToolSpec:
        provider = self.get_provider(provider_id)
        try:
            return provider.tools[tool_name]
        except KeyError as exc:
            raise ApiToolNotFoundError("Tool does not exist.") from exc

    def build_tools(self, provider_id: UUID) -> list[ApiRequestTool]:
        provider = self.get_provider(provider_id)
        return [
            ApiRequestTool(self, provider_id, spec)
            for spec in sorted(provider.tools.values(), key=lambda tool: tool.name)
        ]

    async def invoke_tool(
        self,
        provider_id: UUID,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Invoke one registered API tool with model-supplied arguments."""

        provider = self.get_provider(provider_id)
        spec = self.get_tool(provider_id, tool_name)
        request = _build_request(spec, arguments, provider.headers)
        timeout = aiohttp.ClientTimeout(
            total=get_timeout_policies().tools.openapi.timeout_seconds
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(**request) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "json" in content_type.lower():
                    return await response.json()
                return await response.text()

    def _ensure_unique_name(
        self,
        name: str,
        *,
        exclude_provider_id: UUID | None = None,
    ) -> None:
        if not name.strip():
            raise ApiToolError("Provider name is required.")
        for provider in self._providers.values():
            if provider.id != exclude_provider_id and provider.name == name:
                raise ApiToolError(f"Provider name {name!r} already exists.")


def parse_openapi_schema(openapi_schema_str: str) -> OpenAPISchema:
    """Parse and validate the OpenAPI JSON string used by API tools."""

    try:
        data = json.loads(openapi_schema_str.strip())
    except json.JSONDecodeError as exc:
        raise ApiToolError(
            "Provided data must be a valid OpenAPI-compliant JSON string."
        ) from exc

    if not isinstance(data, dict):
        raise ApiToolError("Provided OpenAPI schema must be a JSON object.")

    paths = data.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise ApiToolError("OpenAPI schema must define at least one path.")

    _info = data.get("info")
    info = _info if isinstance(_info, dict) else {}
    servers = data.get("servers") if isinstance(data.get("servers"), list) else []
    server = ""
    if servers and isinstance(servers[0], dict):
        server = str(servers[0].get("url") or "").rstrip("/")

    return OpenAPISchema(
        title=str(info.get("title") or "API provider"),
        description=str(info.get("description") or info.get("title") or ""),
        server=server,
        paths=paths,
        raw=data,
    )


def _build_tool_specs(schema: OpenAPISchema) -> list[ApiToolSpec]:
    tools: list[ApiToolSpec] = []
    seen: set[str] = set()
    for path, path_item in schema.paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            method_lower = method.lower()
            if method_lower not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            name = _operation_name(method_lower, path, operation)
            if name in seen:
                raise ApiToolError(f"Duplicate operationId/tool name {name!r}.")
            seen.add(name)
            body_schema = _request_body_schema(operation)
            parameters = _json_schema_for_operation(operation, body_schema)
            tools.append(
                ApiToolSpec(
                    name=name,
                    description=str(
                        operation.get("description")
                        or operation.get("summary")
                        or f"{method_lower.upper()} {path}"
                    ),
                    url=f"{schema.server}{path}",
                    method=method_lower,
                    parameters=parameters,
                    openapi_parameters=operation.get("parameters") or [],
                    request_body_schema=body_schema,
                )
            )
    if not tools:
        raise ApiToolError("OpenAPI schema does not define any supported operations.")
    return tools


def _json_schema_for_operation(
    operation: dict[str, Any],
    body_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in operation.get("parameters") or []:
        if not isinstance(parameter, dict):
            continue
        name = parameter.get("name")
        if not name:
            continue
        prop = dict(parameter.get("schema") or {"type": "string"})
        if description := parameter.get("description"):
            prop["description"] = description
        properties[name] = prop
        if parameter.get("required"):
            required.append(name)

    if body_schema is None:
        body_schema = _request_body_schema(operation)
    if body_schema:
        if body_schema.get("type") == "object":
            properties.update(body_schema.get("properties", {}))
            required.extend(body_schema.get("required", []))
        else:
            properties["body"] = body_schema
            required.append("body")

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = sorted(set(required))
    return schema


def _request_body_schema(operation: dict[str, Any]) -> dict[str, Any] | None:
    request_body = operation.get("requestBody")
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content")
    if not isinstance(content, dict):
        return None
    json_content = content.get("application/json")
    if not isinstance(json_content, dict):
        return None
    schema = json_content.get("schema")
    return schema if isinstance(schema, dict) else None


def _build_request(
    spec: ApiToolSpec,
    arguments: dict[str, Any],
    provider_headers: dict[str, str],
) -> dict[str, Any]:
    url = spec.url
    query_params: dict[str, Any] = {}
    headers = dict(provider_headers)
    consumed: set[str] = set()

    for parameter in spec.openapi_parameters:
        if not isinstance(parameter, dict):
            continue
        name = parameter.get("name")
        location = parameter.get("in")
        if not name or name not in arguments:
            continue
        value = arguments[name]
        consumed.add(name)
        if location == "path":
            url = url.replace("{" + name + "}", str(value))
        elif location == "query":
            query_params[name] = value
        elif location == "header":
            headers[name] = str(value)

    for name in _PATH_PARAM_RE.findall(url):
        if name not in arguments:
            raise ApiToolError(f"Missing required path parameter {name!r}.")
        url = url.replace("{" + name + "}", str(arguments[name]))
        consumed.add(name)

    json_body = _json_body_from_arguments(spec, arguments, consumed)
    request: dict[str, Any] = {
        "method": spec.method.upper(),
        "url": url,
        "headers": headers,
    }
    if query_params:
        request["params"] = query_params
    if json_body is not None and spec.method in {"post", "put", "patch", "delete"}:
        request["json"] = json_body
    return request


def _json_body_from_arguments(
    spec: ApiToolSpec,
    arguments: dict[str, Any],
    consumed: set[str],
) -> Any:
    if not spec.request_body_schema:
        return None
    if "body" in arguments:
        return arguments["body"]
    schema = spec.request_body_schema
    if schema.get("type") == "object":
        property_names = set((schema.get("properties") or {}).keys())
        return {
            name: value
            for name, value in arguments.items()
            if name in property_names and name not in consumed
        }
    return None


def _operation_name(method: str, path: str, operation: dict[str, Any]) -> str:
    operation_id = operation.get("operationId")
    if operation_id:
        return _sanitize_name(str(operation_id))
    return _sanitize_name(f"{method}_{path.strip('/') or 'root'}")


def _sanitize_name(value: str) -> str:
    normalized = _NAME_RE.sub("_", value).strip("_")
    return normalized or "api_tool"


def _format_tool_response(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False, indent=2)
