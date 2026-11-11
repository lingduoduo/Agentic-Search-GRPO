"""Tool management REST API.

Endpoints (prefix /admin/tools):

  GET    /                      — list all registered tools
  GET    /providers             — tools grouped by provider, one page at a time
  POST   /openapi               — register tools from an OpenAPI JSON body or URL
  POST   /openapi/validate      — the tools a schema would register, or why not
  GET    /openapi/{provider_id} — an OpenAPI provider's stored configuration
  PUT    /openapi/{provider_id} — replace an OpenAPI provider's schema/name/headers
  DELETE /openapi/{provider_id} — remove all tools from an OpenAPI provider
  GET    /{name}                — get one tool's full schema
  POST   /{name}/invoke         — test-invoke a tool with supplied arguments
"""

from __future__ import annotations

import logging
import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from src.internal.auth import AuthenticatedUser
from src.internal.configs import AppSettings
from src.internal.servers._auth import make_require_admin
from src.internal.tools.api import (
    ApiToolError,
    ApiToolNotFoundError,
    preview_openapi_schema,
)
from src.internal.tools.registry import tool_registry

logger = logging.getLogger(__name__)


def _try_sync_to_mcp(name: str) -> None:
    """Optionally sync a newly registered tool to the running MCP server."""
    try:
        from src.internal.mcp_server.tools.dynamic import sync_tool_to_mcp

        sync_tool_to_mcp(name)
    except Exception as exc:
        logger.debug("Could not sync tool %r to MCP: %s", name, exc)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ToolView(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    source: str
    provider_id: str | None
    # What to show for the provider: the registered OpenAPI provider name (its
    # provider_id is a UUID) or the MCP server name; None for function tools.
    provider_name: str | None = None
    # The OpenAPI schema's info description; None for function and MCP tools.
    provider_description: str | None = None
    # Both come from ToolEntry and were being dropped here: all_summaries()
    # returns them, but a field absent from the model is silently discarded, so
    # no caller could tell a registered-but-not-agent-offered tool from a normal
    # one. That distinction is the usual answer to "why did the agent ignore it".
    agent_callable: bool = True
    user_scoped: bool = False
    effect: str = "unspecified"
    result_kind: str | None = None
    citeable: bool = False
    retries_internally: bool = False


class OpenAPIRegisterRequest(BaseModel):
    name: str = Field(..., description="Human-readable name for this API provider")
    openapi_json: str = Field(..., description="OpenAPI 3.x JSON schema as a string")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers to include in every API call (e.g. Authorization)",
    )


class OpenAPIRegisterResponse(BaseModel):
    provider_id: str
    tool_names: list[str]


class OpenAPIValidateRequest(BaseModel):
    openapi_json: str


class OpenAPIOperation(BaseModel):
    name: str
    description: str
    method: str
    path: str


class OpenAPIValidateResponse(BaseModel):
    tools: list[OpenAPIOperation]


class OpenAPIProviderDetail(BaseModel):
    provider_id: str
    name: str
    description: str
    openapi_json: str
    headers: dict[str, str]


class ToolProviderView(BaseModel):
    """The tools from one place: an OpenAPI provider, an MCP server, or
    ``local`` (built-in function tools)."""

    id: str
    name: str
    description: str
    source: str
    tools: list[ToolView]


class Paginator(BaseModel):
    current_page: int
    page_size: int
    total_page: int
    total_record: int


class ToolProviderPage(BaseModel):
    list: list[ToolProviderView]
    paginator: Paginator


class DiscoverRequest(BaseModel):
    query: str = Field(..., description="Query to rank the registered tools against")


class InvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class InvokeResponse(BaseModel):
    response: str
    raw: Any
    errors: list[str]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_tools_router(settings: AppSettings) -> APIRouter:
    router = APIRouter(prefix="/admin/tools", tags=["tools"])
    require_admin = make_require_admin(settings)

    @router.get("", response_model=list[ToolView])
    async def list_tools(
        _: AuthenticatedUser = Depends(require_admin),
    ) -> list[ToolView]:
        return [ToolView(**s) for s in tool_registry.all_summaries()]

    # Before /{name}, or "providers" is matched as a tool name.
    @router.get("/providers", response_model=ToolProviderPage)
    async def list_providers(
        current_page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        search_word: str = "",
        _: AuthenticatedUser = Depends(require_admin),
    ) -> ToolProviderPage:
        providers = tool_registry.provider_summaries(search_word)
        start = (current_page - 1) * page_size
        return ToolProviderPage(
            list=[ToolProviderView(**p) for p in providers[start : start + page_size]],
            paginator=Paginator(
                current_page=current_page,
                page_size=page_size,
                total_page=math.ceil(len(providers) / page_size),
                total_record=len(providers),
            ),
        )

    # Must be declared before /{name} so FastAPI doesn't match "openapi" as a name
    @router.post(
        "/openapi",
        response_model=OpenAPIRegisterResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def register_openapi(
        req: OpenAPIRegisterRequest,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> OpenAPIRegisterResponse:
        try:
            tool_names = tool_registry.register_from_openapi(
                req.openapi_json,
                name=req.name,
                headers=req.headers or None,
            )
        except ApiToolError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except ValueError as exc:
            # A contract violation raised by ToolRegistry.register() itself
            # (strict mode), not a malformed spec — distinct from ApiToolError.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )

        # Find the provider_id by looking up the first registered tool
        provider_id = ""
        if tool_names:
            entry = tool_registry._entries.get(tool_names[0])
            if entry:
                provider_id = entry.provider_id or ""

        for name in tool_names:
            _try_sync_to_mcp(name)

        return OpenAPIRegisterResponse(provider_id=provider_id, tool_names=tool_names)

    @router.post("/openapi/validate", response_model=OpenAPIValidateResponse)
    async def validate_openapi(
        req: OpenAPIValidateRequest,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> OpenAPIValidateResponse:
        try:
            tools = preview_openapi_schema(req.openapi_json)
        except ApiToolError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        return OpenAPIValidateResponse(tools=[OpenAPIOperation(**t) for t in tools])

    @router.get("/openapi/{provider_id}", response_model=OpenAPIProviderDetail)
    async def get_openapi_provider(
        provider_id: str,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> OpenAPIProviderDetail:
        try:
            provider = tool_registry.get_openapi_provider(provider_id)
        except ApiToolNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {provider_id!r} not found.",
            )
        return OpenAPIProviderDetail(
            provider_id=str(provider.id),
            name=provider.name,
            description=provider.description,
            openapi_json=provider.openapi_schema,
            headers=provider.headers,
        )

    @router.put("/openapi/{provider_id}", response_model=OpenAPIRegisterResponse)
    async def update_openapi_provider(
        provider_id: str,
        req: OpenAPIRegisterRequest,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> OpenAPIRegisterResponse:
        try:
            tool_names = tool_registry.update_from_openapi(
                provider_id, req.openapi_json, name=req.name, headers=req.headers
            )
        except ApiToolNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {provider_id!r} not found.",
            )
        except ApiToolError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        except ValueError as exc:
            # A contract violation raised by ToolRegistry.register() (strict mode).
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            )
        for name in tool_names:
            _try_sync_to_mcp(name)
        return OpenAPIRegisterResponse(provider_id=provider_id, tool_names=tool_names)

    @router.delete(
        "/openapi/{provider_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        response_class=Response,
        response_model=None,
    )
    async def delete_openapi_provider(
        provider_id: str,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> None:
        removed = tool_registry.unregister_provider(provider_id)
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {provider_id!r} not found.",
            )

    # Also before /{name}, or "discover" is matched as a tool name.
    @router.post("/discover")
    async def discover(
        req: DiscoverRequest,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> dict:
        """Rank the registered tools against *query* (TF-IDF, no LLM).

        The same ranking as ``POST /api/debug/tools/discover``, but reachable
        without the debug router — that one is only mounted when
        AGENTIC_SEARCH_DEBUG_PANELS is set, so it does not exist in a normal
        deployment. Imported in-handler to keep the semantic router off this
        module's import path.
        """
        from src.internal.tools.semantic_router import (
            SemanticRouter,
            default_tool_catalog,
        )

        return SemanticRouter(default_tool_catalog()).get_routing_details(req.query)

    @router.get("/{name}", response_model=ToolView)
    async def get_tool(
        name: str,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> ToolView:
        summary = tool_registry.tool_summary(name)
        if summary is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {name!r} not found.",
            )
        return ToolView(**summary)

    @router.post("/{name}/invoke", response_model=InvokeResponse)
    async def invoke_tool(
        name: str,
        req: InvokeRequest,
        _: AuthenticatedUser = Depends(require_admin),
    ) -> InvokeResponse:
        if tool_registry.get(name) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {name!r} not found.",
            )
        result = await tool_registry.invoke_detailed(name, req.arguments)
        errors = result.errors
        if result.failure is not None and not result.response and not errors:
            # The tool raised: report it instead of a bare 500.
            errors = [f"{result.failure.category.value}: {result.failure.message}"]
        return InvokeResponse(response=result.response, raw=result.raw, errors=errors)

    return router
