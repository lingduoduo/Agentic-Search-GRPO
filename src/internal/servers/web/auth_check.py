"""Fail startup when a route is neither explicitly public nor detectably guarded.

Dependency inspection is structural. Inline-call inspection is a source heuristic,
not a proof of control flow: it recognizes calls to reviewed guard functions, not
comments, similarly named functions, or optional identity resolvers. Runtime
ownership and authorization regression tests remain necessary.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Exact path/method exemptions, with the reason beside each group. "Public"
# permits anonymous access; session ownership and optional auth still apply.
PUBLIC_ENDPOINT_SPECS: list[tuple[str, set[str]]] = [
    # Health probes and the signed-out application shell/assets.
    ("/health", {"GET"}),
    ("/", {"GET"}),
    ("/assist", {"GET"}),
    ("/search", {"GET"}),
    ("/chat", {"GET"}),
    ("/tools", {"GET"}),
    ("/assets/app.css", {"GET"}),
    ("/assets/app.js", {"GET"}),
    ("/assets", {"GET", "HEAD"}),
    # FastAPI's API documentation, including its OAuth redirect page.
    ("/openapi.json", {"GET", "HEAD"}),
    ("/docs", {"GET", "HEAD"}),
    ("/docs/oauth2-redirect", {"GET", "HEAD"}),
    ("/redoc", {"GET", "HEAD"}),
    # Credentials are established here; login verifies the supplied password.
    ("/auth/register", {"POST"}),
    ("/auth/login", {"POST"}),
    # Local research supports anonymous conversations. Existing owned sessions
    # still require their owner; history lists are empty for anonymous callers.
    ("/api/sessions", {"POST"}),
    ("/api/agent", {"POST"}),
    ("/api/agent/stream", {"POST"}),
    ("/chat/create-chat-session", {"POST"}),
    ("/chat/get-user-chat-sessions", {"GET"}),
    ("/chat/send-chat-message", {"POST"}),
    ("/tool/send-tool-message", {"POST"}),
    ("/tool/tool-history", {"GET"}),
    ("/search/search-history", {"GET"}),
    # Classification accepts only caller-supplied text, not stored documents.
    ("/search/search-flow-classification", {"POST"}),
    # Shared anonymous memory is intentional unless MEMORY_REQUIRE_AUTH is on.
    # Each handler resolves its bucket and enforces that deployment switch.
    ("/api/memory/save", {"POST"}),
    ("/api/memory/list", {"GET"}),
    ("/api/memory/search", {"POST"}),
    ("/api/memory/consolidate", {"POST"}),
    ("/api/memory/profile", {"GET"}),
    ("/api/memory/profile/generate", {"POST"}),
    ("/api/memory/curate", {"POST"}),
    # Branding and feature availability are needed before login.
    ("/settings", {"GET"}),
    ("/enterprise-settings", {"GET"}),
    ("/enterprise-settings/logo", {"GET"}),
    ("/enterprise-settings/logotype", {"GET"}),
    ("/enterprise-settings/custom-analytics-script", {"GET"}),
    # Public discovery remains reachable before an IdP supplies credentials.
    ("/scim/v2/ServiceProviderConfig", {"GET"}),
    ("/scim/v2/ResourceTypes", {"GET"}),
    ("/scim/v2/Schemas", {"GET"}),
    # The Stripe publishable key is deliberately public, never a secret key.
    ("/admin/billing/stripe-publishable-key", {"GET"}),
    # These self-hosted compatibility stubs always return 501, with no data or
    # mutation. Implementing a stub requires revisiting its auth classification.
    ("/query/standard-answer", {"GET"}),
    ("/tenants/billing-information", {"GET"}),
    ("/tenants/create-subscription-session", {"POST"}),
    ("/tenants/create-customer-portal-session", {"POST"}),
    ("/tenants/stripe-publishable-key", {"GET"}),
    ("/enterprise-settings/refresh-token", {"POST"}),
    # Only registered in explicit INTEGRATION_TESTS_MODE, for fixture reset.
    ("/manage/admin/reset-test-data", {"POST"}),
]

EE_PUBLIC_ENDPOINT_SPECS = PUBLIC_ENDPOINT_SPECS + [
    # External proxy deployments authenticate these using licenses.
    ("/proxy/create-checkout-session", {"POST"}),
    ("/proxy/claim-license", {"POST"}),
    ("/proxy/create-customer-portal-session", {"POST"}),
    ("/proxy/billing-information", {"GET"}),
]

# Module plus qualified name prevents a random `_require_admin` from being
# mistaken for the real guard. Keep this small and review additions as auth code.
_AUTH_DEPENDENCIES = {
    ("src.internal.servers._auth", "make_require_admin.<locals>._require_admin"),
    ("src.internal.servers.scim.api", "create_scim_router.<locals>._auth"),
    (
        "src.internal.servers.user_group.api",
        "create_user_group_router.<locals>._require_user",
    ),
}
_INLINE_GUARDS = _AUTH_DEPENDENCIES | {
    ("src.internal.servers.users.api", "_require_auth"),
    ("src.internal.servers.users.api", "_require_admin_role"),
    ("src.internal.servers._auth", "caller_may_use_session"),
    (
        "src.internal.servers.query_and_chat.chat_backend",
        "create_chat_router.<locals>._session_or_404",
    ),
    (
        "src.internal.servers.query_and_chat.search_backend",
        "_authenticated_search_filters",
    ),
}


def _recognized(call, guards) -> bool:
    return (
        getattr(call, "__module__", None),
        getattr(call, "__qualname__", None),
    ) in guards


def _has_auth_dependency(dependant) -> bool:
    return _recognized(dependant.call, _AUTH_DEPENDENCIES) or any(
        _has_auth_dependency(child) for child in dependant.dependencies
    )


def _has_inline_guard(endpoint) -> bool:
    try:
        source = ast.parse(textwrap.dedent(inspect.getsource(endpoint)))
        closure = inspect.getclosurevars(endpoint)
    except (OSError, TypeError, SyntaxError):
        return False  # Source unavailable: declare a dependency instead.
    namespace = {**closure.globals, **closure.nonlocals}
    function = source.body[0]
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    # Skip decorators/default arguments: only calls in the handler body count.
    pending = list(function.body)
    while pending:
        node = pending.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue  # Defining a nested guard does not execute it.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if _recognized(namespace.get(node.func.id), _INLINE_GUARDS):
                return True
        pending.extend(ast.iter_child_nodes(node))
    return False


def check_router_auth(
    application: FastAPI,
    public_endpoint_specs: list[tuple[str, set[str]]] = PUBLIC_ENDPOINT_SPECS,
) -> None:
    """Reject unclassified routes, including nested mounts and individual methods.

    Public GET never exempts POST at the same path. Every route registration is
    checked independently, so a guarded duplicate cannot conceal a public one.
    Only explicitly listed StaticFiles mounts are exempted as static assets.
    """
    public: dict[str, set[str]] = {}
    for path, methods in public_endpoint_specs:
        public.setdefault(path, set()).update(methods)
    registered: dict[str, set[str]] = {}
    unguarded: list[str] = []

    def visit(routes, prefix=""):
        for route in routes:
            path = prefix + getattr(route, "path", "")
            if isinstance(route, Mount):
                if isinstance(route.app, StaticFiles):
                    methods = {"GET", "HEAD"}
                elif route.routes:
                    visit(route.routes, path)
                    continue
                else:
                    methods = {"ASGI"}
            elif isinstance(route, Route):
                methods = route.methods or set()
            elif isinstance(route, WebSocketRoute):
                methods = {"WEBSOCKET"}
            else:
                unguarded.append(f"UNKNOWN {path}")
                continue
            registered.setdefault(path, set()).update(methods)
            guarded = isinstance(route, APIRoute) and (
                _has_auth_dependency(route.dependant)
                or _has_inline_guard(route.endpoint)
            )
            for method in sorted(methods):
                if method in public.get(path, set()):
                    logger.info("[auth_check] public %s %s", method, path)
                elif guarded:
                    logger.debug("[auth_check] guarded %s %s", method, path)
                else:
                    unguarded.append(f"{method} {path}")

    visit(application.routes)
    if unguarded:
        raise RuntimeError(
            "Routes missing authentication or public classification: "
            + ", ".join(sorted(unguarded))
        )
    for path, methods in public.items():
        missing = methods - registered.get(path, set())
        if missing:
            logger.warning(
                "[auth_check] declared public but not registered: %s %s",
                sorted(missing),
                path,
            )


def check_ee_router_auth(
    application: FastAPI,
    public_endpoint_specs: list[tuple[str, set[str]]] = EE_PUBLIC_ENDPOINT_SPECS,
) -> None:
    """Audit with the additional license-authenticated proxy exemptions."""
    check_router_auth(application, public_endpoint_specs)


__all__ = [
    "EE_PUBLIC_ENDPOINT_SPECS",
    "PUBLIC_ENDPOINT_SPECS",
    "check_ee_router_auth",
    "check_router_auth",
]
