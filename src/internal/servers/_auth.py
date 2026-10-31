"""Shared FastAPI dependency factory for admin-only endpoints."""

from __future__ import annotations

import ipaddress

from fastapi import HTTPException, Request

from src.internal.auth import AuthenticatedUser, user_from_headers
from src.internal.configs import AppSettings


def is_admin_user(user: AuthenticatedUser, app_settings: AppSettings) -> bool:
    """Evaluate admin privileges after the caller's account state is resolved."""
    super_users = app_settings.auth.super_users
    return not user.is_anonymous and (
        user.metadata.get("role") == "admin"
        or user.id in super_users
        or (user.email is not None and user.email in super_users)
    )


def make_require_admin(app_settings: AppSettings):
    """Return a FastAPI dependency that enforces super-user access.

    Active web accounts are admin if listed in configured super_users or their
    current stored role is admin. Standalone services without a user store use
    the signed role claim instead.
    """

    def _require_admin(request: Request) -> AuthenticatedUser:
        if app_settings.auth.dev_admin_bypass:
            # Dev-only shortcut: no token needed so the local admin dashboard
            # works out of the box. Gated by AGENTIC_SEARCH_DEV_ADMIN (default
            # off); a startup warning fires when it is on.
            return AuthenticatedUser(
                id="dev-admin",
                email="dev-admin@localhost",
                is_anonymous=False,
                metadata={"role": "admin"},
            )
        user = user_from_headers(request.headers)
        # The web app owns a user store: deletion, deactivation and role changes
        # take effect immediately, even when a previously issued JWT is valid.
        # Standalone service routers can still use configured stateless tokens.
        store = getattr(request.app.state, "auth_store", None)
        if store is not None:
            from src.internal.servers.users.api import resolve_active_user

            user = resolve_active_user(request, store)
        if user is None or user.is_anonymous:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if not is_admin_user(user, app_settings):
            raise HTTPException(status_code=403, detail="Admin access required.")
        return user

    return _require_admin


def caller_may_use_session(session, caller) -> bool:
    """May *caller* read, rename, delete or continue *session*?

    Shared because the same question is asked from two routers that cannot
    import each other: the `/api/sessions` surface in `servers/web/app.py` and
    the `/chat/*` surface in `query_and_chat/chat_backend.py`. Each had its own
    copy of these endpoints, and each shipped them unguarded -- the second was
    still live after the first was fixed, precisely because the check lived in
    only one of them.

    Two cases, deliberately different:

    - **An owned session** (``user_id`` set) is usable only by that user.
    - **An anonymous session** (``user_id`` is ``NULL``) stays usable by anyone
      holding its id. There is no identity to compare against for signed-out
      callers, so the id is the only capability there is, and the CLI and local
      research flows depend on it. Per-caller anonymous identity is separate,
      unbuilt work.

    Session ids are ``session_<uuid4hex>``, so this is not brute-forceable; what
    it closes is an IDOR, where a leaked id became someone else's transcript --
    or, on the `/chat/*` surface, their deleted transcript.
    """
    if session.user_id is None:
        return True
    return caller is not None and caller.id == session.user_id


__all__ = ["make_require_admin", "is_admin_user", "caller_may_use_session"]


def require_loopback_client(request: Request) -> None:
    """Admit only callers on this host, for routes that are internal by design.

    The in-process retrieval routes stand in for a retrieval server bound to
    127.0.0.1, and /retrieve trusts caller-supplied ACL filters: an outside
    caller who omits them would read every document. A same-host reverse proxy
    that forwards these paths would defeat this check, so never proxy them.
    """
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        address = getattr(address, "ipv4_mapped", None) or address
    if address is None or not address.is_loopback:
        raise HTTPException(status_code=403, detail="Internal endpoint")
