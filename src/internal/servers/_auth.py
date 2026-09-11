"""Shared FastAPI dependency factory for admin-only endpoints."""

from __future__ import annotations

from fastapi import HTTPException, Request

from src.internal.auth import AuthenticatedUser, user_from_headers
from src.internal.configs import AppSettings


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
        super_users = app_settings.auth.super_users
        in_super_users = user.id in super_users or (
            user.email is not None and user.email in super_users
        )
        has_admin_role = user.metadata.get("role") == "admin"
        if not in_super_users and not has_admin_role:
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


__all__ = ["make_require_admin", "caller_may_use_session"]
