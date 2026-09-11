"""User auth router for local development and integration tests.

Provides the endpoints that the integration test suite expects from a
full Danswer-style backend, implemented against AgenticSearchStore (SQLite).

Endpoints
---------
POST  /auth/register                   – register; first user becomes admin
POST  /auth/login                      – login, set fastapiusersauth cookie
GET   /me                              – current user info
GET   /me/permissions                  – current user permissions
PATCH /manage/set-user-role            – change a user's role (admin only)
PATCH /manage/admin/activate-user      – activate user (admin only)
PATCH /manage/admin/deactivate-user    – deactivate user (admin only)
GET   /manage/users/accepted           – paginated accepted user list (admin only)
GET   /manage/users/invited            – invited users stub (always empty)
PUT   /manage/admin/users              – invite users stub (no-op)
POST  /manage/admin/reset-test-data    – clear all users (integration test mode only)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from src.internal.auth import (
    AuthenticatedUser,
    generate_user_jwt_token,
    user_from_headers,
)
from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.db.models import UserRecord
from src.internal.servers._auth import is_admin_user

_COOKIE_NAME = "fastapiusersauth"
_TOKEN_TTL = 86400 * 7  # 7 days
_PASSWORD_ITERATIONS = 600_000


# ---------------------------------------------------------------------------
# Password helpers (stdlib only)
# ---------------------------------------------------------------------------


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${_PASSWORD_ITERATIONS}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if not isinstance(stored_hash, str):
        return False
    try:
        if "$" not in stored_hash:
            # Legacy fixed-salt hashes are upgraded after successful login.
            expected = bytes.fromhex(stored_hash)
            salt, iterations = b"agentic-search", 100_000
        else:
            algorithm, count, encoded_salt, encoded_hash = stored_hash.split("$")
            if algorithm != "pbkdf2_sha256" or count != str(_PASSWORD_ITERATIONS):
                return False
            salt, expected = bytes.fromhex(encoded_salt), bytes.fromhex(encoded_hash)
            if len(salt) != 16:
                return False
            iterations = _PASSWORD_ITERATIONS
        if len(expected) != 32:
            return False
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return hmac.compare_digest(actual, expected)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def resolve_request_user(request: Request) -> AuthenticatedUser | None:
    """Resolve a caller exactly as user-facing identity endpoints do.

    Keep this as the shared authentication boundary for endpoints that must
    accept every credential type supported by ``/me``.
    """
    user = user_from_headers(request.headers)
    if user:
        return user
    if "authorization" in request.headers:
        return None  # An explicitly supplied invalid bearer cannot borrow a cookie identity.
    cookie_val = request.cookies.get(_COOKIE_NAME)
    if cookie_val:
        try:
            from src.internal.auth import user_from_jwt_token

            return user_from_jwt_token(cookie_val)
        except ValueError:
            pass
    return None


def resolve_active_user(
    request: Request, store: AgenticSearchStore
) -> AuthenticatedUser | None:
    """Resolve an active stored caller and refresh their email and role.

    A JWT stays valid until it expires, so it outlives the row it names whenever
    the users table is rebuilt — which the dev default (`:memory:`) does on every
    restart. Such a token is worse than no token at all: the id it carries fails
    the ``chat_sessions.user_id`` foreign key, and re-registering the same email
    under a fresh id makes any auto-provisioning collide with ``users.email``
    UNIQUE. Both surface as a 500 on endpoints that persist a user id, while
    read-only endpoints keep working — the same cookie appearing to succeed and
    fail at once.

    The store is the source of truth: a token with no active matching row is
    treated as unauthenticated. Role and email changes take effect immediately.
    """
    user = resolve_request_user(request)
    if user is None or user.is_anonymous:
        return None
    record = store.get_user(user.id)
    if (
        record is None
        or not record.metadata.get("is_active", True)
        or not store.get_user_active(user.id)
    ):
        return None
    return replace(
        user,
        email=record.email,
        metadata={**user.metadata, "role": record.metadata.get("role", "basic")},
    )


def _require_auth(request: Request, store: AgenticSearchStore) -> AuthenticatedUser:
    user = resolve_active_user(request, store)
    if user is None or user.is_anonymous:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def _require_admin_role(
    request: Request, store: AgenticSearchStore, app_settings: AppSettings
) -> AuthenticatedUser:
    user = _require_auth(request, store)
    if not is_admin_user(user, app_settings):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str
    username: str | None = None
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    is_active: bool


class SetRoleRequest(BaseModel):
    user_email: str
    new_role: str
    explicit_override: bool = False


class SetStatusRequest(BaseModel):
    user_email: str


class InviteRequest(BaseModel):
    emails: list[str]


class PaginatedUsers(BaseModel):
    items: list[UserResponse]
    total_items: int


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_users_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    router = APIRouter(tags=["users"])
    integration_mode = os.getenv("INTEGRATION_TESTS_MODE", "").lower() == "true"

    def _get_user_record_by_email(email: str) -> UserRecord:
        all_users = store.list_users()
        for u in all_users:
            if u.email and u.email.lower() == email.lower():
                return u
        raise HTTPException(status_code=404, detail=f"User {email!r} not found.")

    def _user_response(record: UserRecord) -> UserResponse:
        return UserResponse(
            id=record.id,
            email=record.email or "",
            role=record.metadata.get("role", "basic"),
            is_active=bool(record.metadata.get("is_active", True))
            and store.get_user_active(record.id),
        )

    def _issue_token(record: UserRecord) -> str:
        return generate_user_jwt_token(
            user_id=record.id,
            email=record.email,
            expires_in_seconds=_TOKEN_TTL,
            extra={"role": record.metadata.get("role", "basic")},
        )

    # ------------------------------------------------------------------
    # POST /auth/register
    # ------------------------------------------------------------------
    @router.post("/auth/register")
    def register(body: RegisterRequest) -> UserResponse:
        email = body.email.lower().strip()
        password_hash = _hash_password(body.password)
        try:
            record = store.register_user(
                email=email,
                name=body.username or email,
                password_hash=password_hash,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _user_response(record)

    # ------------------------------------------------------------------
    # POST /auth/login  (form-encoded: username + password)
    # ------------------------------------------------------------------
    @router.post("/auth/login")
    def login(
        response: Response,
        form: Annotated[OAuth2PasswordRequestForm, Depends()],
    ) -> UserResponse:
        email = form.username.lower().strip()
        try:
            record = _get_user_record_by_email(email)
        except HTTPException:
            raise HTTPException(status_code=400, detail="Invalid credentials.")

        if not _verify_password(
            form.password, record.metadata.get("password_hash", "")
        ):
            raise HTTPException(status_code=400, detail="Invalid credentials.")

        if not _user_response(record).is_active:
            raise HTTPException(status_code=403, detail="Account deactivated.")

        stored_hash = record.metadata["password_hash"]
        if "$" not in stored_hash:
            store.upgrade_password_hash(
                record.id, stored_hash, _hash_password(form.password)
            )

        token = _issue_token(record)
        response.set_cookie(
            key=_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            max_age=_TOKEN_TTL,
        )
        return _user_response(record)

    # ------------------------------------------------------------------
    # GET /me
    # ------------------------------------------------------------------
    @router.get("/me")
    def me(request: Request) -> UserResponse:
        user = _require_auth(request, store)
        record = store.get_user(user.id)
        if record is None:
            raise HTTPException(status_code=404, detail="User not found.")
        return _user_response(record)

    # ------------------------------------------------------------------
    # GET /me/permissions
    # ------------------------------------------------------------------
    @router.get("/me/permissions")
    def me_permissions(request: Request) -> list[str]:
        user = _require_auth(request, store)
        if is_admin_user(user, app_settings):
            return ["basic_access", "full_admin_panel_access"]
        return ["basic_access"]

    # ------------------------------------------------------------------
    # PATCH /manage/set-user-role
    # ------------------------------------------------------------------
    @router.patch("/manage/set-user-role")
    def set_user_role(body: SetRoleRequest, request: Request) -> UserResponse:
        _require_admin_role(request, store, app_settings)
        record = _get_user_record_by_email(body.user_email)
        updated = store.upsert_user(
            UserRecord(
                id=record.id,
                email=record.email,
                name=record.name,
                metadata={**record.metadata, "role": body.new_role},
            )
        )
        return _user_response(updated)

    # ------------------------------------------------------------------
    # PATCH /manage/admin/activate-user
    # PATCH /manage/admin/deactivate-user
    # ------------------------------------------------------------------
    @router.patch("/manage/admin/activate-user")
    def activate_user(body: SetStatusRequest, request: Request) -> UserResponse:
        _require_admin_role(request, store, app_settings)
        record = _get_user_record_by_email(body.user_email)
        store.set_user_active(record.id, True)
        updated = store.upsert_user(
            UserRecord(
                id=record.id,
                email=record.email,
                name=record.name,
                metadata={**record.metadata, "is_active": True},
            )
        )
        return _user_response(updated)

    @router.patch("/manage/admin/deactivate-user")
    def deactivate_user(body: SetStatusRequest, request: Request) -> UserResponse:
        _require_admin_role(request, store, app_settings)
        record = _get_user_record_by_email(body.user_email)
        store.set_user_active(record.id, False)
        updated = store.upsert_user(
            UserRecord(
                id=record.id,
                email=record.email,
                name=record.name,
                metadata={**record.metadata, "is_active": False},
            )
        )
        return _user_response(updated)

    # ------------------------------------------------------------------
    # GET /manage/users/accepted
    # ------------------------------------------------------------------
    @router.get("/manage/users/accepted")
    def list_accepted_users(
        request: Request,
        page_num: int = 0,
        page_size: int = 10,
        q: str | None = None,
        roles: list[str] | None = None,
        is_active: bool | None = None,
    ) -> PaginatedUsers:
        _require_admin_role(request, store, app_settings)
        all_users = store.list_users()
        filtered = [
            u
            for u in all_users
            if (q is None or (u.email and q.lower() in u.email.lower()))
            and (roles is None or u.metadata.get("role") in roles)
            and (is_active is None or _user_response(u).is_active == is_active)
        ]
        start = page_num * page_size
        page = filtered[start : start + page_size]
        return PaginatedUsers(
            items=[_user_response(u) for u in page],
            total_items=len(filtered),
        )

    # ------------------------------------------------------------------
    # GET /manage/users/invited  (stub)
    # PUT /manage/admin/users    (stub)
    # ------------------------------------------------------------------
    @router.get("/manage/users/invited")
    def list_invited_users(request: Request) -> list[dict]:
        _require_admin_role(request, store, app_settings)
        return []

    @router.put("/manage/admin/users")
    def invite_users(body: InviteRequest, request: Request) -> dict:
        _require_admin_role(request, store, app_settings)
        return {"invited": body.emails}

    # ------------------------------------------------------------------
    # POST /manage/admin/reset-test-data  (integration test mode only)
    # ------------------------------------------------------------------
    if integration_mode:

        @router.post("/manage/admin/reset-test-data")
        def reset_test_data() -> dict:
            # No auth required — only registered when INTEGRATION_TESTS_MODE=true.
            for user in store.list_users():
                store.delete_user(user.id)
            return {"status": "ok"}

    return router
