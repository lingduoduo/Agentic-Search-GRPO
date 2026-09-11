"""Small auth helpers for Agentic Search services.

The repo does not depend on a full identity provider, but several server paths
need a consistent way to represent the caller and pass it into ACL filters.  The
JWT helpers below intentionally use only the Python standard library so local
examples and tests do not need an extra auth package.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.internal.configs import load_app_settings

AUTH_SECRET_ENV = "AGENTIC_SEARCH_AUTH_SECRET"
JWT_ALGORITHM = "HS256"


@dataclass(frozen=True)
class AuthenticatedUser:
    """Caller identity used by web/API handlers and access filters."""

    id: str
    email: str | None = None
    group_ids: frozenset[str] = frozenset()
    tenant_id: str | None = None
    is_anonymous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


def get_auth_secret(secret: str | None = None) -> str:
    return secret or os.getenv(AUTH_SECRET_ENV) or load_app_settings().auth.secret


def generate_user_jwt_token(
    *,
    user_id: str,
    email: str | None = None,
    group_ids: Iterable[str] | None = None,
    tenant_id: str | None = None,
    expires_in_seconds: int | None = None,
    secret: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create a signed HS256 token for local API clients and tests."""

    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user_id,
        "iat": now,
    }
    if email:
        payload["email"] = email
    if group_ids:
        payload["groups"] = sorted(set(group_ids))
    if tenant_id:
        payload["tenant_id"] = tenant_id
    if expires_in_seconds is not None:
        payload["exp"] = now + expires_in_seconds
    if extra:
        payload.update(extra)
    return _encode_jwt(payload, get_auth_secret(secret))


def decode_user_jwt_token(token: str, *, secret: str | None = None) -> dict[str, Any]:
    """Verify and decode a local HS256 JWT."""

    payload = _decode_jwt(token, get_auth_secret(secret))
    for claim in ("exp", "nbf", "iat"):
        if claim in payload:
            value = payload[claim]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"JWT {claim} must be a numeric date.")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"JWT {claim} must be finite.")
    now = time.time()
    expires_at = payload.get("exp")
    if expires_at is not None and expires_at <= now:
        raise ValueError("JWT has expired.")
    if payload.get("nbf", now) > now:
        raise ValueError("JWT is not yet valid.")
    return payload


def user_from_jwt_token(
    token: str,
    *,
    secret: str | None = None,
) -> AuthenticatedUser:
    payload = decode_user_jwt_token(token, secret=secret)
    user_id = payload.get("sub") or payload.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("JWT subject is required.")
    groups = payload.get("groups") or payload.get("group_ids") or []
    if not isinstance(groups, list) or any(
        not isinstance(group, str) for group in groups
    ):
        raise ValueError("JWT groups must be a list of strings.")
    for claim in ("email", "tenant_id"):
        if payload.get(claim) is not None and not isinstance(payload[claim], str):
            raise ValueError(f"JWT {claim} must be a string.")
    return AuthenticatedUser(
        id=user_id,
        email=payload.get("email"),
        group_ids=frozenset(str(group) for group in groups),
        tenant_id=payload.get("tenant_id"),
        is_anonymous=False,
        metadata={k: v for k, v in payload.items() if k not in _REGISTERED_CLAIMS},
    )


def anonymous_user(tenant_id: str | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(id="anonymous", tenant_id=tenant_id, is_anonymous=True)


def extract_bearer_token(headers: Mapping[str, str]) -> str | None:
    authorization = headers.get("Authorization") or headers.get("authorization")
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


_COOKIE_NAME = "fastapiusersauth"


def _extract_cookie_token(headers: Mapping[str, str]) -> str | None:
    cookie_header = headers.get("Cookie") or headers.get("cookie") or ""
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name.strip() == _COOKIE_NAME and value.strip():
            return value.strip()
    return None


def user_from_headers(headers: Mapping[str, str]) -> AuthenticatedUser | None:
    token = extract_bearer_token(headers) or _extract_cookie_token(headers)
    if not token:
        return None
    try:
        return user_from_jwt_token(token)
    except ValueError:
        return None


_REGISTERED_CLAIMS = {
    "sub",
    "user_id",
    "email",
    "groups",
    "group_ids",
    "tenant_id",
    "iat",
    "exp",
    "nbf",
}


def _encode_jwt(payload: dict[str, Any], secret: str) -> str:
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    signing_input = ".".join(
        [
            _b64url_json(header),
            _b64url_json(payload),
        ]
    )
    signature = _sign(signing_input.encode("ascii"), secret)
    return f"{signing_input}.{_b64url_encode(signature)}"


def _decode_jwt(token: str, secret: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("JWT must have three segments.")
    encoded_header, encoded_payload, encoded_signature = parts
    header = json.loads(_b64url_decode(encoded_header))
    if not isinstance(header, dict):
        raise ValueError("JWT header must be an object.")
    if header.get("alg") != JWT_ALGORITHM:
        raise ValueError(f"Unsupported JWT algorithm: {header.get('alg')!r}.")

    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    expected = _sign(signing_input, secret)
    actual = _b64url_decode(encoded_signature)
    if not hmac.compare_digest(expected, actual):
        raise ValueError("JWT signature is invalid.")
    payload = json.loads(_b64url_decode(encoded_payload))
    if not isinstance(payload, dict):
        raise ValueError("JWT payload must be an object.")
    return payload


def _sign(data: bytes, secret: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), data, hashlib.sha256).digest()


def _b64url_json(data: dict[str, Any]) -> str:
    raw = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _b64url_encode(raw)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


__all__ = [
    "AUTH_SECRET_ENV",
    "AuthenticatedUser",
    "anonymous_user",
    "decode_user_jwt_token",
    "extract_bearer_token",
    "generate_user_jwt_token",
    "get_auth_secret",
    "user_from_headers",
    "user_from_jwt_token",
]
