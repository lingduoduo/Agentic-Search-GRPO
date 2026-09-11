"""Single-issuer workload JWT verification; remote claims never grant permissions."""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from functools import lru_cache
from http.client import HTTPException

import jwt

from src.internal.configs import load_app_settings


def token_header(token: str) -> dict:
    try:
        return jwt.get_unverified_header(token)
    except (jwt.PyJWTError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError("Malformed JWT") from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep JWKS retrieval on the explicitly trusted HTTPS endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_jwks(url: str, *, timeout: int):
    return urllib.request.build_opener(_NoRedirect()).open(url, timeout=timeout)


class WorkloadVerifier:
    """Cache keys for five minutes; throttle all refresh attempts to one per 30s.

    Expired caches fail closed during outages. Unknown key IDs cannot trigger
    unbounded downloads, and concurrent requests share a refresh lock.
    """

    def __init__(self, url: str, issuer: str, audience: str):
        self.url, self.issuer, self.audience = url, issuer, audience
        self._keys = {}
        self._expires = 0.0
        self._retry_after = 0.0
        self._lock = threading.Lock()

    def _key(self, kid: str):
        with self._lock:
            now = time.monotonic()
            if now < self._expires and kid in self._keys:
                return self._keys[kid]
            if now >= self._retry_after:
                self._retry_after = now + 30
                try:
                    with _open_jwks(self.url, timeout=3) as response:
                        raw = response.read(1024 * 1024 + 1)
                    if len(raw) > 1024 * 1024:
                        raise ValueError("JWKS response is too large")
                    data = json.loads(raw)
                    keys = {}
                    for item in data["keys"]:
                        if (
                            item.get("kty") != "RSA"
                            or item.get("alg", "RS256") != "RS256"
                            or item.get("use", "sig") != "sig"
                        ):
                            continue
                        if "key_ops" in item and "verify" not in item["key_ops"]:
                            continue
                        key_id = item.get("kid")
                        if not isinstance(key_id, str) or not key_id or key_id in keys:
                            raise ValueError("JWKS requires unique key IDs")
                        keys[key_id] = jwt.PyJWK(item, algorithm="RS256").key
                    self._keys, self._expires = keys, now + 300
                except (
                    OSError,
                    HTTPException,
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                    RecursionError,
                    jwt.PyJWTError,
                ) as exc:
                    raise ValueError("Workload signing keys unavailable") from exc
            if now >= self._expires or kid not in self._keys:
                raise ValueError("Unknown or unavailable workload signing key")
            return self._keys[kid]

    def verify(self, token: str) -> dict:
        header = token_header(token)
        if (
            header.get("alg") != "RS256"
            or not isinstance(header.get("kid"), str)
            or not header["kid"]
        ):
            raise ValueError("Workload JWT requires RS256 and a key ID")
        try:
            payload = jwt.decode(
                token,
                self._key(header["kid"]),
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=self.audience,
                options={"require": ["sub", "iat", "exp"]},
            )
            if not payload["sub"]:
                raise ValueError("Workload subject is required")
            for claim in ("iat", "exp", "nbf"):
                value = payload.get(claim, 0)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Workload times must be finite numbers")
            if not 0 < payload["exp"] - payload["iat"] <= 3600:
                raise ValueError("Workload token lifetime must be at most 3600 seconds")
            return payload
        except (jwt.PyJWTError, TypeError, OverflowError, RecursionError) as exc:
            raise ValueError("Invalid workload JWT") from exc


@lru_cache(maxsize=8)
def _verifier(url: str, issuer: str, audience: str) -> WorkloadVerifier:
    return WorkloadVerifier(url, issuer, audience)


def workload_user(token: str):
    from .users import AuthenticatedUser

    settings = load_app_settings().auth
    if not settings.workload_issuer:
        raise ValueError("Workload identity is disabled")
    payload = _verifier(
        settings.jwt_public_key_url,
        settings.workload_issuer,
        settings.workload_audience,
    ).verify(token)
    identity = settings.workload_subjects.get(payload["sub"])
    if identity is None:
        raise ValueError("Unmapped workload subject")
    return AuthenticatedUser(
        id=identity["user_id"],
        group_ids=frozenset(identity.get("group_ids", [])),
        tenant_id=identity.get("tenant_id"),
    )
