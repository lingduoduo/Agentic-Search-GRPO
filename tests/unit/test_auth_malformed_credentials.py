"""Malformed bearer tokens are rejected at the auth boundary, never a 500."""

import base64
import hashlib
import hmac
import json
import time

import pytest

from src.internal.auth import user_from_headers


def _token(header, payload):
    def encode(value):
        return (
            base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
        )

    message = encode(header) + "." + encode(payload)
    signature = hmac.new(
        b"malformed-test-secret", message.encode(), hashlib.sha256
    ).digest()
    return message + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()


@pytest.mark.parametrize(
    "header,payload",
    [
        ([], {"sub": "alice"}),
        (None, {"sub": "alice"}),
        ({"alg": "HS256"}, {"sub": "alice", "exp": []}),
        ({"alg": "HS256"}, {"sub": "alice", "exp": float("inf")}),
        ({"alg": "HS256"}, {"sub": "alice", "exp": float("nan")}),
        ({"alg": "HS256"}, {"sub": "alice", "groups": 123}),
        ({"alg": "HS256"}, {"sub": "alice", "groups": "admins"}),
        ({"alg": "HS256"}, {"sub": ["alice"]}),
        ({"alg": "HS256"}, {"sub": "alice", "email": ["admin@example.test"]}),
        ({"alg": "HS256"}, {"sub": "alice", "nbf": time.time() + 3600}),
    ],
)
def test_malformed_claims_are_unauthenticated(monkeypatch, header, payload):
    monkeypatch.setenv("AGENTIC_SEARCH_AUTH_SECRET", "malformed-test-secret")
    assert (
        user_from_headers({"Authorization": "Bearer " + _token(header, payload)})
        is None
    )


def test_expiry_boundary_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_AUTH_SECRET", "malformed-test-secret")
    monkeypatch.setattr("src.internal.auth.users.time.time", lambda: 1000)
    token = _token({"alg": "HS256"}, {"sub": "alice", "exp": 1000})
    assert user_from_headers({"Authorization": "Bearer " + token}) is None


def test_invalid_bearer_does_not_fall_back_to_cookie(monkeypatch):
    from starlette.requests import Request
    from src.internal.servers.users.api import resolve_request_user

    monkeypatch.setenv("AGENTIC_SEARCH_AUTH_SECRET", "malformed-test-secret")
    cookie = "fastapiusersauth=" + _token({"alg": "HS256"}, {"sub": "alice"})
    request = Request(
        {
            "type": "http",
            "headers": [
                (b"authorization", b"Bearer invalid"),
                (b"cookie", cookie.encode()),
            ],
        }
    )
    assert resolve_request_user(request) is None


@pytest.mark.parametrize(
    "authorization", ["Bearer", "Bearer ", "Bearer   ", "Bearer\tinvalid"]
)
def test_malformed_bearer_cannot_borrow_cookie_identity(monkeypatch, authorization):
    from starlette.requests import Request
    from src.internal.servers.users.api import resolve_request_user

    monkeypatch.setenv("AGENTIC_SEARCH_AUTH_SECRET", "malformed-test-secret")
    cookie = "fastapiusersauth=" + _token({"alg": "HS256"}, {"sub": "alice"})
    headers = {"Authorization": authorization, "Cookie": cookie}
    assert user_from_headers(headers) is None
    request = Request(
        {
            "type": "http",
            "headers": [
                (key.lower().encode(), value.encode()) for key, value in headers.items()
            ],
        }
    )
    assert resolve_request_user(request) is None
