import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from src.internal.auth import users, workload_identity
from src.internal.configs import load_app_settings


@pytest.fixture
def federation(monkeypatch):
    workload_identity._verifier.cache_clear()
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update(kid="key1", alg="RS256", use="sig")
    monkeypatch.setenv("AGENTIC_SEARCH_JWT_PUBLIC_KEY_URL", "https://issuer.test/keys")
    monkeypatch.setenv("AGENTIC_SEARCH_WORKLOAD_ISSUER", "https://issuer.test")
    monkeypatch.setenv("AGENTIC_SEARCH_WORKLOAD_AUDIENCE", "search")
    monkeypatch.setenv(
        "AGENTIC_SEARCH_WORKLOAD_SUBJECTS",
        json.dumps(
            {
                "workload": {
                    "user_id": "alice",
                    "group_ids": ["eng"],
                    "tenant_id": "acme",
                }
            }
        ),
    )

    def sign(**updates):
        payload = dict(
            sub="workload",
            iss="https://issuer.test",
            aud="search",
            iat=int(time.time()),
            exp=int(time.time()) + 300,
        )
        payload.update(updates)
        return jwt.encode(payload, private, algorithm="RS256", headers={"kid": "key1"})

    return sign, {"keys": [jwk]}


def test_workload_maps_only_trusted_identity(federation, monkeypatch):
    sign, keys = federation
    monkeypatch.setattr(
        "src.internal.auth.workload_identity._open_jwks", lambda *a, **k: Response(keys)
    )
    user = users.user_from_headers(
        {
            "Authorization": "Bearer "
            + sign(role="admin", groups=["admin"], tenant_id="evil", email="admin@test")
        }
    )
    assert user is not None
    assert (user.id, user.group_ids, user.tenant_id, user.email, user.metadata) == (
        "alice",
        frozenset({"eng"}),
        "acme",
        None,
        {},
    )


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size=-1):
        return json.dumps(self.data).encode()


@pytest.mark.parametrize(
    "overrides",
    [
        {"sub": "unknown"},
        {"iss": "https://evil.test"},
        {"aud": "other"},
        {"exp": 0},
        {"exp": None},
        {"iat": None},
        {"sub": None},
        {"exp": int(time.time()) + 4000},
        {"iat": int(time.time()) + 100},
        {"nbf": int(time.time()) + 100},
    ],
)
def test_invalid_workloads_rejected(federation, monkeypatch, overrides):
    sign, keys = federation
    monkeypatch.setattr(
        "src.internal.auth.workload_identity._open_jwks", lambda *a, **k: Response(keys)
    )
    assert (
        users.user_from_headers({"Authorization": "Bearer " + sign(**overrides)})
        is None
    )


@pytest.mark.parametrize(
    "token", ["W10.e30.eA", "!!!!.!!!!.!!!!", "é.e30.eA", "e30.W10.eA"]
)
def test_malformed_tokens_are_auth_failure(token):
    assert users.user_from_headers({"Authorization": "Bearer " + token}) is None


def test_local_default_expiration():
    payload = users.decode_user_jwt_token(
        users.generate_user_jwt_token(user_id="alice", secret="test"), secret="test"
    )
    assert 0 < payload["exp"] - payload["iat"] <= 3600


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"AGENTIC_SEARCH_AUTH_SECRET": ""},
        {"AGENTIC_SEARCH_AUTH_SECRET": "agentic-search-dev-secret"},
        {
            "AGENTIC_SEARCH_AUTH_SECRET": "production-test-secret-32-characters",
            "AGENTIC_SEARCH_DEV_ADMIN": "true",
        },
    ],
)
def test_production_rejects_unsafe_settings(env):
    with pytest.raises(ValueError):
        load_app_settings({"AGENTIC_SEARCH_ENVIRONMENT": "production", **env})


def test_production_requires_local_expiration(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_ENVIRONMENT", "production")
    monkeypatch.setenv(
        "AGENTIC_SEARCH_AUTH_SECRET", "production-test-secret-32-characters"
    )
    token = jwt.encode(
        {"sub": "alice"}, "production-test-secret-32-characters", algorithm="HS256"
    )
    assert users.user_from_headers({"Authorization": "Bearer " + token}) is None


def test_invalid_authorization_does_not_fallback_cookie():
    token = users.generate_user_jwt_token(user_id="alice")
    assert (
        users.user_from_headers(
            {"Authorization": "Bearer ", "Cookie": "fastapiusersauth=" + token}
        )
        is None
    )


@pytest.mark.parametrize(
    "claims", [{"iat": None}, {"nbf": None}, {"groups": 123}, {"exp": 10**400}]
)
def test_malformed_signed_local_claims_are_auth_failure(claims):
    token = users.generate_user_jwt_token(user_id="alice", extra=claims)
    assert users.user_from_headers({"Authorization": "Bearer " + token}) is None


def test_production_expiration_required_with_explicit_secret(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_ENVIRONMENT", "production")
    monkeypatch.setenv("AGENTIC_SEARCH_AUTH_SECRET", "x" * 32)
    token = jwt.encode({"sub": "alice"}, "x" * 32, algorithm="HS256")
    with pytest.raises(ValueError):
        users.decode_user_jwt_token(token, secret="x" * 32)


@pytest.mark.parametrize(
    "updates",
    [
        {"iat": "123"},
        {"exp": True},
        {"nbf": float("nan")},
        {"exp": float("inf")},
        {"sub": ["workload"]},
        {"exp": 10**400},
    ],
)
def test_workload_claim_types_rejected(federation, monkeypatch, updates):
    sign, keys = federation
    monkeypatch.setattr(
        "src.internal.auth.workload_identity._open_jwks", lambda *a, **k: Response(keys)
    )
    assert (
        users.user_from_headers({"Authorization": "Bearer " + sign(**updates)}) is None
    )


@pytest.mark.parametrize("claim", ["sub", "iat", "exp"])
def test_required_workload_claims(federation, monkeypatch, claim):
    sign, keys = federation
    token = sign()
    payload = jwt.decode(token, options={"verify_signature": False})
    del payload[claim]
    # Re-sign with a real key so rejection tests claim validation, not the signature.
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk["kid"] = "key1"
    monkeypatch.setattr(
        "src.internal.auth.workload_identity._open_jwks",
        lambda *a, **k: Response({"keys": [jwk]}),
    )
    token = jwt.encode(payload, private, algorithm="RS256", headers={"kid": "key1"})
    assert users.user_from_headers({"Authorization": "Bearer " + token}) is None


def test_external_issuer_cannot_use_local_secret(federation):
    token = users.generate_user_jwt_token(
        user_id="workload", extra={"iss": "https://issuer.test"}
    )
    assert users.user_from_headers({"Authorization": "Bearer " + token}) is None


def test_bad_signature_rejected(federation, monkeypatch):
    sign, keys = federation
    keys["keys"][0].update(
        json.loads(
            jwt.algorithms.RSAAlgorithm.to_jwk(
                rsa.generate_private_key(
                    public_exponent=65537, key_size=2048
                ).public_key()
            )
        )
    )
    monkeypatch.setattr(
        "src.internal.auth.workload_identity._open_jwks", lambda *a, **k: Response(keys)
    )
    assert users.user_from_headers({"Authorization": "Bearer " + sign()}) is None


def test_jwks_rotation_refresh_and_outage(federation, monkeypatch):
    sign, keys = federation
    clock = [1000.0]
    calls = []

    def fetch(*a, **kw):
        calls.append(kw)
        return Response(keys)

    monkeypatch.setattr("src.internal.auth.workload_identity._open_jwks", fetch)
    monkeypatch.setattr(workload_identity.time, "monotonic", lambda: clock[0])
    assert users.user_from_jwt_token(sign()).id == "alice"
    assert users.user_from_jwt_token(sign()).id == "alice"
    assert len(calls) == 1
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    key["kid"] = "key2"
    payload = jwt.decode(sign(), options={"verify_signature": False})
    rotated = jwt.encode(payload, private, algorithm="RS256", headers={"kid": "key2"})
    keys["keys"] = [key]
    assert users.user_from_headers({"Authorization": "Bearer " + rotated}) is None
    assert len(calls) == 1
    clock[0] += 31
    assert users.user_from_jwt_token(rotated).id == "alice"
    assert len(calls) == 2

    def unavailable(*a, **kw):
        calls.append(kw)
        raise OSError("offline")

    monkeypatch.setattr("src.internal.auth.workload_identity._open_jwks", unavailable)
    assert users.user_from_jwt_token(rotated).id == "alice"
    clock[0] += 301
    assert users.user_from_headers({"Authorization": "Bearer " + rotated}) is None
    assert users.user_from_headers({"Authorization": "Bearer " + rotated}) is None
    assert len(calls) == 3


@pytest.mark.parametrize(
    "keys",
    [{}, {"keys": None}, {"keys": [None]}, {"keys": [{"kty": "RSA", "kid": "key1"}]}],
)
def test_malformed_jwks_fails_closed(federation, monkeypatch, keys):
    sign, _ = federation
    monkeypatch.setattr(
        "src.internal.auth.workload_identity._open_jwks", lambda *a, **k: Response(keys)
    )
    assert users.user_from_headers({"Authorization": "Bearer " + sign()}) is None


@pytest.mark.parametrize(
    "updates",
    [
        {"AGENTIC_SEARCH_WORKLOAD_ISSUER": "http://issuer.test"},
        {"AGENTIC_SEARCH_JWT_PUBLIC_KEY_URL": "https://user:password@issuer.test/keys"},
        {"AGENTIC_SEARCH_WORKLOAD_AUDIENCE": ""},
        {"AGENTIC_SEARCH_WORKLOAD_SUBJECTS": "[]"},
        {"AGENTIC_SEARCH_WORKLOAD_SUBJECTS": "{invalid"},
        {
            "AGENTIC_SEARCH_WORKLOAD_SUBJECTS": '{"workload":{"user_id":"alice","is_admin":true}}'
        },
        {
            "AGENTIC_SEARCH_WORKLOAD_SUBJECTS": '{"workload":{"user_id":"alice","group_ids":"admin"}}'
        },
    ],
)
def test_workload_configuration_fails_closed(federation, monkeypatch, updates):
    for name, value in updates.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_app_settings()


def test_truncated_jwks_fails_closed(federation, monkeypatch):
    from http.client import IncompleteRead

    sign, _ = federation

    def truncated(*a, **kw):
        raise IncompleteRead(b"{")

    monkeypatch.setattr("src.internal.auth.workload_identity._open_jwks", truncated)
    assert users.user_from_headers({"Authorization": "Bearer " + sign()}) is None


def test_jwks_redirects_are_disabled():
    import urllib.request

    handler_type = getattr(
        workload_identity, "_NoRedirect", urllib.request.HTTPRedirectHandler
    )
    handler = handler_type()
    request = urllib.request.Request("https://issuer.test/keys")
    assert (
        handler.redirect_request(
            request, None, 302, "Found", {}, "http://issuer.test/keys"
        )
        is None
    )


def test_deeply_nested_jwt_header_fails_closed():
    import base64

    header = (
        base64.urlsafe_b64encode(("[" * 2000 + "0" + "]" * 2000).encode())
        .decode()
        .rstrip("=")
    )
    assert (
        users.user_from_headers({"Authorization": "Bearer " + header + ".e30.eA"})
        is None
    )
