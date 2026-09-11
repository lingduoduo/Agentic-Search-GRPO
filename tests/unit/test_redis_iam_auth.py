import asyncio
import datetime
import importlib
import ssl
import threading
from urllib.parse import parse_qs, urlsplit

import botocore.auth
import pytest
import redis
from botocore.credentials import Credentials, DeferredRefreshableCredentials
from botocore.exceptions import NoCredentialsError
from redis import asyncio as aioredis

from src.internal.servers.redis import iam_auth


@pytest.fixture(autouse=True)
def iam_env(monkeypatch):
    for key, value in {
        "REDIS_IAM_USER": "search",
        "REDIS_IAM_CACHE_NAME": "search-cache",
        "AWS_REGION_NAME": "us-east-1",
        "REDIS_IAM_SERVERLESS": "false",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AWS_REGION", raising=False)


def provider():
    kwargs = {"password": "obsolete", "username": "obsolete"}
    iam_auth.configure_redis_iam_auth(kwargs)
    assert "password" not in kwargs
    assert "username" not in kwargs
    return kwargs["credential_provider"]


@pytest.mark.parametrize(
    "key",
    [
        "REDIS_IAM_USER",
        "REDIS_IAM_CACHE_NAME",
        "AWS_REGION_NAME",
        "REDIS_IAM_SERVERLESS",
    ],
)
def test_missing_configuration_fails_closed(monkeypatch, key):
    monkeypatch.delenv(key)
    with pytest.raises(ValueError, match=key):
        provider()


def test_invalid_serverless_fails_closed(monkeypatch):
    monkeypatch.setenv("REDIS_IAM_SERVERLESS", "maybe")
    with pytest.raises(ValueError, match="REDIS_IAM_SERVERLESS"):
        provider()


@pytest.mark.parametrize("serverless", [True, False])
def test_signed_token_refreshes_credentials(monkeypatch, serverless):
    monkeypatch.setenv("REDIS_IAM_SERVERLESS", str(serverless).lower())
    current = [Credentials("FIRST", "secret", "session-one")]

    class Session:
        def get_credentials(self):
            return current[0]

    monkeypatch.setattr("boto3.Session", Session)
    monkeypatch.setattr(
        botocore.auth, "get_current_datetime", lambda: datetime.datetime(2026, 9, 9, 12)
    )
    p = provider()
    username, first = p.get_credentials()
    assert username == "search"
    parsed = urlsplit("https://" + first)
    assert parsed.netloc == "search-cache"
    q = parse_qs(parsed.query)
    assert q["Action"] == ["connect"]
    assert q["User"] == ["search"]
    assert q["X-Amz-Expires"] == ["900"]
    assert q["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert q["X-Amz-Credential"] == [
        "FIRST/20260909/us-east-1/elasticache/aws4_request"
    ]
    assert q["X-Amz-Security-Token"] == ["session-one"]
    assert len(q["X-Amz-Signature"][0]) == 64
    assert ("ResourceType" in q) == serverless
    if serverless:
        assert q["ResourceType"] == ["ServerlessCache"]
    current[0] = Credentials("SECOND", "new-secret", "session-two")
    second = p.get_credentials()[1]
    assert second != first
    assert parse_qs(urlsplit("https://" + second).query)["X-Amz-Credential"][
        0
    ].startswith("SECOND/")


def test_async_credentials_run_off_event_loop(monkeypatch):
    owner = threading.get_ident()

    class Session:
        def __init__(self):
            assert threading.get_ident() != owner

        def get_credentials(self):
            assert threading.get_ident() != owner
            return Credentials("KEY", "secret", "session")

    monkeypatch.setattr("boto3.Session", Session)
    p = provider()
    assert asyncio.run(p.get_credentials_async())[0] == "search"


def test_real_sync_primary_replica_and_async_connections(monkeypatch):
    pools = importlib.import_module("src.internal.servers.redis.redis_pool")
    monkeypatch.setattr(pools, "USE_REDIS_IAM_AUTH", True)
    for host in ["primary.example", "replica.example"]:
        pool = pools.RedisPool.create_pool(host=host, ssl_cert_reqs="none")
        conn = pool.make_connection()
        assert isinstance(conn, redis.SSLConnection)
        assert conn.host == host
        assert conn.credential_provider is not None
        assert conn.cert_reqs == ssl.CERT_REQUIRED
        assert conn.check_hostname is True
    monkeypatch.setattr(pools, "_async_redis_connection", None)

    async def check():
        client = await pools.get_async_redis_connection()
        conn = client.connection_pool.make_connection()
        assert isinstance(conn, aioredis.connection.SSLConnection)
        context = conn.ssl_context.get()
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        assert conn.credential_provider is not None
        await client.aclose()

    asyncio.run(check())


def test_non_iam_password_and_tls_compatibility(monkeypatch):
    pools = importlib.import_module("src.internal.servers.redis.redis_pool")
    monkeypatch.setattr(pools, "USE_REDIS_IAM_AUTH", False)
    conn = pools.RedisPool.create_pool(password="local-secret").make_connection()
    assert type(conn) is redis.Connection
    assert conn.password == "local-secret"
    conn = pools.RedisPool.create_pool(
        password="local-secret", ssl=True
    ).make_connection()
    assert isinstance(conn, redis.SSLConnection)
    assert conn.password == "local-secret"


def test_missing_aws_credentials_fail_closed(monkeypatch):
    class Session:
        def get_credentials(self):
            return None

    monkeypatch.setattr("boto3.Session", Session)
    with pytest.raises(NoCredentialsError):
        provider().get_credentials()


def test_sdk_refreshable_credentials_renew_token(monkeypatch):
    now = [datetime.datetime(2026, 9, 9, 12, tzinfo=datetime.timezone.utc)]
    generation = [0]

    def refresh():
        generation[0] += 1
        return {
            "access_key": f"KEY{generation[0]}",
            "secret_key": "secret",
            "token": f"session{generation[0]}",
            "expiry_time": (now[0] + datetime.timedelta(hours=1)).isoformat(),
        }

    credentials = DeferredRefreshableCredentials(
        refresh_using=refresh, method="test-role", time_fetcher=lambda: now[0]
    )

    class Session:
        def get_credentials(self):
            return credentials

    monkeypatch.setattr("boto3.Session", Session)
    p = provider()
    first = p.get_credentials()[1]
    now[0] += datetime.timedelta(hours=2)
    second = p.get_credentials()[1]
    assert first != second
    query = parse_qs(urlsplit("https://" + second).query)
    assert query["X-Amz-Credential"][0].startswith("KEY2/")
    assert query["X-Amz-Security-Token"] == ["session2"]


def test_primary_and_replica_pool_initialization(monkeypatch):
    pools = importlib.import_module("src.internal.servers.redis.redis_pool")
    monkeypatch.setattr(pools, "USE_REDIS_IAM_AUTH", True)
    monkeypatch.setattr(pools, "REDIS_REPLICA_HOST", "replica.example")
    instance = object.__new__(pools.RedisPool)
    instance._init_pools()
    primary = instance._pool.make_connection()
    replica = instance._replica_pool.make_connection()
    assert primary.host == pools.REDIS_HOST
    assert replica.host == "replica.example"
    for conn in (primary, replica):
        assert isinstance(conn, redis.SSLConnection)
        assert isinstance(
            conn.credential_provider, redis.credentials.CredentialProvider
        )
        assert conn.cert_reqs == ssl.CERT_REQUIRED
        assert conn.check_hostname is True
