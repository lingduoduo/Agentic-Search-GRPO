from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import threading
from typing import Any
from typing import cast
from typing import Optional

import redis
from fastapi import Request
from redis import asyncio as aioredis
from redis.backoff import ExponentialBackoff
from redis.client import Redis
from redis.exceptions import BusyLoadingError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.lock import Lock as RedisLock
from redis.retry import Retry

from src.shared_configs.contextvars import get_current_tenant_id

from src.internal.servers.redis.iam_auth import configure_redis_iam_auth
from src.internal.servers.redis.tenant_redis_client import TenantRedisClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection config — read from environment with safe defaults
# ---------------------------------------------------------------------------
REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD: str = os.environ.get("REDIS_PASSWORD", "")
REDIS_DB_NUMBER: int = int(os.environ.get("REDIS_DB_NUMBER", "0"))
REDIS_POOL_MAX_CONNECTIONS: int = int(
    os.environ.get("REDIS_POOL_MAX_CONNECTIONS", "128")
)
REDIS_SSL: bool = os.environ.get("REDIS_SSL", "false").lower() == "true"
REDIS_SSL_CA_CERTS: str | None = os.environ.get("REDIS_SSL_CA_CERTS") or None
REDIS_SSL_CERT_REQS: str = os.environ.get("REDIS_SSL_CERT_REQS", "required")
REDIS_REPLICA_HOST: str = os.environ.get("REDIS_REPLICA_HOST", REDIS_HOST)
REDIS_HEALTH_CHECK_INTERVAL: int = int(
    os.environ.get("REDIS_HEALTH_CHECK_INTERVAL", "60")
)
USE_REDIS_IAM_AUTH: bool = (
    os.environ.get("USE_REDIS_IAM_AUTH", "false").lower() == "true"
)
REDIS_AUTH_KEY_PREFIX: str = os.environ.get("REDIS_AUTH_KEY_PREFIX", "auth_token:")

# Socket keepalive options — left empty by default; callers may override.
REDIS_SOCKET_KEEPALIVE_OPTIONS: dict[int, int] = {}

# Cookie name used by the FastAPI auth layer.
FASTAPI_USERS_AUTH_COOKIE_NAME: str = os.environ.get(
    "FASTAPI_USERS_AUTH_COOKIE_NAME", "fastapiusersauth"
)

# Shared namespace prefix — used for keys that are not tenant-specific.
DEFAULT_REDIS_PREFIX: str = os.environ.get("DEFAULT_REDIS_PREFIX", "shared")

SCAN_ITER_COUNT_DEFAULT = 4096

_RETRYABLE_ERRORS: list[type[Exception]] = [
    BusyLoadingError,
    RedisConnectionError,
    RedisTimeoutError,
]


def _client_retry_kwargs() -> dict[str, Any]:
    return {
        "retry": Retry(ExponentialBackoff(cap=2.0, base=0.1), retries=3),
        "retry_on_error": _RETRYABLE_ERRORS,
    }


class RedisPool:
    _instance: Optional["RedisPool"] = None
    _lock: threading.Lock = threading.Lock()
    _pool: redis.BlockingConnectionPool
    _replica_pool: redis.BlockingConnectionPool

    def __new__(cls) -> "RedisPool":
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RedisPool, cls).__new__(cls)
                    cls._instance._init_pools()
        return cls._instance

    def _init_pools(self) -> None:
        self._pool = RedisPool.create_pool(ssl=REDIS_SSL)
        self._replica_pool = RedisPool.create_pool(
            host=REDIS_REPLICA_HOST, ssl=REDIS_SSL
        )

    def get_client(self, tenant_id: str) -> TenantRedisClient:
        return TenantRedisClient(
            tenant_id,
            redis.Redis(connection_pool=self._pool, **_client_retry_kwargs()),
        )

    def get_replica_client(self, tenant_id: str) -> TenantRedisClient:
        return TenantRedisClient(
            tenant_id,
            redis.Redis(connection_pool=self._replica_pool, **_client_retry_kwargs()),
        )

    def get_raw_client(self) -> Redis:
        return redis.Redis(connection_pool=self._pool, **_client_retry_kwargs())

    def get_raw_replica_client(self) -> Redis:
        return redis.Redis(connection_pool=self._replica_pool, **_client_retry_kwargs())

    @staticmethod
    def create_pool(
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB_NUMBER,
        password: str = REDIS_PASSWORD,
        max_connections: int = REDIS_POOL_MAX_CONNECTIONS,
        ssl_ca_certs: str | None = REDIS_SSL_CA_CERTS,
        ssl_cert_reqs: str = REDIS_SSL_CERT_REQS,
        ssl: bool = False,
    ) -> redis.BlockingConnectionPool:
        if USE_REDIS_IAM_AUTH:
            connection_kwargs: dict[str, Any] = {"ssl_ca_certs": ssl_ca_certs}
            configure_redis_iam_auth(connection_kwargs)
            connection_kwargs.pop("ssl")
            return redis.BlockingConnectionPool(
                host=host,
                port=port,
                db=db,
                max_connections=max_connections,
                timeout=None,
                health_check_interval=REDIS_HEALTH_CHECK_INTERVAL,
                socket_keepalive=True,
                socket_keepalive_options=REDIS_SOCKET_KEEPALIVE_OPTIONS,
                connection_class=redis.SSLConnection,
                **connection_kwargs,
            )

        if ssl:
            return redis.BlockingConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                timeout=None,
                health_check_interval=REDIS_HEALTH_CHECK_INTERVAL,
                socket_keepalive=True,
                socket_keepalive_options=REDIS_SOCKET_KEEPALIVE_OPTIONS,
                connection_class=redis.SSLConnection,
                ssl_ca_certs=ssl_ca_certs,
                ssl_cert_reqs=ssl_cert_reqs,
            )

        return redis.BlockingConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            timeout=None,
            health_check_interval=REDIS_HEALTH_CHECK_INTERVAL,
            socket_keepalive=True,
            socket_keepalive_options=REDIS_SOCKET_KEEPALIVE_OPTIONS,
        )


redis_pool = RedisPool()


def get_redis_client(
    *,
    tenant_id: str | None = None,
) -> TenantRedisClient:
    """Returns a Redis client with tenant-specific key prefixing."""
    if tenant_id is None:
        tenant_id = get_current_tenant_id()

    return redis_pool.get_client(tenant_id or DEFAULT_REDIS_PREFIX)


def get_redis_replica_client(
    *,
    tenant_id: str | None = None,
) -> TenantRedisClient:
    """Returns a Redis replica client with tenant-specific key prefixing."""
    if tenant_id is None:
        tenant_id = get_current_tenant_id()

    return redis_pool.get_replica_client(tenant_id or DEFAULT_REDIS_PREFIX)


def get_shared_redis_client() -> TenantRedisClient:
    """Returns a Redis client using the shared namespace prefix."""
    return redis_pool.get_client(DEFAULT_REDIS_PREFIX)


def get_shared_redis_replica_client() -> TenantRedisClient:
    """Returns a Redis replica client using the shared namespace prefix."""
    return redis_pool.get_replica_client(DEFAULT_REDIS_PREFIX)


def get_raw_redis_client() -> Redis:
    """Returns a Redis client without tenant prefixing."""
    return redis_pool.get_raw_client()


def get_raw_redis_replica_client() -> Redis:
    """Returns a Redis replica client without tenant prefixing."""
    return redis_pool.get_raw_replica_client()


SSL_CERT_REQS_MAP = {
    "none": ssl.CERT_NONE,
    "optional": ssl.CERT_OPTIONAL,
    "required": ssl.CERT_REQUIRED,
}

_async_redis_connection: aioredis.Redis | None = None
_async_lock = asyncio.Lock()


async def get_async_redis_connection() -> aioredis.Redis:
    """Provides a shared async Redis connection, lazily created and reused."""
    global _async_redis_connection

    if _async_redis_connection is None:
        async with _async_lock:
            if _async_redis_connection is None:
                connection_kwargs: dict[str, Any] = {
                    "host": REDIS_HOST,
                    "port": REDIS_PORT,
                    "db": REDIS_DB_NUMBER,
                    "password": REDIS_PASSWORD,
                    "max_connections": REDIS_POOL_MAX_CONNECTIONS,
                    "health_check_interval": REDIS_HEALTH_CHECK_INTERVAL,
                    "socket_keepalive": True,
                    "socket_keepalive_options": REDIS_SOCKET_KEEPALIVE_OPTIONS,
                }

                if USE_REDIS_IAM_AUTH:
                    connection_kwargs["ssl_ca_certs"] = REDIS_SSL_CA_CERTS
                    configure_redis_iam_auth(connection_kwargs)
                elif REDIS_SSL:
                    ssl_context = ssl.create_default_context()
                    if REDIS_SSL_CA_CERTS:
                        ssl_context.load_verify_locations(REDIS_SSL_CA_CERTS)
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = SSL_CERT_REQS_MAP.get(
                        REDIS_SSL_CERT_REQS, ssl.CERT_NONE
                    )
                    connection_kwargs["ssl"] = ssl_context

                _async_redis_connection = aioredis.Redis(**connection_kwargs)

    return _async_redis_connection


async def retrieve_auth_token_data(token: str) -> dict | None:
    """Validate auth token against Redis and return token data."""
    try:
        r = await get_async_redis_connection()
        redis_key = REDIS_AUTH_KEY_PREFIX + token
        token_data_str = await r.get(redis_key)

        if not token_data_str:
            logger.debug("Token key %s not found or expired in Redis", redis_key)
            return None

        return json.loads(token_data_str)
    except json.JSONDecodeError:
        logger.error("Error decoding token data from Redis")
        return None
    except Exception as e:
        logger.error("Unexpected error in retrieve_auth_token_data: %s", str(e))
        raise ValueError(f"Unexpected error in retrieve_auth_token_data: {str(e)}")


async def retrieve_auth_token_data_from_redis(request: Request) -> dict | None:
    """Validate auth token from request cookie."""
    token = request.cookies.get(FASTAPI_USERS_AUTH_COOKIE_NAME)
    if not token:
        logger.debug("No auth token cookie found")
        return None
    return await retrieve_auth_token_data(token)


REDIS_WS_TOKEN_PREFIX = "ws_token:"
WS_TOKEN_TTL_SECONDS = 60
WS_TOKEN_RATE_LIMIT_MAX = 10
WS_TOKEN_RATE_LIMIT_WINDOW_SECONDS = 60
REDIS_WS_TOKEN_RATE_LIMIT_PREFIX = "ws_token_rate:"


class WsTokenRateLimitExceeded(Exception):
    """Raised when a user exceeds the WS token generation rate limit."""


async def store_ws_token(token: str, user_id: str) -> None:
    """Store a short-lived WebSocket authentication token in Redis."""
    r = await get_async_redis_connection()

    rate_limit_key = REDIS_WS_TOKEN_RATE_LIMIT_PREFIX + user_id
    pipe = r.pipeline()
    pipe.incr(rate_limit_key)
    pipe.expire(rate_limit_key, WS_TOKEN_RATE_LIMIT_WINDOW_SECONDS)
    results = await pipe.execute()
    new_count = results[0]

    if new_count > WS_TOKEN_RATE_LIMIT_MAX:
        await r.decr(rate_limit_key)
        logger.warning("WS token rate limit exceeded for user %s", user_id)
        raise WsTokenRateLimitExceeded(
            f"Rate limit exceeded. Maximum {WS_TOKEN_RATE_LIMIT_MAX} tokens per minute."
        )

    redis_key = REDIS_WS_TOKEN_PREFIX + token
    token_data = json.dumps({"sub": user_id})
    await r.set(redis_key, token_data, ex=WS_TOKEN_TTL_SECONDS)


async def retrieve_ws_token_data(token: str) -> dict | None:
    """Validate a WebSocket token and atomically consume it."""
    try:
        r = await get_async_redis_connection()
        redis_key = REDIS_WS_TOKEN_PREFIX + token
        token_data_str = await r.getdel(redis_key)

        if not token_data_str:
            return None

        return json.loads(token_data_str)
    except json.JSONDecodeError:
        logger.error("Error decoding WS token data from Redis")
        return None
    except Exception as e:
        logger.error("Unexpected error in retrieve_ws_token_data: %s", str(e))
        return None


def redis_lock_dump(lock: RedisLock, r: TenantRedisClient) -> None:
    name = lock.name
    raw = r.raw_client
    ttl = raw.ttl(name)
    locked = lock.locked()
    owned = lock.owned()
    local_token: str | None = lock.local.token

    remote_token_raw = raw.get(name)
    if remote_token_raw:
        remote_token_bytes = cast(bytes, remote_token_raw)
        remote_token = remote_token_bytes.decode("utf-8")
    else:
        remote_token = None

    logger.warning(
        "RedisLock diagnostic: name=%s locked=%s owned=%s local_token=%s remote_token=%s ttl=%s",
        name,
        locked,
        owned,
        local_token,
        remote_token,
        ttl,
    )
