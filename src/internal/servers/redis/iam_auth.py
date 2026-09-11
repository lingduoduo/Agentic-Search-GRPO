"""Renewable AWS SigV4 credentials for ElastiCache Redis connections."""

import asyncio
import os
import threading
from typing import Any

import boto3
from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import NoCredentialsError
from redis.credentials import CredentialProvider


class RedisIAMCredentialProvider(CredentialProvider):
    """Sign each connection with current credentials from the AWS SDK chain."""

    def __init__(self, username: str, cache_name: str, region: str, serverless: bool):
        self.username = username
        self.cache_name = cache_name.lower()
        self.region = region
        self.serverless = serverless
        self._session: boto3.Session | None = None
        self._lock = threading.Lock()

    def get_credentials(self) -> tuple[str, str]:
        # Session initialization and SDK refresh can involve network I/O. Serialize
        # access because boto3 sessions are not thread safe, and keep the session
        # so refreshable role credentials retain the SDK's refresh machinery.
        with self._lock:
            if self._session is None:
                self._session = boto3.Session()
            credentials = self._session.get_credentials()
            if credentials is None:
                raise NoCredentialsError()
            frozen = credentials.get_frozen_credentials()
        params = {"Action": "connect", "User": self.username}
        if self.serverless:
            params["ResourceType"] = "ServerlessCache"
        request = AWSRequest(
            method="GET", url=f"https://{self.cache_name}/", params=params
        )
        SigV4QueryAuth(frozen, "elasticache", self.region, expires=900).add_auth(
            request
        )
        return self.username, request.url.removeprefix("https://")

    async def get_credentials_async(self) -> tuple[str, str]:
        return await asyncio.to_thread(self.get_credentials)


def configure_redis_iam_auth(connection_kwargs: dict[str, Any]) -> None:
    """Attach renewable credentials and verified TLS options for async Redis.

    Sync connection pools must replace ``ssl=True`` with ``SSLConnection``.
    Configuration is validated without resolving credentials or contacting AWS.
    """
    config = {}
    for key in (
        "REDIS_IAM_USER",
        "REDIS_IAM_CACHE_NAME",
        "AWS_REGION_NAME",
        "REDIS_IAM_SERVERLESS",
    ):
        value = os.environ.get(key, "").strip()
        if not value:
            raise ValueError(f"{key} is required for Redis IAM authentication")
        config[key] = value
    serverless = config["REDIS_IAM_SERVERLESS"].lower()
    if serverless not in ("true", "false"):
        raise ValueError("REDIS_IAM_SERVERLESS must be true or false")
    connection_kwargs.pop("password", None)
    connection_kwargs.pop("username", None)
    connection_kwargs.pop("ssl_context", None)
    connection_kwargs.update(
        credential_provider=RedisIAMCredentialProvider(
            config["REDIS_IAM_USER"],
            config["REDIS_IAM_CACHE_NAME"],
            config["AWS_REGION_NAME"],
            serverless == "true",
        ),
        ssl=True,
        ssl_cert_reqs="required",
        ssl_check_hostname=True,
    )
