# Workload Identity Implementation Plan

> **For agentic workers:** Use Superpowers test-driven-development and dispatching-parallel-agents for the independent authentication and Redis tasks; review integration before completion.

**Goal:** Safely authenticate Kubernetes automation using renewable credentials.
**Architecture:** Explicit inbound JWT federation plus native outbound cloud credential providers. Preserve local authentication with production safeguards.
**Tech Stack:** Python, FastAPI, PyJWT, boto3/botocore, redis-py, pytest.
**Spec:** docs/superpowers/specs/2026-09-09-workload-identity-design.md

## Global Constraints

- External JWTs: RS256 only; configured HTTPS issuer/JWKS; configured audience; required sub/iat/exp; maximum lifetime 3600 seconds.
- Trusted subject mappings supply local user IDs, tenant IDs and groups; remote claims never supply permissions.
- ElastiCache connect tokens expire in 900 seconds. TLS verifies hostnames and certificates.
- Keep local development and non-IAM Redis behavior compatible. No deployed cloud resources.

## Task 1: Inbound workload identity and production safeguards

Files: src/internal/auth/users.py; new src/internal/auth/workload_identity.py; src/internal/configs/app_configs.py; src/internal/configs/default_config.py; requirements.txt; requirements-unit-test.txt; tests/unit/test_workload_identity.py; tests/unit/test_auth_access.py.

Interface: shared user_from_headers(headers) continues returning AuthenticatedUser | None; external JWT verifier returns identity from trusted subject configuration. Add typed environment configuration with explicit production validation. Keep existing call sites working.

- [ ] Add real-signature regression tests. Example rejection contract:
  ```python
  def test_unmapped_subject_is_rejected(configured_verifier, sign_token):
      with pytest.raises(ValueError):
          configured_verifier(sign_token(sub="unmapped"))
  ```
  Cover valid mapping, forged admin claims, expiration, missing claims, wrong issuer/audience, algorithm confusion, bad JSON, key rotation and transport failure.
- [ ] Run `python -m pytest tests/unit/test_workload_identity.py tests/unit/test_auth_access.py -q`; record expected failures before implementation.
- [ ] Implement maintained JWT verification, cached JWKS retrieval and trusted mapping; add dependency declarations. Default local tokens to finite lifetime; reject unsafe production settings and missing production expiration.
- [ ] Run auth/config/web/MCP regression tests; format/lint edited files. Record exact outcomes.

## Task 2: Renewable Redis IAM credentials

Files: src/internal/servers/redis/iam_auth.py; src/internal/servers/redis/redis_pool.py; tests/unit/test_redis_iam_auth.py. Coordinate dependency additions with Task 1.

Interface: configure_redis_iam_auth(connection_kwargs) configures a redis-py credential provider and supported TLS options. Sync pools use SSLConnection; async Redis uses ssl=True. Credential provider supplies (username, token) and supports async callers without blocking the event loop.

- [ ] Add token and connection tests, including:
  ```python
  def test_iam_configuration_requires_credentials_provider(monkeypatch):
      monkeypatch.setenv("REDIS_IAM_USER", "search")
      monkeypatch.setenv("REDIS_IAM_CACHE_NAME", "search-cache")
      monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
      kwargs = {"password": "obsolete"}
      configure_redis_iam_auth(kwargs)
      assert "password" not in kwargs
      assert kwargs["credential_provider"] is not None
  ```
  Inspect real SigV4 query fields; simulate rotating AWS credentials; construct actual sync and async connection objects to catch unsupported kwargs.
- [ ] Run `python -m pytest tests/unit/test_redis_iam_auth.py -q`; record expected failures.
- [ ] Implement renewable signing and wire primary/replica/async clients; keep existing non-IAM path.
- [ ] Run Redis tests and lint; record exact outcomes.

## Task 3: Deployment guide and integrated verification

Files: docs/workload-identity.md; this plan and spec as needed.

- [ ] Document exact environment variables, mapping schema, credential boundaries, cloud setup links and Kubernetes service-account examples without embedded secrets.
- [ ] Explain provisioned local account requirement, remaining local auth/route boundaries, JWKS refresh and Redis reconnection behavior.
- [ ] Run full `python -m pytest tests/unit -q`, repository Ruff checks, and `git diff --check`.
- [ ] Independently review security/correctness and resolve findings with regression tests.
- [ ] Commit and push feature branch; open PR describing behavior, migration and actual validation limits. Do not merge.


## Resume record — 2026-09-10

Resumed the existing `security/workload-identity` worktree at `6983007d`; retained the partial authentication implementation and both test files. Inbound authentication and Redis implementation run independently, with dependency declarations, documentation and integrated verification owned by the controller. No new worktree or intermediate planning folder was created.

The user requested completion and workspace cleanup. After verification and PR creation, preserve the feature branch in the main working directory and remove the extra worktree; do not merge into main or deploy cloud resources.

Redis task verification: the resumed baseline produced 9 failures and 1 pass (missing credential provider/configuration checks and unsupported TLS options). The completed implementation passes all 13 Redis IAM tests, including real SigV4 parameters, SDK credential expiry/refresh, primary/replica and async connection construction. Targeted Ruff and whitespace checks passed.
