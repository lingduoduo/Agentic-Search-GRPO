# Kubernetes workload identity and authentication hardening

The user approved implementing the security investigation's recommendations. This change adds an opt-in inbound workload identity boundary and repairs outbound Redis IAM authentication. Local development remains usable. No cloud resources are deployed.

## Architecture and alternatives

Use provider-native SDK credentials for outbound calls and a separate, explicitly configured JWT verifier for inbound calls. A gateway-only solution would leave the application's shared auth helpers unaware of workload identities. A universal token abstraction would obscure AWS request signing and services that still require API keys. The selected approach keeps those trust boundaries explicit.

Inbound: support one configured HTTPS issuer/JWKS URL and application audience using maintained PyJWT cryptographic verification, an RS256 allowlist, required sub/iat/exp, bounded lifetime (3600 seconds), and explicit subject-to-local-user mapping. Reuse cached signing keys with bounded refresh, including rotation; reject invalid tokens and unavailable keys without credential downgrade. Never import remote role, email, tenant, or group claims as authorization. Mapped user IDs refer to provisioned local accounts for endpoints requiring store membership. Mapping supplies trusted tenant/group ACL context. Existing HS256 login and MCP internal tokens remain supported; external tokens cannot choose an HS256 verifier/key. No automatic user provisioning.

Production is explicit via AGENTIC_SEARCH_ENVIRONMENT=production: reject default/empty signing secrets and admin bypass. Locally minted tokens expire by default. Production rejects missing expiration. Normalize malformed JWTs to authentication failure rather than server exceptions. These safeguards are not a replacement for per-route access control.

Outbound Redis: generate AWS SigV4 ElastiCache connect tokens with the default boto3 credential chain. Use the redis-py credential provider interface for synchronous primary/replica and asynchronous clients, retrieving current AWS credentials when signing each new connection. Required nonsecret settings identify cache, IAM username and region, with an explicit serverless flag. Tokens last 900 seconds; established connections reconnect with fresh credentials after server disconnects. TLS verifies certificates and hostnames. Do not cache bearer tokens indefinitely or store AWS keys in config. Missing IAM config fails closed. Non-IAM connections retain existing behavior.

## Optimization

Cache JWKS clients/key sets with bounded lifetime and network timeouts. Reuse AWS SDK credential providers, preserving automatic credential refresh; sign locally per Redis connection rather than making a cloud request per operation. Avoid a new general-purpose authentication framework.

## Delivery and validation

Separate implementation tasks for inbound auth and Redis, followed by deployment guidance. Tests use real JWT signatures, controlled JWKS transport, and deterministic AWS credentials; no cloud account is required. Cover expired/missing/overlong tokens, issuer/audience/algorithm mismatch, unmapped subjects, claim privilege injection, malformed data, signing-key rotation, unavailable identity provider, Redis token signature parameters, refreshed AWS credentials, TLS and sync/async connection construction. Run related and full unit tests plus repository Ruff checks and independent code review before PR creation.

Document AWS EKS Pod Identity/IRSA, GKE Workload Identity Federation, and AKS Entra Workload ID. Document that AWS credentials sign cloud requests and cannot be sent directly as bearer JWTs to this API. Do not claim live-cloud validation. Cloud-native LLM provider behavior and RDS auth remain unchanged; arbitrary SaaS API keys cannot be replaced without provider support.
