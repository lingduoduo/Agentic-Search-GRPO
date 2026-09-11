# Workload identity for automated deployments

Use a Kubernetes ServiceAccount bound to a narrowly scoped cloud identity for cloud API access. Use an application-audience JWT from an explicitly trusted issuer when an automated client calls Agentic Search. These are separate authentication paths: AWS temporary credentials sign AWS requests; they are not bearer JWTs for this API.

## Application authentication

Configure the API process with:

| Environment variable | Value |
| --- | --- |
| `AGENTIC_SEARCH_ENVIRONMENT` | `production` enables production authentication safeguards. |
| `AGENTIC_SEARCH_AUTH_SECRET` | A unique, securely generated signing secret for existing browser sessions and internal MCP tokens; never the development default. |
| `AGENTIC_SEARCH_DEV_ADMIN` | `false` |
| `AGENTIC_SEARCH_JWT_PUBLIC_KEY_URL` | Trusted issuer's HTTPS JWKS endpoint, configured by an operator. |
| `AGENTIC_SEARCH_WORKLOAD_ISSUER` | Exact HTTPS issuer expected in `iss`. |
| `AGENTIC_SEARCH_WORKLOAD_AUDIENCE` | Dedicated audience for this API, e.g. `https://search.example.com`. |
| `AGENTIC_SEARCH_WORKLOAD_SUBJECTS` | JSON mapping of allowed issuer subjects to local identities (below). |

For example, a mapping for a Kubernetes-issued subject:

```json
{
  "system:serviceaccount:search:search-worker": {
    "user_id": "search-automation",
    "tenant_id": "example-tenant",
    "group_ids": ["search-readers"]
  }
}
```

For cloud-issued JWTs, use the provider's actual verified `sub` value; exchanging a Kubernetes token can change the issuer and subject. Provision the mapped local account before using routes such as `/me` and session persistence. Treat changes to the mapping as permission changes. Assign a dedicated account with the minimum permissions the automation needs. Token-supplied roles, email, groups and tenant do not grant permissions; configured mappings and existing endpoint/store authorization determine access.

The verifier accepts RS256, requires `sub`, `iat`, and `exp`, validates issuer/audience and temporal claims, and rejects lifetimes above 3600 seconds. Obtain an audience-specific token with a lifetime within this limit; a token intended for a cloud management API will be rejected. Opaque OAuth tokens are unsupported. Federation is opt-in and configured as one issuer per deployment.

JWKS are cached for 300 seconds. Refresh attempts, including unknown signing keys and failures, have a 30-second cooldown and a 3-second network timeout. Publish new signing keys before using them and overlap keys during rotation. An unknown key can be rejected during the cooldown; callers should retry with bounded backoff. Failure to obtain a usable key denies authentication. Configure the final JWKS URL directly: redirects are rejected to prevent transport downgrade or redirection to a different trust endpoint. Never derive the JWKS endpoint from an untrusted token.

Clients must renew tokens using their provider's SDK or reread a rotating projected token file before requests. Do not copy a short-lived token into a Kubernetes Secret or a process environment variable and expect it to renew itself. Never log authorization headers.

This feature does not disable existing browser login, internal HS256 tokens, or intentionally anonymous routes. Production guards reject the development signing secret and admin bypass and require expiration for local tokens; they do not make every route private. Review route authorization and ingress access for your deployment. Keep the local signing secret in a managed secret store; workload federation removes the need to distribute it to automated callers.

## Kubernetes-issued application token

If you intentionally trust the cluster's HTTPS issuer and JWKS, this projected volume requests a token for this application rather than for the Kubernetes API. Place it in the automated caller's pod template:

```yaml
spec:
  serviceAccountName: search-worker
  automountServiceAccountToken: false
  containers:
    - name: worker
      image: your-registry/search-worker:your-reviewed-version
      volumeMounts:
        - name: application-identity
          mountPath: /var/run/search-identity
          readOnly: true
  volumes:
    - name: application-identity
      projected:
        sources:
          - serviceAccountToken:
              path: token
              audience: https://search.example.com
              expirationSeconds: 3600
```

This is a pod-template fragment, not a complete application deployment. The API must be able to reach the configured JWKS endpoint. Kubernetes rotates projected tokens; clients must read the current file, not retain its initial contents. Some cluster configurations extend token lifetimes: verify the actual `exp - iat` meets the application's limit, or obtain a token through a suitable identity provider. See [Kubernetes ServiceAccounts](https://kubernetes.io/docs/concepts/security/service-accounts/).

## AWS EKS and ElastiCache

Create an EKS Pod Identity association between the workload's namespace/ServiceAccount and an IAM role. Install/configure the Pod Identity Agent and use a supported SDK. IRSA is an alternative using the cluster OIDC issuer and `AssumeRoleWithWebIdentity`. Restrict trust to the intended cluster, namespace and service account; restrict node instance metadata access. Remove static AWS credential overrides after validating migration because earlier entries in the SDK credential chain take precedence. See [EKS Pod Identity flow](https://docs.aws.amazon.com/eks/latest/userguide/pod-id-how-it-works.html).

Configure Redis with nonsecret resource identifiers:

```dotenv
USE_REDIS_IAM_AUTH=true
REDIS_HOST=your-cache-endpoint
REDIS_PORT=6379
REDIS_IAM_USER=search-worker
REDIS_IAM_CACHE_NAME=your-cache-name
REDIS_IAM_SERVERLESS=false
AWS_REGION_NAME=us-east-1
```

Use the lowercase cache name/replication group identifier for signing, not its DNS endpoint. For a serverless cache set `REDIS_IAM_SERVERLESS=true`. Create an IAM-enabled ElastiCache user with matching username/user ID, attach it to the cache user group, enable TLS, and grant `elasticache:Connect` on both the cache and user resources. Use a Redis ACL restricting commands and key patterns to the application.

The credential provider signs a 900-second token using the current AWS SDK credentials whenever a connection authenticates. Both synchronous primary/replica pools and asynchronous clients use the provider and verified TLS. Async credential retrieval runs off the event loop. Established connections do not require a new token for each command. ElastiCache disconnects IAM connections after 12 hours unless reauthenticated; a reconnect obtains fresh credentials. Applications must handle connection interruption and avoid blindly retrying non-idempotent commands. No background reauthentication loop is introduced. See [ElastiCache IAM authentication](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/auth-iam.html).

RDS token generation already uses the AWS SDK credential chain and connection hooks. This change does not alter RDS behavior or deploy database IAM permissions.

## Google GKE

Enable Workload Identity Federation for GKE and grant the Kubernetes principal only the Google Cloud resource permissions required. Use direct resource access where supported or explicitly configure service-account impersonation. Use Application Default Credentials and avoid service-account JSON keys. The existing Vertex AI integration supports `workload_identity` mode, which omits embedded Vertex credentials so the SDK can use default credentials. See [GKE Workload Identity Federation](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/workload-identity).

This authenticates calls to Google Cloud. To call Agentic Search, obtain a compatible audience-specific signed JWT and configure its issuer, JWKS and subject mapping as above; a generic Google Cloud access token is not interchangeable with an application JWT.

## Azure AKS

Enable the cluster OIDC issuer and Microsoft Entra Workload ID. Create a federated identity credential matching the namespace/ServiceAccount subject and the provider-required federation audience. Annotate the ServiceAccount with the managed identity's client ID and label the pod template `azure.workload.identity/use: "true"`. Azure SDK clients can use `DefaultAzureCredential` to obtain and refresh service-specific tokens. See [AKS workload identity setup](https://learn.microsoft.com/en-us/azure/aks/workload-identity-deploy-cluster).

This PR does not add an Azure-specific outbound LLM adapter. To call Agentic Search, configure an Entra application audience and trusted issuer/subject mapping, and ensure issued tokens meet the verifier's algorithm and lifetime constraints. The audience used for federation exchange is different from the audience of the final application token.

## Rollout validation

1. Create cloud trust bindings and least-privilege permissions in a test environment.
2. Provision the mapped application account and configure its ACLs.
3. Confirm a valid workload can call the intended protected endpoint and cannot access another tenant's resources or admin-only operations.
4. Confirm wrong issuer/audience, expired tokens, unmapped subjects and invalid signatures fail.
5. Exercise signing-key rotation, token-file rotation, temporary identity-provider outage, Redis reconnection and AWS credential refresh.
6. Remove old automated-client credentials after verifying the new path.

Unit tests use real cryptographic signatures and connection construction with controlled credentials. They do not validate your live cluster, IAM policies, cloud endpoints or ingress. Third-party APIs that only accept API keys still require managed secrets and their own rotation procedures.
