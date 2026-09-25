# Circuit breakers for serving dependencies: design

## Problem

No dependency on the serving path has a circuit breaker. When SerpAPI, the
browser search server, the HTTP cross-encoder reranker or the remote
(OpenAI-compatible) LLM server is down, **every request** pays the full
timeout (and any retries) before degrading:

| Dependency | Call site | Cost per request when down |
|---|---|---|
| SerpAPI | `serpapi_search` (`src/internal/tools/search.py:318`) | up to `serpapi_timeout_seconds` (15 s), then the cascade tries the browser |
| Browser search | the browser leg of `make_web_cascade_search` (`search.py:535`) | up to the `search_tool` timeout (15 s) × its attempts |
| HTTP reranker | `RerankHTTPRankingStage.rank` (`src/internal/search/stages.py:133`) | up to `rerank.timeout_seconds` (10 s), then `DefaultRankingStage` degrades to the fused order |
| Remote LLM | `generate` / `generate_stream` on the remote server manager (`src/model/serving.py:300-410`) | a connect error or up to `remote_total_timeout_seconds` (120 s), surfaced as `RuntimeError` |

The degradation paths already exist and work. What is missing is
**remembering** that a dependency is down, so that later requests skip it
straight to the degradation path.

The only breaker in the repo is billing's (`servers/billing/api.py:52-75`). It
is a module-global boolean that is only ever closed by hand, with no half-open
state, so it is not reusable.

## Decision

### 1. `src/internal/resilience/circuit_breaker.py` (new, torch-free, stdlib only)

```python
class CircuitOpenError(RuntimeError):
    def __init__(self, name: str, retry_in_seconds: float): ...

@dataclass(frozen=True)
class BreakerSnapshot:
    name: str
    state: str            # "closed" | "open" | "half_open"
    consecutive_failures: int
    opened_at: float | None     # monotonic clock
    retry_in_seconds: float     # 0 unless open

class CircuitBreaker:
    def __init__(self, name, *, failure_threshold, open_seconds, clock=time.monotonic): ...
    def before_call(self) -> None          # raises CircuitOpenError when open, or when half-open with the probe already out
    def record_success(self) -> None       # -> closed, failures = 0
    def record_failure(self) -> None       # failures += 1; opens at threshold; a half-open probe failure re-opens
    def snapshot(self) -> BreakerSnapshot

def get_breaker(name: str) -> CircuitBreaker   # process-wide registry, created lazily from policy
def breaker_snapshots() -> list[BreakerSnapshot]
def reset_breakers() -> None                   # tests only
```

The state machine is standard:

- **closed**: calls pass. `failure_threshold` *consecutive* failures move the
  breaker to open.
- **open**: `before_call` raises `CircuitOpenError` until `open_seconds` have
  passed.
- **half-open**: the first `before_call` after the wait is let through as the
  single probe; any concurrent call still raises. The probe's
  `record_success` closes the breaker and its `record_failure` re-opens it
  with a fresh `opened_at`.

A `threading.Lock` guards the state, because the web app calls these from the
event loop and from worker threads. Calls are never wrapped. The caller calls
`before_call` and then `record_success` or `record_failure`, so each call site
decides for itself what counts as a failure (section 3).

Every state transition logs one WARNING (opened, re-opened) or INFO (closed).
The state of every breaker is available to the admin metrics endpoint
(section 4).

### 2. Policy lives in `timeouts.toml`

The drift guard forbids new timeout or retry literals in code, so the policy
goes in the bundled file and is validated by the existing loader:

```toml
[circuit_breaker]
failure_threshold = 5      # consecutive failures that open a breaker
open_seconds = 30.0        # how long an open breaker fails fast before one probe
```

This adds a `CircuitBreakerPolicy` dataclass and a `circuit_breaker` field on
`TimeoutPolicies`, and documents both in `docs/configuration/timeouts.md`. One
policy covers all four dependencies (YAGNI). Operators override it through
`AGENTIC_SEARCH_TIMEOUTS_PATH` as with any other key. Breakers read
`get_timeout_policies()` when they are created, not at import.

### 3. Call sites: what counts as a failure

A **failure** means the dependency is unhealthy: a transport or connect error,
a timeout, HTTP 5xx, or HTTP 429. A **success** means any response the
dependency actually produced, including 4xx and empty results. Configuration
errors are neither, because the dependency was never called. A missing
SerpAPI key is the example.

| Breaker name | Site | When open |
|---|---|---|
| `serpapi` | `serpapi_search`: `before_call` after the key check; success or failure around `_get_json` | returns `[SearchPage(error="SerpAPI is temporarily skipped after repeated failures (circuit open).")]`. The cascade already treats error pages as a failed leg and moves on to the browser without waiting. |
| `browser_search` | `make_web_cascade_search`'s browser leg | appends a `SearchPage(error="Browser search is temporarily skipped after repeated failures (circuit open).")` failure and skips the call |
| `rerank` | `RerankHTTPRankingStage.rank`, around the HTTP post (a cache hit touches no breaker) | raises `CircuitOpenError`. `DefaultRankingStage.rank` gains an `except CircuitOpenError` arm before its generic one that sets `rerank_status = "circuit_open"`, `degraded = True`. |
| `remote_llm:<base_url>` (one per server, so one down server never skips another) | remote `generate` and `generate_stream` | raises `RuntimeError(f"Inference server at {base_url} is temporarily skipped after repeated failures (circuit open).")`, the same exception type callers already handle for a down server |

`CircuitOpenError` subclasses `RuntimeError`, so any caller's broad handler
keeps working.

The browser leg uses `search_tool`, which returns error pages rather than
raising for HTTP failures. For that leg a failure is "raised an exception, or
every returned page is an error page". An empty but error-free list is a
success. For the remote LLM, `aiohttp.ClientResponseError` with status ≥500 or
429 counts as a failure along with the existing connect and timeout errors,
and a 4xx is recorded as a success before it is re-raised.

`asyncio.CancelledError` records nothing: a cancelled call says nothing about
the dependency.

### 4. Visibility

`GET /api/admin/metrics` (`src/internal/servers/web/metrics_router.py`) gains
`"circuits": [asdict(s) for s in breaker_snapshots()]`. Breakers are created
lazily, so the list only shows dependencies this process has actually called.
Exporting this to Prometheus is left for a follow-up once the metrics-export
PR lands; the two PRs are independent.

## Out of scope

- Breakers for the retrieval server, MCP, public-data tools and OpenAPI tools.
  Retrieval is a local, co-deployed process whose outage means the product is
  down anyway. Public-data tools already retry and degrade through the tool
  loop.
- Sliding-window or error-rate thresholds. Consecutive failures are enough and
  easy to explain.
- Sharing breaker state across processes. It is per process, like the rest of
  the in-memory telemetry. The Dockerfile runs one worker.
- A UI for breaker state.

## Testing

- **State machine** (`tests/unit/resilience/test_circuit_breaker.py`, fake
  clock):
  - It opens after exactly `failure_threshold` failures, and a success
    resets the count.
  - While open it raises, and `retry_in_seconds` counts down.
  - After `open_seconds` exactly one probe is admitted and a concurrent
    second call raises.
  - A probe success closes the breaker; a probe failure re-opens it with a
    fresh timer.
  - Policy values come from `timeouts.toml`, and an override file changes
    them.
- **Each call site**, with the network faked:
  - Once open, the site makes **no** outbound call and returns or raises the
    documented degraded value.
  - A 4xx does not count toward opening.
  - SerpAPI with no key never touches the breaker.
  - A rerank cache hit never touches the breaker.
  - `DefaultRankingStage` reports `rerank_status == "circuit_open"`.
  - `remote_llm` raises `RuntimeError` with no network call.
- **Metrics:** `/api/admin/metrics` includes `circuits` after a breaker has
  been used.
- An autouse fixture calls `reset_breakers()` so tests do not leak state
  between each other through the process-wide registry.
- **Mutation checks:** remove the `before_call` at one site and watch its
  "no outbound call" test fail.
- The timeout-policy drift and site tests must pass with the new section.
