import type {
  AdminSurfaceSummary,
  AgentExperienceRequest,
  AgentExperienceResponse,
  AuditSummary,
  BreakdownAnalytics,
  ChatSessionView,
  DebugRetrievalOutcome,
  DebugRetrievalParams,
  DebugRetrievalResponse,
  EvalResultFile,
  QueryTransformResult,
  RetrievalMode,
  FeedbackSignal,
  FeedbackTarget,
  RouteLatencyResponse,
  ServerHealth,
  DebugToolsResult,
  ToolDiscoverResult,
  OpenAPIRegisterRequest,
  OpenAPIRegisterResponse,
  QueryHistoryPage,
  RequestSnapshot,
  RequestSummary,
  SendToolMessageBody,
  SessionCreateRequest,
  SSEEvent,
  ToolInvokeRequest,
  ToolInvokeResponse,
  ToolSessionSummary,
  ToolStreamEvent,
  ToolView,
  ChatStreamEvent,
  SendChatMessageBody,
  SendSearchMessageBody,
  SearchFullResponse,
  SearchDomainOption,
} from "./types";

/**
 * Run one per-mode retrieval against the dev-console proxy.
 *
 * Unlike {@link requestJson}, this never throws on a non-2xx response: the
 * Retrieval Lab needs to render mode-specific states (503 = dense unavailable,
 * 404 = endpoint not mounted) rather than a generic error.
 */
export async function runDebugRetrieval(
  mode: RetrievalMode,
  params: DebugRetrievalParams,
): Promise<DebugRetrievalOutcome> {
  const response = await fetch(`/api/debug/retrieval/${mode}`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data
        ? String((data as { detail: unknown }).detail)
        : `Request failed with ${response.status}`;
    return { status: response.status, ok: false, data: null, detail };
  }
  return {
    status: response.status,
    ok: true,
    data: data as DebugRetrievalResponse,
    detail: null,
  };
}

/** Fetch reachability of the configured backend servers (dev console). */
export function getServerHealth(): Promise<{ servers: ServerHealth[] }> {
  return requestJson<{ servers: ServerHealth[] }>("/api/debug/health");
}

/** List evaluation result files from the configured results dir (dev console). */
export function getEvalResults(): Promise<{ results: EvalResultFile[] }> {
  return requestJson<{ results: EvalResultFile[] }>("/api/debug/eval-results");
}

/** Per-route request latency from the dev-console middleware. */
export function getRouteLatency(): Promise<RouteLatencyResponse> {
  return requestJson<RouteLatencyResponse>("/api/debug/latency");
}

/**
 * Session-level thumbs, the signal the SFT and feedback-GRPO loaders read.
 * `target` says whether the sources or the answer were at fault.
 */
export function submitSessionFeedback(
  sessionId: string,
  signal: FeedbackSignal,
  target: FeedbackTarget = "overall",
  init?: Pick<RequestInit, "signal">,
): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>("/api/feedback", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, signal, target, source: "answer_panel" }),
    signal: init?.signal,
  });
}

/** Registered tools + the discovery catalog grouped by server (dev console). */
export function getDebugTools(): Promise<DebugToolsResult> {
  return requestJson<DebugToolsResult>("/api/debug/tools");
}

/** Rank tools for a query via the semantic router (dev console). */
export function discoverTools(query: string): Promise<ToolDiscoverResult> {
  return requestJson<ToolDiscoverResult>("/api/debug/tools/discover", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

/** Run only the query-transform pipeline for a query (dev console). */
export function runQueryTransform(query: string): Promise<QueryTransformResult> {
  return requestJson<QueryTransformResult>("/api/debug/query-transform", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

/** List recent captured request runs (dev console — Request Inspector). */
export async function listDebugRequests(): Promise<{ requests: RequestSummary[] }> {
  return requestJson<{ requests: RequestSummary[] }>("/api/debug/requests");
}

/** Fetch one request's full stage-by-stage snapshot (dev console — Request Inspector). */
export async function getDebugRequest(id: string): Promise<RequestSnapshot> {
  return requestJson<RequestSnapshot>(`/api/debug/request/${encodeURIComponent(id)}`);
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data
        ? String(data.detail)
        : `Request failed with ${response.status}`;
    // Carry the status so callers can tell "you lack the role" from "it broke".
    // Additive: the message is unchanged, so existing handlers are unaffected.
    throw Object.assign(new Error(detail), { status: response.status });
  }
  return data as T;
}

/** HTTP status attached to a `requestJson` rejection, when there was one. */
export function errorStatus(err: unknown): number | undefined {
  return typeof err === "object" && err !== null && "status" in err
    ? (err as { status?: number }).status
    : undefined;
}

export function createSession(
  request: SessionCreateRequest = {},
  init?: Pick<RequestInit, "signal">,
): Promise<ChatSessionView> {
  return requestJson<ChatSessionView>("/api/sessions", {
    method: "POST",
    body: JSON.stringify(request),
    signal: init?.signal,
  });
}

export function getSession(
  sessionId: string,
  init?: Pick<RequestInit, "signal">,
): Promise<ChatSessionView> {
  return requestJson<ChatSessionView>(`/api/sessions/${sessionId}`, {
    signal: init?.signal,
  });
}

export function getAdminSummary(
  init?: Pick<RequestInit, "signal">,
): Promise<AdminSurfaceSummary> {
  return requestJson<AdminSurfaceSummary>("/admin/observability/summary", {
    signal: init?.signal,
  });
}

export function getAnalyticsByLLM(
  init?: Pick<RequestInit, "signal">,
): Promise<BreakdownAnalytics> {
  return requestJson<BreakdownAnalytics>("/analytics/by-llm", {
    signal: init?.signal,
  });
}

export function getAnalyticsByPersona(
  init?: Pick<RequestInit, "signal">,
): Promise<BreakdownAnalytics> {
  return requestJson<BreakdownAnalytics>("/analytics/by-persona", {
    signal: init?.signal,
  });
}

export function getAnalyticsByFlow(
  init?: Pick<RequestInit, "signal">,
): Promise<BreakdownAnalytics> {
  return requestJson<BreakdownAnalytics>("/analytics/by-flow", {
    signal: init?.signal,
  });
}

export function runAgent(
  request: AgentExperienceRequest,
  init?: Pick<RequestInit, "signal">,
): Promise<AgentExperienceResponse> {
  return requestJson<AgentExperienceResponse>("/api/agent", {
    method: "POST",
    body: JSON.stringify(request),
    signal: init?.signal,
  });
}

export function submitToolApproval(
  approvalId: string,
  decision: "approve" | "deny",
  init?: Pick<RequestInit, "signal">,
): Promise<unknown> {
  return requestJson<unknown>(`/api/agent/approvals/${approvalId}`, {
    method: "POST",
    body: JSON.stringify({ decision }),
    signal: init?.signal,
  });
}

/**
 * Decode an SSE byte stream into typed events.
 *
 * The server frames each event as a single `data: <json>` line closed by a
 * blank line (see src/internal/servers/sse.py), so a frame never spans lines
 * and this does not reassemble multi-line `data:` runs. It also ignores
 * `event:`, `id:` and `retry:`, none of which the server emits.
 */
async function* readSSE<T>(body: ReadableStream<Uint8Array>): AsyncGenerator<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice("data:".length).trim();
      if (payload) yield JSON.parse(payload) as T;
    }
  }
}

export async function* streamAgent(
  request: AgentExperienceRequest,
  init?: Pick<RequestInit, "signal">,
): AsyncGenerator<SSEEvent> {
  const response = await fetch("/api/agent/stream", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal: init?.signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream request failed: ${response.status}`);
  }

  yield* readSSE<SSEEvent>(response.body);
}

export function getQueryHistory(
  params: {
    page_num?: number;
    page_size?: number;
    start_time?: string;
    end_time?: string;
    user_id?: string;
    feedback_type?: "like" | "dislike";
  } = {},
  init?: Pick<RequestInit, "signal">,
): Promise<QueryHistoryPage> {
  const qs = new URLSearchParams();
  if (params.page_num !== undefined) qs.set("page_num", String(params.page_num));
  if (params.page_size !== undefined) qs.set("page_size", String(params.page_size));
  if (params.start_time) qs.set("start_time", params.start_time);
  if (params.end_time) qs.set("end_time", params.end_time);
  if (params.user_id) qs.set("user_id", params.user_id);
  if (params.feedback_type) qs.set("feedback_type", params.feedback_type);
  const query = qs.toString();
  return requestJson<QueryHistoryPage>(
    `/admin/chat-session-history${query ? `?${query}` : ""}`,
    { signal: init?.signal },
  );
}

export function getAuditSummary(
  params: { start_time?: string; end_time?: string } = {},
  init?: Pick<RequestInit, "signal">,
): Promise<AuditSummary> {
  const qs = new URLSearchParams();
  if (params.start_time) qs.set("start_time", params.start_time);
  if (params.end_time) qs.set("end_time", params.end_time);
  const query = qs.toString();
  return requestJson<AuditSummary>(
    `/admin/query-history/audit${query ? `?${query}` : ""}`,
    { signal: init?.signal },
  );
}

export function listTools(
  init?: Pick<RequestInit, "signal">,
): Promise<ToolView[]> {
  return requestJson<ToolView[]>("/admin/tools", { signal: init?.signal });
}

/**
 * Rank the registered tools against a query (TF-IDF, no LLM).
 *
 * The admin-gated twin of `discoverTools`, which hits the debug router — that
 * router is only mounted when AGENTIC_SEARCH_DEBUG_PANELS is set, so it does not
 * exist in a normal deployment. Use this one outside the dev console.
 */
export function discoverAdminTools(
  query: string,
  init?: Pick<RequestInit, "signal">,
): Promise<ToolDiscoverResult> {
  return requestJson<ToolDiscoverResult>("/admin/tools/discover", {
    method: "POST",
    body: JSON.stringify({ query }),
    signal: init?.signal,
  });
}


export function registerOpenAPITools(
  req: OpenAPIRegisterRequest,
  init?: Pick<RequestInit, "signal">,
): Promise<OpenAPIRegisterResponse> {
  return requestJson<OpenAPIRegisterResponse>("/admin/tools/openapi", {
    method: "POST",
    body: JSON.stringify(req),
    signal: init?.signal,
  });
}

export function deleteOpenAPIProvider(
  providerId: string,
  init?: Pick<RequestInit, "signal">,
): Promise<void> {
  return requestJson<void>(`/admin/tools/openapi/${providerId}`, {
    method: "DELETE",
    signal: init?.signal,
  });
}

export function invokeTool(
  name: string,
  req: ToolInvokeRequest,
  init?: Pick<RequestInit, "signal">,
): Promise<ToolInvokeResponse> {
  return requestJson<ToolInvokeResponse>(`/admin/tools/${name}/invoke`, {
    method: "POST",
    body: JSON.stringify(req),
    signal: init?.signal,
  });
}

export function submitFeedback(
  messageId: string,
  isPositive: boolean,
  feedbackText?: string,
  init?: Pick<RequestInit, "signal">,
): Promise<void> {
  return requestJson<void>("/chat/create-chat-message-feedback", {
    method: "POST",
    body: JSON.stringify({
      chat_message_id: messageId,
      is_positive: isPositive,
      feedback_text: feedbackText ?? null,
    }),
    signal: init?.signal,
  });
}

export async function* sendToolMessage(
  body: SendToolMessageBody,
  init?: Pick<RequestInit, "signal">,
): AsyncGenerator<ToolStreamEvent> {
  const response = await fetch("/tool/send-tool-message", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stream: true, ...body }),
    signal: init?.signal,
  });
  if (response.status === 400) {
    throw new Error("NO_LOCAL_MODEL");
  }
  if (!response.ok || !response.body) {
    throw new Error(`Tool stream failed: ${response.status}`);
  }
  yield* readSSE<ToolStreamEvent>(response.body);
}

export function getToolHistory(): Promise<{ sessions: ToolSessionSummary[] }> {
  return requestJson<{ sessions: ToolSessionSummary[] }>("/tool/tool-history");
}

export async function* sendChatMessage(
  body: SendChatMessageBody,
  init?: Pick<RequestInit, "signal">,
): AsyncGenerator<ChatStreamEvent> {
  const response = await fetch("/chat/send-chat-message", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stream: true, ...body }),
    signal: init?.signal,
  });
  if (response.status === 400) throw new Error("NO_LOCAL_MODEL");
  if (!response.ok || !response.body) {
    throw new Error(`Chat stream failed: ${response.status}`);
  }
  yield* readSSE<ChatStreamEvent>(response.body);
}

export function fetchSearchDomains(): Promise<{
  domains: SearchDomainOption[];
}> {
  return requestJson<{ domains: SearchDomainOption[] }>("/api/search-domains");
}

export function sendSearchMessage(
  body: SendSearchMessageBody,
): Promise<SearchFullResponse> {
  return requestJson<SearchFullResponse>("/search/send-search-message", {
    method: "POST",
    body: JSON.stringify({
      search_query: body.search_query,
      num_hits: body.num_hits ?? 8,
      run_query_expansion: false,
      stream: false,
    }),
  });
}
