export type ChatRole = "user" | "assistant" | "system" | string;

export interface ChatMessageView {
  role: ChatRole;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface ChatSessionView {
  id: string;
  title: string | null;
  user_id: string | null;
  messages: ChatMessageView[];
}

export interface SourceDocumentView {
  id: string;
  citation: string;
  title: string;
  content: string;
  url: string | null;
  score: number;
  metadata: Record<string, unknown>;
}

export type AgentMode = "search_tool" | "hybrid_search" | "chat_once" | "chat_loop" | "search_agent" | "tool_agent";
export type SearchSourceProvider =
  | "auto"
  | "retrieval"
  | "google"
  | "serpapi"
  | "browser"
  | "all";

export interface AgentExperienceRequest {
  query: string;
  session_id?: string | null;
  user_id?: string | null;
  search_url?: string | null;
  top_k?: number;
  source_provider?: SearchSourceProvider;
  mode?: AgentMode;
  route?: "chat" | "search" | "tool";
}

export interface ClarificationOptionView {
  route: "chat" | "search" | "tool";
  label: string;
}

export interface ClarificationView {
  question: string;
  options: ClarificationOptionView[];
}

export interface ToolCallTraceView {
  tool_name: string;
  status: "completed" | "failed";
  arguments: Record<string, unknown>;
  result_summary: string;
  latency_ms: number;
  error: string | null;
}

export interface ControlFlowEventView {
  sequence: number;
  timestamp: string;
  turn: number;
  component: string;
  action: string;
  status: "started" | "completed" | "decided" | "skipped" | "failed" | string;
  duration_ms: number | null;
  details: Record<string, string | number | boolean | null>;
}

export interface AgentExperienceResponse {
  session_id: string;
  answer: string;
  citations: string[];
  documents: SourceDocumentView[];
  messages: ChatMessageView[];
  intent?: "search" | "chat" | "tool" | "clarify";
  tool_calls?: ToolCallTraceView[];
  control_flow_trace?: ControlFlowEventView[];
  clarification?: ClarificationView | null;
}

export interface SessionCreateRequest {
  title?: string | null;
  user_id?: string | null;
}

export interface QueryHistoryItem {
  id: string;
  user_id: string | null;
  name: string | null;
  first_user_message: string;
  first_ai_message: string;
  time_created: string;
  feedback_type: "like" | "dislike" | "mixed" | null;
  flow_type: "chat" | "slack";
  conversation_length: number;
  llm_name: string | null;
}

export interface QueryHistoryPage {
  items: QueryHistoryItem[];
  total_items: number;
}

export interface AuditSummary {
  total_sessions: number;
  total_messages: number;
  sessions_with_feedback: number;
  sessions_with_dislike: number;
  dislike_rate: number;
  top_disliked_queries: string[];
}

export type AdminSurfaceKey =
  | "access"
  | "auth"
  | "models"
  | "tools"
  | "analytics"
  | "enterprise";

export interface AdminSurfaceMetric {
  label: string;
  value: string;
  detail: string;
}

export interface AdminSurfaceSection {
  key: AdminSurfaceKey;
  title: string;
  status: string;
  tone: "good" | "watch" | "neutral";
  description: string;
  items: string[];
}

export interface AdminSurfaceSummary {
  health_label: string;
  health_score: number;
  metrics: AdminSurfaceMetric[];
  sections: AdminSurfaceSection[];
}

export interface BreakdownItem {
  label: string;
  session_count: number;
}

export interface BreakdownAnalytics {
  dimension: string;
  items: BreakdownItem[];
  total_sessions: number;
}

export interface ToolView {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  source: "function" | "openapi" | "mcp" | string;
  provider_id: string | null;
  /** False when no agent loop may be offered this tool (still directly invocable). */
  agent_callable: boolean;
  /** True when the tool is backed by per-user storage and needs a signed-in caller. */
  user_scoped: boolean;
}

export interface CatalogTool {
  name: string;
  description: string;
  source: string;
  server: string;
  /**
   * Optional: the /admin/tools payload carries these, the /api/debug/tools
   * catalog does not (its ToolDefinition has no such fields). Undefined means
   * "unknown", so the badges are simply not rendered.
   */
  agent_callable?: boolean;
  user_scoped?: boolean;
}

export interface CatalogServer {
  name: string;
  description: string;
  tools: CatalogTool[];
}

export interface DebugToolsResult {
  registered: ToolView[];
  catalog: CatalogServer[];
}

export interface ToolDiscoverResult {
  request: string;
  stage1_servers: { name: string; score: number }[];
  stage2_tools: Record<string, { server_score: number; tools: [string, number][] }>;
  final_tools: { name: string; server: string; score: number }[];
}

export interface OpenAPIRegisterRequest {
  name: string;
  openapi_json: string;
  headers?: Record<string, string>;
}

export interface OpenAPIRegisterResponse {
  provider_id: string;
  tool_names: string[];
}

export interface ToolInvokeRequest {
  arguments: Record<string, unknown>;
}

export interface ToolInvokeResponse {
  response: string;
  raw: unknown;
  errors: string[];
}

// ---------------------------------------------------------------------------
// SSE streaming types
// ---------------------------------------------------------------------------

export interface SSEProgressEvent {
  type: "progress";
  text: string;
  turn: number;
}

export interface SSEAnswerEvent {
  type: "answer";
  text: string;
}

export interface SSEClaimEvent {
  type: "claim";
  text: string;
}

export interface SSETraceEvent {
  type: "trace";
  event: ControlFlowEventView;
}

export interface ToolApprovalView {
  id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  expires_at: string;
}

export interface SSEApprovalRequiredEvent {
  type: "approval_required";
  approval: ToolApprovalView;
}

export interface SSEDoneEvent {
  type: "done";
  session_id: string;
  citations: string[];
  documents: SourceDocumentView[];
  intent?: "search" | "chat" | "tool" | "clarify";
  route?: string | null;
  route_degraded?: string | null;
  tool_calls?: ToolCallTraceView[];
  control_flow_trace?: ControlFlowEventView[];
  request_id?: string;
  clarification?: ClarificationView | null;
}

export interface SSEErrorEvent {
  type: "error";
  detail: string;
}

export type SSEEvent =
  | SSEProgressEvent
  | SSEAnswerEvent
  | SSEClaimEvent
  | SSETraceEvent
  | SSEApprovalRequiredEvent
  | SSEDoneEvent
  | SSEErrorEvent;

export interface ProgressStep {
  turn: number;
  text: string;  // e.g. "search_routing_tool · 5 docs" or "writing answer..."
}

// --- Dev console: per-mode retrieval inspection ---

export type RetrievalMode = "sparse" | "dense" | "hybrid" | "graph";

export interface RetrievalResultRow {
  doc_id: string;
  title: string;
  text?: string;
  score: number;
}

export interface DebugRetrievalResponse {
  results: RetrievalResultRow[];
  retrieval_mode: string;
  executed_queries: string[];
  latency_ms: number;
}

export interface DebugRetrievalParams {
  query: string;
  top_k: number;
  rrf_k?: number;
  mmr_lambda?: number;
  over_fetch?: number;
  rerank?: boolean;
}

export interface DebugRetrievalOutcome {
  status: number;
  ok: boolean;
  data: DebugRetrievalResponse | null;
  detail: string | null;
}

export interface ServerHealth {
  name: string;
  url: string;
  status: "up" | "down" | string;
}

export interface StageRecordView {
  stage: "intent" | "search" | "tool" | "llm" | "final";
  label: string;
  timestamp: number;
  duration_ms: number | null;
  payload: Record<string, unknown>;
}

export interface RequestSummary {
  request_id: string;
  query: string;
  created_at: number;
  route: string | null;
  stage_count: number;
}

export interface RequestSnapshot {
  request_id: string;
  query: string;
  created_at: number;
  route: string | null;
  route_degraded: string | null;
  total_ms: number | null;
  stages: StageRecordView[];
}

/** Which side of the pipeline an eval metric describes; order is display order. */
export const METRIC_GROUPS = ["retrieval", "generation", "reward", "latency", "other"] as const;
export type MetricGroup = (typeof METRIC_GROUPS)[number];

export interface EvalResultFile {
  name: string;
  modified: number;
  /** Every finite number in the file, nested keys dotted. */
  metrics: Record<string, number>;
  /** The same numbers bucketed by the shared taxonomy; empty groups omitted. */
  groups?: Partial<Record<MetricGroup, Record<string, number>>>;
}

/** One row of GET /api/debug/latency — a route's recent-request window. */
export interface RouteLatencyRow {
  method: string;
  route: string;
  count: number;
  errors: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
}

/** Per-stage window from GET /api/debug/latency: requests that used the stage. */
export interface StageLatency {
  count: number;
  p50_ms?: number;
  p95_ms?: number;
  max_ms?: number;
  avg_docs?: number;
  cache_hit_rate?: number;
  avg_prompt_tokens?: number;
  avg_completion_tokens?: number;
}

export interface RouteLatencyResponse {
  routes: RouteLatencyRow[];
  stages?: { retrieval: StageLatency; generation: StageLatency };
}

export type FeedbackSignal = "thumbs_up" | "thumbs_down";
/** What a thumbs is about: the sources, the answer, or the turn as a whole. */
export type FeedbackTarget = "retrieval" | "generation" | "overall";

export interface QueryTransformResult {
  original: string;
  variants: string[];
  merged_filters: Record<string, unknown>;
  active: boolean;
  legs: {
    sub_queries?: string[];
    multi_query?: string[];
    rewrite?: string | null;
    hyde_text?: string | null;
    step_back?: string | null;
    keywords?: string[];
  };
}

export interface ToolSessionSummary {
  session_id: string;
  title: string;
  created_at: string;
}

export type ToolStreamEvent =
  | { type: "progress"; turn: number; text: string }
  | ({ type: "tool_call" } & ToolCallTraceView)
  | { type: "answer"; text: string }
  | { type: "approval_required"; approval: ToolApprovalView }
  | {
      type: "done";
      session_id: string;
      tool_calls: ToolCallTraceView[];
      num_turns: number;
      // The answer is a fragment: a generation hit the wall-clock stop.
      truncated?: boolean;
    }
  | { type: "error"; detail: string };

export interface SendToolMessageBody {
  session_id?: string;
  message: string;
  run_search_tool?: boolean;
  stream?: boolean;
}

// ---------------------------------------------------------------------------
// Direct chat/search surfaces
// ---------------------------------------------------------------------------

export type ChatStreamEvent =
  | { type: "answer"; text: string }
  | { type: "done"; session_id: string }
  | { type: "error"; detail: string };

export interface SendChatMessageBody {
  session_id?: string;
  message: string;
  stream?: boolean;
}

export interface SendSearchMessageBody {
  search_query: string;
  num_hits?: number;
}

// Mirrors SearchDocWithContent in
// src/internal/servers/query_and_chat/models.py — no document_id field.
export interface SearchDocView {
  title: string | null;
  url: string | null;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

// Mirrors SearchFullResponse in
// src/internal/servers/query_and_chat/models.py.
export interface SearchFullResponse {
  all_executed_queries: string[];
  search_docs: SearchDocView[];
  llm_selected_doc_ids?: string[] | null;
  error?: string | null;
}

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCallTraceView[];
  progress?: string[];
  pending?: boolean;
}
