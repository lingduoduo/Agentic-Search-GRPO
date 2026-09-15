import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Bot, Gauge, MessageSquarePlus, Search } from "lucide-react";
import {
  createSession,
  fetchSearchDomains,
  streamAgent,
  submitToolApproval,
} from "../api";
import { AnswerPanel } from "../components/AnswerPanel";
import { ClarificationPrompt } from "../components/ClarificationPrompt";
import { DevConsole } from "../components/debug/DevConsole";
import { SearchComposer } from "../components/SearchComposer";
import { SessionTimeline } from "../components/SessionTimeline";
import { SourceGrid } from "../components/SourceGrid";
import { ToolApprovalCard } from "../components/ToolApprovalCard";
import { ToolCallTracePanel } from "../components/ToolCallTracePanel";
import type {
  AgentExperienceRequest,
  ChatMessageView,
  ClarificationView,
  ControlFlowEventView,
  ProgressStep,
  SearchDomainOption,
  SearchSourceProvider,
  SourceDocumentView,
  ToolApprovalView,
  ToolCallTraceView,
} from "../types";

const DEFAULT_SEARCH_URL = "http://localhost:8001/retrieve";
const DEFAULT_BROWSER_SEARCH_URL = "http://localhost:8001/retrieve";

function upsertTrace(
  events: ControlFlowEventView[],
  event: ControlFlowEventView,
): ControlFlowEventView[] {
  return [...events.filter((item) => item.sequence !== event.sequence), event]
    .sort((a, b) => a.sequence - b.sequence);
}

// Dev escape hatch: `?dev=1` reveals the raw Retrieval URL override. In normal
// use the backend resolves the retrieval URL from the selected Source, so the
// client never dictates what the server fetches.
const DEV_MODE =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).has("dev");

export function AssistPage() {
  const [query, setQuery] = useState("");
  const [searchUrl, setSearchUrl] = useState(DEFAULT_SEARCH_URL);
  const [topK, setTopK] = useState(5);
  const [domain, setDomain] = useState("general");
  const [domainOptions, setDomainOptions] = useState<SearchDomainOption[]>([
    { name: "general", description: "Broad or mixed-topic search; default" },
  ]);
  const [intent, setIntent] = useState<"search" | "chat" | "tool" | undefined>(undefined);
  const [clarification, setClarification] = useState<
    { view: ClarificationView; query: string } | null
  >(null);
  const [route, setRoute] = useState<string | undefined>(undefined);
  const [routeDegraded, setRouteDegraded] = useState<string | undefined>(undefined);
  const [sourceProvider, setSourceProvider] =
    useState<SearchSourceProvider>("auto");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<string[]>([]);
  const [documents, setDocuments] = useState<SourceDocumentView[]>([]);
  const [messages, setMessages] = useState<ChatMessageView[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([]);
  const [completedSteps, setCompletedSteps] = useState<ProgressStep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [toolCalls, setToolCalls] = useState<ToolCallTraceView[]>([]);
  const [controlFlowTrace, setControlFlowTrace] = useState<ControlFlowEventView[]>([]);
  const [lastRequestId, setLastRequestId] = useState<string | null>(null);
  const [pendingApprovals, setPendingApprovals] = useState<ToolApprovalView[]>([]);
  const [showConsole, setShowConsole] = useState(false);
  // Dev-only observability console; gated at build time, never on in prod.
  const debugPanels = import.meta.env.VITE_DEBUG_PANELS === "1";
  const requestRef = useRef<AbortController | null>(null);

  // The taxonomy lives in DOMAIN_REGISTRY; fetching it keeps the 17
  // identifiers and their order from being copied into the bundle. On
  // failure the composer keeps its "general" fallback rather than
  // disappearing, since domain selection is optional.
  useEffect(() => {
    let active = true;
    fetchSearchDomains()
      .then((r) => {
        if (active && r.domains.length > 0) setDomainOptions(r.domains);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  const status = useMemo(() => {
    if (isLoading) return "Searching";
    if (error) return "Needs attention";
    // "Grounded" only when the answer is backed by retrieved sources; an
    // answer with no citations is parametric, so call it "Answered".
    if (answer) return citations.length > 0 ? "Grounded" : "Answered";
    return "Ready";
  }, [answer, citations, error, isLoading]);
  const ensureSession = useCallback(
    async (signal: AbortSignal) => {
      if (sessionId) return sessionId;
      const session = await createSession({ title: "Search session" }, { signal });
      setSessionId(session.id);
      setMessages(session.messages);
      return session.id;
    },
    [sessionId],
  );

  const handleSubmit = useCallback(async (
    eventOrQuery?: FormEvent | string,
    options?: { route?: "chat" | "search" | "tool" },
  ) => {
    if (eventOrQuery && typeof eventOrQuery !== "string") eventOrQuery.preventDefault();
    const raw = typeof eventOrQuery === "string" ? eventOrQuery : query;
    const normalizedQuery = raw.trim();
    if (!normalizedQuery) return;

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;

    setIsLoading(true);
    setError(null);
    setIntent(undefined);
    setRoute(undefined);
    setRouteDegraded(undefined);
    // Clear the previous turn's answer so a failed/next query never renders a
    // stale answer or citation anchors pointing at now-removed source cards.
    setAnswer("");
    setClarification(null);
    setStreamingAnswer("");
    setCitations([]);
    setDocuments([]);
    setToolCalls([]);
    setProgressSteps([]);
    setCompletedSteps([]);
    setControlFlowTrace([]);
    setPendingApprovals([]);
    try {
      const activeSessionId = await ensureSession(controller.signal);
      // Append the user turn to the local timeline. Done after ensureSession so
      // the new-session reset inside it can't wipe the message we just added.
      setMessages((current) => [
        ...current,
        { role: "user", content: normalizedQuery },
      ]);
      const agentRequest: AgentExperienceRequest = {
        query: normalizedQuery,
        session_id: activeSessionId,
        // Only forwarded in dev mode; otherwise the backend resolves the URL.
        search_url: DEV_MODE ? searchUrl : undefined,
        top_k: topK,
        domain,
        source_provider: DEV_MODE ? sourceProvider : undefined,
        route: options?.route,
      };

      let accumulatedAnswer = "";
      let liveSteps: ProgressStep[] = [];
      for await (const event of streamAgent(agentRequest, { signal: controller.signal })) {
        if (event.type === "progress") {
          liveSteps = [...liveSteps, { turn: event.turn, text: event.text }];
          setProgressSteps(liveSteps);
        } else if (event.type === "trace") {
          setControlFlowTrace((current) => upsertTrace(current, event.event));
        } else if (event.type === "approval_required") {
          setPendingApprovals((current) => [
            ...current.filter((approval) => approval.id !== event.approval.id),
            event.approval,
          ]);
        } else if (event.type === "claim") {
          accumulatedAnswer = accumulatedAnswer
            ? `${accumulatedAnswer} ${event.text}`
            : event.text;
          setStreamingAnswer(accumulatedAnswer);
        } else if (event.type === "answer") {
          accumulatedAnswer = event.text;
          setStreamingAnswer(event.text);
        } else if (event.type === "done") {
          setSessionId(event.session_id);
          setCitations(event.citations);
          setDocuments(event.documents);
          setToolCalls(event.tool_calls ?? []);
          setControlFlowTrace(
            [...(event.control_flow_trace ?? [])].sort(
              (a, b) => a.sequence - b.sequence,
            ),
          );
          if (event.intent && event.intent !== "clarify") setIntent(event.intent);
          setClarification(
            event.clarification
              ? { view: event.clarification, query: normalizedQuery }
              : null,
          );
          setRoute(event.route ?? undefined);
          setRouteDegraded(event.route_degraded ?? undefined);
          // ClarificationPrompt already renders the question; don't duplicate
          // it in the answer panel.
          const displayedAnswer = event.clarification ? "" : accumulatedAnswer;
          setStreamingAnswer(displayedAnswer);
          setAnswer(displayedAnswer);
          setMessages((current) => [
            ...current,
            { role: "assistant", content: accumulatedAnswer },
          ]);
          setCompletedSteps(liveSteps);
          setProgressSteps([]);
          setPendingApprovals([]);
          setLastRequestId(event.request_id ?? null);
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      }
    } catch (caught) {
      if (requestRef.current !== controller) return;
      setPendingApprovals([]);
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "Search failed");
      setAnswer("");
      setCitations([]);
      setDocuments([]);
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsLoading(false);
      }
    }
  }, [ensureSession, query, searchUrl, sourceProvider, topK]);

  const handleApprovalDecision = useCallback(async (
    approvalId: string,
    decision: "approve" | "deny",
  ) => {
    const signal = requestRef.current?.signal;
    await submitToolApproval(approvalId, decision, { signal });
    setPendingApprovals((current) =>
      current.filter((approval) => approval.id !== approvalId),
    );
  }, []);

  const handleNewSession = useCallback(async () => {
    requestRef.current?.abort();
    setPendingApprovals([]);
    const session = await createSession({ title: "Search session" });
    setSessionId(session.id);
    setAnswer("");
    setCitations([]);
    setDocuments([]);
    setToolCalls([]);
    setControlFlowTrace([]);
    setMessages([]);
    setError(null);
    setIntent(undefined);
    setClarification(null);
    setIsLoading(false);
  }, []);

  const handleTopKChange = useCallback((value: number) => {
    setTopK(Math.min(20, Math.max(1, value || 1)));
  }, []);

  const handleSourceProviderChange = useCallback((value: SearchSourceProvider) => {
    setSourceProvider(value);
    setSearchUrl((current) => {
      if (value === "browser" && current === DEFAULT_SEARCH_URL) {
        return DEFAULT_BROWSER_SEARCH_URL;
      }
      if (
        (value === "retrieval" || value === "all") &&
        current === DEFAULT_BROWSER_SEARCH_URL
      ) {
        return DEFAULT_SEARCH_URL;
      }
      return current;
    });
  }, []);

  return (
    <>
      <div className="page-actions">
        <span className="status-pill">{status}</span>
        {route && (
          <span
            className={`route-pill${routeDegraded ? " route-pill--degraded" : ""}`}
            title={
              routeDegraded
                ? `Routed to ${route} (degraded: ${routeDegraded})`
                : `Routed to ${route}`
            }
          >
            via {route}
            {routeDegraded ? " ⚠" : ""}
          </span>
        )}
        {debugPanels && (
          <button
            className={`icon-button${showConsole ? " active" : ""}`}
            type="button"
            onClick={() => setShowConsole((v) => !v)}
            title="Dev console — backend observability"
          >
            <Gauge size={18} />
            <span>Console</span>
          </button>
        )}
        <button className="icon-button" type="button" onClick={handleNewSession}>
          <MessageSquarePlus size={18} />
          <span>New</span>
        </button>
      </div>

      <SearchComposer
        query={query}
        searchUrl={searchUrl}
        topK={topK}
        sourceProvider={sourceProvider}
        isLoading={isLoading}
        showUrlField={DEV_MODE}
        showSourcePicker={DEV_MODE}
        onQueryChange={setQuery}
        onSearchUrlChange={setSearchUrl}
        onTopKChange={handleTopKChange}
        onSourceProviderChange={handleSourceProviderChange}
        domain={domain}
        domainOptions={domainOptions}
        onDomainChange={setDomain}
        onSubmit={handleSubmit}
        onExampleSelect={(q) => {
          setQuery(q);
          handleSubmit(q);
        }}
      />

      {error && <div className="error-banner">{error}</div>}

      {debugPanels && showConsole && (
        <DevConsole
          answer={streamingAnswer || answer}
          citations={citations}
          controlFlowTrace={controlFlowTrace}
          selectedRequestId={lastRequestId}
        />
      )}

      {clarification && (
        <ClarificationPrompt
          clarification={clarification.view}
          onSelect={(route) => {
            const asked = clarification.query;
            setClarification(null);
            void handleSubmit(asked, { route });
          }}
        />
      )}

      <div className={`results-layout${intent ? ` intent-${intent}` : ""}`}>
        <section className="answer-column" aria-label="Answer">
          <div className="section-heading">
            <Bot size={18} />
            <h2>Answer</h2>
          </div>
          <AnswerPanel
            answer={streamingAnswer || answer}
            citations={citations}
            intent={intent}
            documentCount={documents.length}
            progressSteps={progressSteps}
            completedSteps={completedSteps}
            sessionId={isLoading ? null : sessionId}
          />
          {pendingApprovals.map((approval) => (
            <ToolApprovalCard
              key={approval.id}
              approval={approval}
              onDecision={(decision) => handleApprovalDecision(approval.id, decision)}
            />
          ))}
          {intent === "tool" && <ToolCallTracePanel calls={toolCalls} />}
        </section>

        <section className="panel sources-panel wide" aria-label="Sources">
          <div className="section-heading">
            <Search size={18} />
            <h2>Sources</h2>
            <span className="count">{documents.length}</span>
          </div>
          <SourceGrid documents={documents} />
        </section>

        <section className="panel session-panel" aria-label="Session">
          <div className="section-heading">
            <MessageSquarePlus size={18} />
            <h2>Session</h2>
          </div>
          <SessionTimeline messages={messages} />
        </section>
      </div>
    </>
  );
}
