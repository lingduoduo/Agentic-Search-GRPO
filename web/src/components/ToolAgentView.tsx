import { useEffect, useState } from "react";
import {
  discoverAdminTools,
  errorStatus,
  listTools,
  sendToolMessage,
  submitToolApproval,
  submitToolEscalation,
} from "../api";
import { groupToolsByServer } from "../toolCatalog";
import type {
  CatalogServer,
  ConversationTurn,
  ToolApprovalView,
  ToolDiscoverResult,
  ToolEscalationView,
} from "../types";
import { ToolApprovalCard } from "./ToolApprovalCard";
import { ToolCatalog } from "./ToolCatalog";
import { ToolEscalationCard } from "./ToolEscalationCard";
import { Transcript } from "./Transcript";

export function ToolAgentView() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [pendingApprovals, setPendingApprovals] = useState<ToolApprovalView[]>([]);
  const [pendingEscalations, setPendingEscalations] = useState<ToolEscalationView[]>([]);
  const [noModel, setNoModel] = useState(false);
  const [truncated, setTruncated] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [servers, setServers] = useState<CatalogServer[] | null>(null);
  const [registeredCount, setRegisteredCount] = useState(0);
  const [discovery, setDiscovery] = useState<ToolDiscoverResult | null>(null);
  const [catalogNote, setCatalogNote] = useState<string | undefined>(undefined);

  // The inventory the agent is choosing from. /admin/tools is require_admin, so
  // a non-admin gets a note rather than an error banner — using the agent
  // without being able to list its tools is a normal state, not a failure.
  useEffect(() => {
    let alive = true;
    listTools().then(
      (tools) => {
        if (!alive) return;
        setServers(groupToolsByServer(tools));
        setRegisteredCount(tools.length);
      },
      (err) => {
        if (!alive) return;
        const status = errorStatus(err);
        setServers([]);
        setCatalogNote(
          status === 401 || status === 403
            ? "Tool inventory needs an admin session."
            : "Tool inventory is unavailable right now.",
        );
      },
    );
    return () => {
      alive = false;
    };
  }, []);

  function patchLastAssistant(fn: (t: ConversationTurn) => ConversationTurn) {
    setTurns((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") {
          next[i] = fn(next[i]);
          break;
        }
      }
      return next;
    });
  }

  async function submit() {
    const text = message.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    setNoModel(false);
    setTruncated(false);
    setPendingApprovals([]);
    setPendingEscalations([]);
    setTurns((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", toolCalls: [], progress: [], pending: true },
    ]);
    try {
      for await (const e of sendToolMessage({ message: text, session_id: sessionId })) {
        if (e.type === "progress")
          patchLastAssistant((t) => ({ ...t, progress: [...(t.progress ?? []), e.text] }));
        else if (e.type === "tool_call")
          patchLastAssistant((t) => ({ ...t, toolCalls: [...(t.toolCalls ?? []), e] }));
        else if (e.type === "answer")
          patchLastAssistant((t) => ({ ...t, content: e.text }));
        else if (e.type === "approval_required")
          setPendingApprovals((a) => [...a, e.approval]);
        else if (e.type === "escalation_required")
          setPendingEscalations((a) => [
            ...a.filter((escalation) => escalation.id !== e.escalation.id),
            e.escalation,
          ]);
        else if (e.type === "done") {
          setSessionId(e.session_id);
          setTruncated(Boolean(e.truncated));
          setPendingApprovals([]);
          setPendingEscalations([]);
          patchLastAssistant((t) => ({ ...t, pending: false }));
        } else if (e.type === "error") {
          setError(e.detail);
          setPendingApprovals([]);
          setPendingEscalations([]);
          patchLastAssistant((t) => ({ ...t, pending: false }));
        }
      }
      setMessage("");
    } catch (err) {
      patchLastAssistant((t) => ({ ...t, pending: false }));
      setPendingApprovals([]);
      setPendingEscalations([]);
      if (err instanceof Error && err.message === "NO_LOCAL_MODEL") setNoModel(true);
      else setError(err instanceof Error ? err.message : "Tool agent failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="tool-agent-view" aria-label="Tool Agent">
      {noModel && (
        <div className="error-banner" role="alert">
          Tool Agent needs a local model — set <code>SEARCH_AGENT_MODEL</code> (or{" "}
          <code>SEARCH_AGENT_SERVER_URL</code>) in <code>.env</code> and restart the backend.
        </div>
      )}
      {truncated && (
        <p className="truncation-notice" role="status">
          This answer was cut short — generation hit the time limit. Raise{" "}
          <code>AGENTIC_SEARCH_GENERATION_TIMEOUT</code> (seconds, <code>0</code>{" "}
          disables) and restart the backend for longer answers.
        </p>
      )}
      <Transcript turns={turns} />
      {pendingApprovals.map((approval) => (
        <ToolApprovalCard
          key={approval.id}
          approval={approval}
          onDecision={(decision) =>
            submitToolApproval(approval.id, decision).finally(() =>
              setPendingApprovals((a) => a.filter((p) => p.id !== approval.id)),
            )
          }
        />
      ))}
      {pendingEscalations.map((escalation) => (
        <ToolEscalationCard
          key={escalation.id}
          escalation={escalation}
          onDecision={(decision) =>
            submitToolEscalation(escalation.id, decision).finally(() =>
              setPendingEscalations((a) => a.filter((p) => p.id !== escalation.id)),
            )
          }
        />
      ))}
      {error && <div className="error-banner">{error}</div>}
      <div className="tool-agent-view__composer">
        <input
          aria-label="Tool agent message"
          value={message}
          onChange={(ev) => setMessage(ev.target.value)}
          onKeyDown={(ev) => ev.key === "Enter" && submit()}
          placeholder="Ask the tool agent to do something…"
          disabled={busy}
        />
        <button onClick={submit} disabled={busy}>
          {busy ? "Running…" : "Send"}
        </button>
      </div>
      <ToolCatalog
        servers={servers}
        registeredCount={registeredCount}
        discovery={discovery}
        onDiscover={(q) =>
          discoverAdminTools(q).then(setDiscovery, () => setDiscovery(null))
        }
        note={catalogNote}
      />
    </section>
  );
}
