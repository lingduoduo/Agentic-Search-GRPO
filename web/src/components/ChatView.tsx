import { useState } from "react";
import { sendChatMessage } from "../api";
import type { ConversationTurn } from "../types";
import { Transcript } from "./Transcript";

export function ChatView() {
  const [message, setMessage] = useState("");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [noModel, setNoModel] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function patchLastAssistant(patch: Partial<ConversationTurn>) {
    setTurns((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") {
          next[i] = { ...next[i], ...patch };
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
    setTurns((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", pending: true },
    ]);
    try {
      for await (const e of sendChatMessage({ message: text, session_id: sessionId })) {
        if (e.type === "answer") patchLastAssistant({ content: e.text });
        else if (e.type === "done") {
          setSessionId(e.session_id);
          patchLastAssistant({ pending: false, degraded: e.degraded ?? null });
        } else if (e.type === "error") {
          setError(e.detail);
          patchLastAssistant({ pending: false });
        }
      }
      setMessage("");
    } catch (err) {
      patchLastAssistant({ pending: false });
      if (err instanceof Error && err.message === "NO_LOCAL_MODEL") setNoModel(true);
      else setError(err instanceof Error ? err.message : "Chat failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="chat-view" aria-label="Chat">
      {noModel && (
        <div className="error-banner" role="alert">
          Chat needs a local model — set <code>SEARCH_AGENT_MODEL</code> (or{" "}
          <code>SEARCH_AGENT_SERVER_URL</code>) in <code>.env</code> and restart the backend.
        </div>
      )}
      <Transcript turns={turns} />
      {error && <div className="error-banner">{error}</div>}
      <div className="chat-view__composer">
        <input
          aria-label="Chat message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Message the model directly…"
          disabled={busy}
        />
        <button onClick={submit} disabled={busy}>
          {busy ? "…" : "Send"}
        </button>
      </div>
    </section>
  );
}
