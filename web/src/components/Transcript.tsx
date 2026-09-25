import type { ConversationTurn } from "../types";
import { ToolCallTracePanel } from "./ToolCallTracePanel";

export function Transcript({ turns }: { turns: ConversationTurn[] }) {
  return (
    <div className="transcript">
      {turns.map((turn, i) => (
        <div key={i} className={`turn turn--${turn.role}`}>
          {turn.role === "assistant" &&
            turn.pending &&
            turn.progress &&
            turn.progress.length > 0 && (
              <ul className="turn__progress">
                {turn.progress.map((p, j) => (
                  <li key={j}>{p}</li>
                ))}
              </ul>
            )}
          {turn.toolCalls && turn.toolCalls.length > 0 && (
            <ToolCallTracePanel calls={turn.toolCalls} />
          )}
          {turn.content && <div className="turn__content">{turn.content}</div>}
          {turn.role === "assistant" && turn.degraded && (
            <p className="degraded-notice" role="status" title={turn.degraded}>
              ⚠ Model unavailable — degraded answer
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
