import { useEffect, useState } from "react";
import type { ToolEscalationView } from "../types";

type Decision = "retry" | "skip" | "cancel";
type DecisionState = "idle" | "submitting" | "decided" | "error";

interface ToolEscalationCardProps {
  escalation: ToolEscalationView;
  onDecision: (decision: Decision) => Promise<unknown> | unknown;
}

function secondsUntil(expiresAt: string): number {
  return Math.max(0, Math.ceil((new Date(expiresAt).getTime() - Date.now()) / 1_000));
}

function argumentText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null) return "null";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value) ?? String(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

export function ToolEscalationCard({ escalation, onDecision }: ToolEscalationCardProps) {
  const [state, setState] = useState<DecisionState>("idle");
  const [error, setError] = useState("");
  const [secondsRemaining, setSecondsRemaining] = useState(() =>
    secondsUntil(escalation.expires_at),
  );

  useEffect(() => {
    const updateCountdown = () => setSecondsRemaining(secondsUntil(escalation.expires_at));
    updateCountdown();
    const timer = window.setInterval(updateCountdown, 1_000);
    return () => window.clearInterval(timer);
  }, [escalation.expires_at]);

  const decide = async (decision: Decision) => {
    if (state === "submitting" || state === "decided") return;
    setState("submitting");
    setError("");
    try {
      await onDecision(decision);
      setState("decided");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to submit decision");
      setState("error");
    }
  };

  const locked = state === "submitting" || state === "decided";

  return (
    <section
      className="tool-approval-card"
      aria-label={`Tool failure: ${escalation.tool_name}`}
    >
      <div className="tool-approval-header">
        <div>
          <p className="tool-approval-eyebrow">Tool failed</p>
          <h3>{escalation.tool_name}</h3>
        </div>
        <p className="tool-approval-countdown" aria-live="polite">
          Expires in {secondsRemaining} {secondsRemaining === 1 ? "second" : "seconds"}
        </p>
      </div>

      <p>
        {escalation.message} · attempt {escalation.attempts}
      </p>

      <dl className="tool-approval-arguments">
        {Object.entries(escalation.arguments).map(([name, value]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd>{argumentText(value)}</dd>
          </div>
        ))}
      </dl>

      <p className="tool-approval-warning">
        This action may or may not have happened. Retry runs it again.
      </p>

      {error && <p role="alert" className="tool-approval-error">{error}</p>}
      {state === "decided" && <p className="tool-approval-status">Decision submitted</p>}

      <div className="tool-approval-actions">
        <button type="button" onClick={() => void decide("retry")} disabled={locked}>
          Retry
        </button>
        <button type="button" onClick={() => void decide("skip")} disabled={locked}>
          Skip
        </button>
        <button type="button" onClick={() => void decide("cancel")} disabled={locked}>
          Cancel
        </button>
      </div>
    </section>
  );
}
