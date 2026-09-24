import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ToolEscalationCard } from "../ToolEscalationCard";

const escalation = {
  id: "esc-1",
  tool_name: "send_email",
  arguments: { to: "a@b.c" },
  category: "unknown",
  message: "remote tool reported an error",
  attempts: 1,
  expires_at: new Date(Date.now() + 60_000).toISOString(),
};

describe("ToolEscalationCard", () => {
  it("shows the failure and warns the action may have happened", () => {
    render(<ToolEscalationCard escalation={escalation} onDecision={vi.fn()} />);
    expect(screen.getByText("send_email")).toBeInTheDocument();
    expect(screen.getByText(/remote tool reported an error/)).toBeInTheDocument();
    expect(screen.getByText(/may or may not have happened/)).toBeInTheDocument();
  });

  it.each(["retry", "skip", "cancel"] as const)("submits %s once", async (decision) => {
    const onDecision = vi.fn().mockResolvedValue(undefined);
    render(<ToolEscalationCard escalation={escalation} onDecision={onDecision} />);
    const label = { retry: "Retry", skip: "Skip", cancel: "Cancel" }[decision];
    await userEvent.click(screen.getByRole("button", { name: label }));
    await userEvent.click(screen.getByRole("button", { name: label }));
    expect(onDecision).toHaveBeenCalledTimes(1);
    expect(onDecision).toHaveBeenCalledWith(decision);
  });
});
