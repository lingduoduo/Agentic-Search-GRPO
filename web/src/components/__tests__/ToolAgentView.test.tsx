import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ToolAgentView } from "../ToolAgentView";
import * as api from "../../api";

describe("ToolAgentView", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("shows an approval card and posts the decision", async () => {
    const submitSpy = vi.spyOn(api, "submitToolApproval").mockResolvedValue({});
    // Only an approval_required event: the real backend blocks on the decision
    // before continuing, so no done/answer arrives yet and the card stays mounted.
    async function* fake() {
      yield {
        type: "approval_required",
        approval: {
          id: "ap1",
          tool_name: "web_search",
          arguments: {},
          expires_at: "2030-01-01T00:00:00Z",
        },
      } as const;
    }
    vi.spyOn(api, "sendToolMessage").mockImplementation(fake as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "go" },
    });
    fireEvent.click(screen.getByText("Send"));

    const approveBtn = await screen.findByRole("button", { name: /approve/i });
    fireEvent.click(approveBtn);
    await waitFor(() =>
      expect(submitSpy).toHaveBeenCalledWith("ap1", "approve"),
    );
  });

  it("shows an escalation card, posts the decision, and clears it on done", async () => {
    let resolveDecision!: () => void;
    const decided = new Promise<void>((resolve) => {
      resolveDecision = resolve;
    });
    const submitSpy = vi.spyOn(api, "submitToolEscalation").mockImplementation(async () => {
      resolveDecision();
      return {};
    });
    // Mirrors the approval test's real-backend-blocks-on-the-decision setup:
    // `done` only arrives after the decision endpoint has been called.
    async function* fake() {
      yield {
        type: "escalation_required",
        escalation: {
          id: "esc1",
          tool_name: "send_email",
          arguments: {},
          category: "unknown",
          message: "remote tool reported an error",
          attempts: 1,
          expires_at: "2030-01-01T00:00:00Z",
        },
      } as const;
      await decided;
      yield { type: "done", session_id: "s1", tool_calls: [], num_turns: 1 } as const;
    }
    vi.spyOn(api, "sendToolMessage").mockImplementation(fake as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "go" },
    });
    fireEvent.click(screen.getByText("Send"));

    const skipBtn = await screen.findByRole("button", { name: /skip/i });
    fireEvent.click(skipBtn);
    await waitFor(() =>
      expect(submitSpy).toHaveBeenCalledWith("esc1", "skip"),
    );
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: /skip/i })).not.toBeInTheDocument(),
    );
  });

  it("renders streamed tool calls and the answer", async () => {
    async function* fake() {
      yield { type: "progress", turn: 1, text: "search · 3 docs" } as const;
      yield {
        type: "tool_call",
        tool_name: "search",
        status: "completed",
        arguments: {},
        result_summary: "3 items",
        latency_ms: 10,
        error: null,
      } as const;
      yield { type: "answer", text: "done answer" } as const;
      yield { type: "done", session_id: "s1", tool_calls: [], num_turns: 1 } as const;
    }
    vi.spyOn(api, "sendToolMessage").mockImplementation(fake as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "find X" },
    });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() => expect(screen.getByText("done answer")).toBeInTheDocument());
    // Progress is ephemeral (shown only while a turn is pending); the durable
    // record after completion is the tool-call trace.
    expect(screen.getByText(/3 items/)).toBeInTheDocument();
  });

  it("keeps prior turns visible across two submits", async () => {
    const answers = ["first answer", "second answer"];
    let call = 0;
    vi.spyOn(api, "sendToolMessage").mockImplementation((() => {
      const text = answers[call++];
      async function* g() {
        yield { type: "answer", text } as const;
        yield { type: "done", session_id: "s1", tool_calls: [], num_turns: 1 } as const;
      }
      return g();
    }) as never);

    render(<ToolAgentView />);
    const input = screen.getByLabelText("Tool agent message");
    fireEvent.change(input, { target: { value: "q1" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("first answer")).toBeInTheDocument());
    fireEvent.change(input, { target: { value: "q2" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("second answer")).toBeInTheDocument());
    expect(screen.getByText("first answer")).toBeInTheDocument();
    expect(screen.getByText("q1")).toBeInTheDocument();
  });

  it("shows the no-model banner on NO_LOCAL_MODEL", async () => {
    vi.spyOn(api, "sendToolMessage").mockImplementation((() => {
      async function* g() {
        throw new Error("NO_LOCAL_MODEL");
        // eslint-disable-next-line no-unreachable
        yield undefined as never;
      }
      return g();
    }) as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "hi" },
    });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() =>
      expect(screen.getByText(/needs a local model/)).toBeInTheDocument(),
    );
  });
});

describe("ToolAgentView truncation notice", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("tells the user when the answer was cut short", async () => {
    async function* fake() {
      yield { type: "answer", text: "Hybrid retrieval combines dense and" } as const;
      yield {
        type: "done",
        session_id: "s1",
        tool_calls: [],
        num_turns: 2,
        truncated: true,
      } as const;
    }
    vi.spyOn(api, "sendToolMessage").mockImplementation(fake as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "explain retrieval" },
    });
    fireEvent.click(screen.getByText("Send"));

    expect(await screen.findByRole("status")).toHaveTextContent(/cut short/i);
  });

  it("shows no notice when the answer completed", async () => {
    async function* fake() {
      yield { type: "answer", text: "All done." } as const;
      yield {
        type: "done",
        session_id: "s1",
        tool_calls: [],
        num_turns: 2,
        truncated: false,
      } as const;
    }
    vi.spyOn(api, "sendToolMessage").mockImplementation(fake as never);

    render(<ToolAgentView />);
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "hi" },
    });
    fireEvent.click(screen.getByText("Send"));

    await screen.findByText("All done.");
    expect(screen.queryByRole("status")).toBeNull();
  });
});

describe("ToolAgentView tool catalog", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("lists the tool inventory the agent is choosing from", async () => {
    vi.spyOn(api, "listTools").mockResolvedValue([
      {
        name: "web_search",
        description: "Search the web",
        parameters: {},
        source: "function",
        provider_id: null,
        agent_callable: true,
        user_scoped: false,
      },
      {
        name: "search",
        description: "Search the corpus",
        parameters: {},
        source: "function",
        provider_id: null,
        agent_callable: false,
        user_scoped: false,
      },
    ]);

    render(<ToolAgentView />);

    expect(await screen.findByText(/2 registered across 1 server/)).toBeInTheDocument();
    expect(screen.getByText("Search the web")).toBeInTheDocument();
    // The flag that explains why the agent cannot call this one.
    expect(screen.getByText("not offered to agents")).toBeInTheDocument();
  });

  it("asks for an admin session rather than erroring when listing is forbidden", async () => {
    vi.spyOn(api, "listTools").mockRejectedValue(
      Object.assign(new Error("Forbidden"), { status: 403 }),
    );

    render(<ToolAgentView />);

    expect(
      await screen.findByText("Tool inventory needs an admin session."),
    ).toBeInTheDocument();
    // Not an error banner: using the agent without admin is a normal state.
    expect(document.querySelector(".error-banner")).toBeNull();
    // And the agent itself still works.
    expect(screen.getByLabelText("Tool agent message")).toBeInTheDocument();
  });

  it("reports an unavailable inventory distinctly from a permission problem", async () => {
    vi.spyOn(api, "listTools").mockRejectedValue(
      Object.assign(new Error("boom"), { status: 500 }),
    );

    render(<ToolAgentView />);

    expect(
      await screen.findByText("Tool inventory is unavailable right now."),
    ).toBeInTheDocument();
  });

  it("ranks tools for a query without involving the model", async () => {
    vi.spyOn(api, "listTools").mockResolvedValue([]);
    const discover = vi.spyOn(api, "discoverAdminTools").mockResolvedValue({
      request: "weather",
      stage1_servers: [],
      stage2_tools: {},
      final_tools: [{ name: "web_search", server: "local", score: 0.91 }],
    });
    const send = vi.spyOn(api, "sendToolMessage");

    render(<ToolAgentView />);
    await screen.findByText(/0 registered across 0 servers/);

    fireEvent.change(screen.getByLabelText("Discovery query"), {
      target: { value: "weather" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discover" }));

    await waitFor(() => expect(discover).toHaveBeenCalledWith("weather"));
    expect(await screen.findByText(/\(local, 0\.910\)/)).toBeInTheDocument();
    // Discovery must not go through the agent.
    expect(send).not.toHaveBeenCalled();
  });
});
