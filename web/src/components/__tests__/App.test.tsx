import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { App } from "../../App";

vi.mock("../../api", () => ({
  createSession: vi.fn().mockResolvedValue({ id: "s1", messages: [], title: null, user_id: null }),
  runAgent: vi.fn(),
  streamAgent: vi.fn(),
  // The Assist composer loads the domain taxonomy on mount.
  fetchSearchDomains: vi.fn().mockResolvedValue({ domains: [] }),
  submitToolApproval: vi.fn(),
  getAdminSummary: vi.fn().mockRejectedValue(new Error("no admin")),
  getAnalyticsByLLM: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByPersona: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByFlow: vi.fn().mockRejectedValue(new Error()),
  // The /tools page lists the tool inventory on mount.
  listTools: vi.fn().mockResolvedValue([]),
  discoverAdminTools: vi.fn(),
  errorStatus: vi.fn().mockReturnValue(undefined),
}));

import * as api from "../../api";

const mockStreamAgent = api.streamAgent as ReturnType<typeof vi.fn>;
const mockSubmitToolApproval = api.submitToolApproval as ReturnType<typeof vi.fn>;
const mockCreateSession = api.createSession as ReturnType<typeof vi.fn>;

const baseResponse = {
  session_id: "s1", answer: "The answer", citations: ["[D1]"],
  documents: [{ id: "D1", citation: "[D1]", title: "Doc", content: "c", url: null, score: 0.9, metadata: {} }],
  messages: [{ role: "user", content: "q", metadata: {} }, { role: "assistant", content: "The answer", metadata: {} }],
};

beforeEach(() => { vi.clearAllMocks(); });

async function submitQuery(query = "explain FAISS") {
  const textarea = screen.getByRole("textbox", { name: /question/i });
  await userEvent.clear(textarea);
  await userEvent.type(textarea, query);
  await userEvent.click(screen.getByRole("button", { name: /search/i }));
}

// The answer now renders in both the answer panel and the session timeline,
// so scope answer-presence checks to the answer column to stay unambiguous.
function answerPanelText(): string {
  return document.querySelector(".answer-column")?.textContent ?? "";
}

function fakeStream(intent: string, documents = baseResponse.documents) {
  async function* gen() {
    yield { type: "answer" as const, text: baseResponse.answer };
    yield { type: "done" as const, session_id: baseResponse.session_id, citations: baseResponse.citations, documents, intent };
  }
  return gen();
}

function errorStream(message: string) {
  async function* gen() {
    yield { type: "error" as const, detail: message };
  }
  return gen();
}

describe("App adaptive layout", () => {
  it("has no intent class on initial render", () => {
    render(<App />);
    const layout = document.querySelector(".results-layout");
    expect(layout?.classList).not.toContain("intent-search");
    expect(layout?.classList).not.toContain("intent-chat");
    expect(layout?.classList).not.toContain("intent-tool");
  });


  it("adds intent-search class to results layout on search response", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("search"));
    render(<App />);
    await submitQuery("find docs on cross-encoder reranking");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-search");
    });
  });

  it("adds intent-chat class to results layout on chat response", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("chat"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-chat");
    });
  });

  it("adds intent-tool class to results layout on tool response", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("tool", []));
    render(<App />);
    await submitQuery("run API tool");
    await waitFor(() => {
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-tool");
    });
  });

  it("shows error banner on API failure", async () => {
    mockStreamAgent.mockReturnValue(errorStream("No LLM configured"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => {
      expect(screen.getByText(/no llm configured/i)).toBeInTheDocument();
    });
  });

  it("clears error when new session is created", async () => {
    mockStreamAgent.mockReturnValue(errorStream("failed"));
    render(<App />);
    await submitQuery("q");
    await waitFor(() => expect(screen.getByText(/failed/i)).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /new/i }));
    await waitFor(() => expect(screen.queryByText(/failed/i)).not.toBeInTheDocument());
  });

  it("has no intent-badge or intent class when response omits intent", async () => {
    async function* noIntentStream() {
      yield { type: "answer" as const, text: baseResponse.answer };
      yield {
        type: "done" as const,
        session_id: baseResponse.session_id,
        citations: baseResponse.citations,
        documents: baseResponse.documents,
        intent: undefined as unknown as string,
      };
    }
    mockStreamAgent.mockReturnValue(noIntentStream());
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() => expect(answerPanelText()).toContain(baseResponse.answer));
    const layout = document.querySelector(".results-layout");
    expect(layout?.className).not.toMatch(/intent-(search|chat|tool)/);
    expect(document.querySelector(".intent-badge")).toBeNull();
  });
});

describe("App answer-state reset", () => {
  it("clears the prior answer and citation links from the answer panel when a follow-up query errors", async () => {
    async function* groundedStream() {
      yield { type: "answer" as const, text: "See [D1] for details." };
      yield {
        type: "done" as const,
        session_id: baseResponse.session_id,
        citations: baseResponse.citations,
        documents: baseResponse.documents,
        intent: "chat" as const,
      };
    }
    mockStreamAgent
      .mockReturnValueOnce(groundedStream())
      .mockReturnValueOnce(errorStream("second failed"));

    render(<App />);
    await submitQuery("first query");
    await waitFor(() =>
      expect(document.querySelector(".answer-column .citation-link")).not.toBeNull(),
    );

    await submitQuery("second query");
    await waitFor(() =>
      expect(screen.getByText(/second failed/i)).toBeInTheDocument(),
    );

    const answerColumn = document.querySelector(".answer-column");
    expect(answerColumn?.querySelector(".citation-link")).toBeNull();
    expect(answerColumn?.textContent).not.toContain("See");
  });
});

describe("App session timeline", () => {
  it("appends the user query then the assistant answer after a streamed turn", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("chat"));
    render(<App />);
    await submitQuery("what is FAISS");
    await waitFor(() =>
      expect(answerPanelText()).toContain(baseResponse.answer),
    );

    const rows = document.querySelectorAll(".session-panel .chat-row");
    expect(rows.length).toBe(2);
    expect(rows[0].classList.contains("chat-row--user")).toBe(true);
    expect(rows[0].textContent).toContain("what is FAISS");
    expect(rows[1].classList.contains("chat-row--assistant")).toBe(true);
    expect(rows[1].textContent).toContain(baseResponse.answer);
  });
});

describe("App grounding status", () => {
  it("shows 'Grounded' when the answer has citations", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("chat"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() =>
      expect(answerPanelText()).toContain(baseResponse.answer),
    );
    expect(document.querySelector(".status-pill")?.textContent).toBe("Grounded");
  });

  it("shows 'Answered' (not 'Grounded') when the answer has no citations", async () => {
    async function* noCitationsStream() {
      yield { type: "answer" as const, text: baseResponse.answer };
      yield {
        type: "done" as const,
        session_id: baseResponse.session_id,
        citations: [] as string[],
        documents: [],
        intent: "chat",
      };
    }
    mockStreamAgent.mockReturnValue(noCitationsStream());
    render(<App />);
    await submitQuery("FAISS");
    await waitFor(() =>
      expect(answerPanelText()).toContain(baseResponse.answer),
    );
    expect(document.querySelector(".status-pill")?.textContent).toBe("Answered");
  });

  it("shows the chosen route chip, with a degraded marker", async () => {
    async function* routedStream() {
      yield { type: "answer" as const, text: baseResponse.answer };
      yield {
        type: "done" as const,
        session_id: baseResponse.session_id,
        citations: [] as string[],
        documents: [],
        intent: "search",
        route: "search_agent",
        route_degraded: "no_local_model",
      };
    }
    mockStreamAgent.mockReturnValue(routedStream());
    render(<App />);
    await submitQuery("FAISS");
    await waitFor(() =>
      expect(answerPanelText()).toContain(baseResponse.answer),
    );
    const pill = document.querySelector(".route-pill");
    expect(pill?.textContent).toContain("search_agent");
    expect(pill?.getAttribute("title")).toContain("degraded: no_local_model");
    expect(pill?.classList.contains("route-pill--degraded")).toBe(true);
  });

  it("renders no route chip when the response omits route", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("chat"));
    render(<App />);
    await submitQuery("explain FAISS");
    await waitFor(() =>
      expect(answerPanelText()).toContain(baseResponse.answer),
    );
    expect(document.querySelector(".route-pill")).toBeNull();
  });
});

describe("App example chips", () => {
  it("runs the agent with the chip's query in one click and applies the intent layout", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("search"));
    render(<App />);
    await userEvent.click(
      screen.getByRole("button", { name: /find docs on cross-encoder reranking/i }),
    );
    await waitFor(() => {
      expect(mockStreamAgent).toHaveBeenCalledTimes(1);
      const layout = document.querySelector(".results-layout");
      expect(layout?.classList).toContain("intent-search");
    });
    const sentRequest = mockStreamAgent.mock.calls[0][0] as { query: string };
    expect(sentRequest.query).toBe("find docs on cross-encoder reranking");
  });
});

describe("App retrieval URL handling", () => {
  it("does not send a client search_url in normal (non-dev) mode", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("search"));
    render(<App />);
    await submitQuery("find docs");
    await waitFor(() => expect(mockStreamAgent).toHaveBeenCalledTimes(1));
    const sentRequest = mockStreamAgent.mock.calls[0][0] as {
      search_url?: string;
      source_provider?: string;
    };
    expect(sentRequest.search_url).toBeUndefined();
    expect(sentRequest.source_provider).toBeUndefined();
  });
});

describe("App tool approvals", () => {
  const approval = {
    id: "approval-1",
    tool_name: "send_email",
    arguments: { recipient: "ops@example.com" },
    expires_at: "2099-01-01T00:00:00.000Z",
  };

  it.each(["approve", "deny"] as const)(
    "submits an %s decision without restarting the stream",
    async (decision) => {
      let resumeStream!: () => void;
      const decisionSubmitted = new Promise<void>((resolve) => { resumeStream = resolve; });
      async function* approvalStream() {
        yield { type: "approval_required" as const, approval };
        await decisionSubmitted;
        yield { type: "answer" as const, text: baseResponse.answer };
        yield {
          type: "done" as const,
          session_id: baseResponse.session_id,
          citations: baseResponse.citations,
          documents: baseResponse.documents,
          intent: "tool" as const,
        };
      }
      mockStreamAgent.mockReturnValue(approvalStream());
      mockSubmitToolApproval.mockImplementation(async () => { resumeStream(); });

      render(<App />);
      await submitQuery("send an email");
      await userEvent.click(await screen.findByRole("button", {
        name: decision === "approve" ? /approve/i : /deny/i,
      }));

      await waitFor(() => expect(answerPanelText()).toContain(baseResponse.answer));
      expect(mockSubmitToolApproval).toHaveBeenCalledTimes(1);
      expect(mockSubmitToolApproval).toHaveBeenCalledWith(
        approval.id,
        decision,
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
      expect(mockStreamAgent).toHaveBeenCalledTimes(1);
      expect(screen.queryByLabelText(/approval required for send_email/i)).not.toBeInTheDocument();
    },
  );

  it("keeps the approval available and shows an error when submission fails", async () => {
    async function* approvalStream() {
      yield { type: "approval_required" as const, approval };
      await new Promise(() => undefined);
    }
    mockStreamAgent.mockReturnValue(approvalStream());
    mockSubmitToolApproval.mockRejectedValue(new Error("Approval service unavailable"));

    render(<App />);
    await submitQuery("send an email");
    await userEvent.click(await screen.findByRole("button", { name: /approve/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Approval service unavailable");
    expect(screen.getByLabelText(/approval required for send_email/i)).toBeInTheDocument();
    expect(mockStreamAgent).toHaveBeenCalledTimes(1);
  });

  it("does not let an aborted older search clear a newer search's approval", async () => {
    let rejectOlder!: (reason: unknown) => void;
    let olderSettled = false;
    const olderFailure = new Promise<never>((_, reject) => { rejectOlder = reject; });
    async function* olderStream() {
      try {
        await olderFailure;
        yield { type: "answer" as const, text: "unreachable" };
      } finally {
        olderSettled = true;
      }
    }
    async function* newerStream() {
      yield { type: "approval_required" as const, approval };
      await new Promise(() => undefined);
    }
    mockStreamAgent
      .mockReturnValueOnce(olderStream())
      .mockReturnValueOnce(newerStream());

    render(<App />);
    await submitQuery("first search");
    const textarea = screen.getByRole("textbox", { name: /question/i });
    await userEvent.clear(textarea);
    await userEvent.type(textarea, "second search{Control>}{Enter}{/Control}");
    expect(await screen.findByLabelText(/approval required for send_email/i)).toBeInTheDocument();

    rejectOlder(new DOMException("Aborted", "AbortError"));

    await waitFor(() => expect(olderSettled).toBe(true));
    expect(mockStreamAgent).toHaveBeenCalledTimes(2);
    expect(screen.getByLabelText(/approval required for send_email/i)).toBeInTheDocument();
  });

  it("clears approvals immediately when starting a new session", async () => {
    const newSession = { id: "s2", messages: [], title: null, user_id: null };
    const pendingSession = new Promise<typeof newSession>(() => undefined);
    mockCreateSession
      .mockResolvedValueOnce({ id: "s1", messages: [], title: null, user_id: null })
      .mockReturnValueOnce(pendingSession);
    async function* approvalStream() {
      yield { type: "approval_required" as const, approval };
      await new Promise(() => undefined);
    }
    mockStreamAgent.mockReturnValue(approvalStream());

    render(<App />);
    await submitQuery("send an email");
    expect(await screen.findByLabelText(/approval required for send_email/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /new/i }));

    expect(screen.queryByLabelText(/approval required for send_email/i)).not.toBeInTheDocument();
  });

  it("upserts repeated approval IDs", async () => {
    async function* approvalsStream() {
      yield { type: "approval_required" as const, approval };
      yield {
        type: "approval_required" as const,
        approval: { ...approval, tool_name: "send_updated_email" },
      };
      await new Promise(() => undefined);
    }
    mockStreamAgent.mockReturnValue(approvalsStream());

    render(<App />);
    await submitQuery("send an email");

    expect(await screen.findByLabelText(/approval required for send_updated_email/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/^approval required for send_email$/i)).not.toBeInTheDocument();
  });

  it("renders concurrent approvals independently", async () => {
    async function* approvalsStream() {
      yield { type: "approval_required" as const, approval };
      yield {
        type: "approval_required" as const,
        approval: { ...approval, id: "approval-2", tool_name: "delete_file" },
      };
      await new Promise(() => undefined);
    }
    mockStreamAgent.mockReturnValue(approvalsStream());

    render(<App />);
    await submitQuery("run tools");

    expect(await screen.findByLabelText(/approval required for send_email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/approval required for delete_file/i)).toBeInTheDocument();
  });

  it.each(["done", "error"] as const)("clears approvals on stream %s", async (ending) => {
    let finish!: () => void;
    const readyToFinish = new Promise<void>((resolve) => { finish = resolve; });
    async function* approvalStream() {
      yield { type: "approval_required" as const, approval };
      await readyToFinish;
      if (ending === "error") {
        yield { type: "error" as const, detail: "stream failed" };
      } else {
        yield {
          type: "done" as const,
          session_id: baseResponse.session_id,
          citations: [],
          documents: [],
        };
      }
    }
    mockStreamAgent.mockReturnValue(approvalStream());

    render(<App />);
    await submitQuery("run a tool");
    expect(await screen.findByLabelText(/approval required for send_email/i)).toBeInTheDocument();
    finish();

    await waitFor(() => {
      expect(screen.queryByLabelText(/approval required for send_email/i)).not.toBeInTheDocument();
    });
  });
});

describe("App dashboard layout order", () => {
  it("renders results-layout above the admin overview panel", async () => {
    vi.mocked(api.getAdminSummary).mockResolvedValueOnce({
      health_label: "OK",
      health_score: 1,
      metrics: [],
      sections: [],
    });
    render(<App />);
    const admin = await screen.findByLabelText(/admin and observability/i);
    const layout = document.querySelector(".results-layout");
    expect(layout).toBeInTheDocument();
    expect(
      layout!.compareDocumentPosition(admin) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});

describe("page navigation", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/");
  });

  it("renders the assistant page at /", () => {
    render(<App />);
    expect(screen.getByRole("textbox", { name: /question/i })).toBeInTheDocument();
    expect(window.location.pathname).toBe("/assist");
  });

  it("exposes the four surfaces as links, not tabs", () => {
    render(<App />);
    const nav = screen.getByRole("navigation", { name: /surfaces/i });
    expect(nav).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /assistant/i })).toHaveAttribute("href", "/assist");
    expect(screen.getByRole("link", { name: /^search$/i })).toHaveAttribute("href", "/search");
    expect(screen.getByRole("link", { name: /^chat$/i })).toHaveAttribute("href", "/chat");
    expect(screen.getByRole("link", { name: /^tools$/i })).toHaveAttribute("href", "/tools");
  });

  it("names the registry button 'Manage tools' so only the nav link is 'Tools'", async () => {
    // The header used to carry two controls both labelled "Tools": this nav link
    // (the agent surface) and the wrench button (the admin registry panel).
    render(<App />);
    expect(
      screen.getByRole("button", { name: /manage tools/i }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^tools$/i })).not.toBeInTheDocument();
    // Exactly one control is named exactly "Tools", and it is the link.
    expect(screen.getAllByRole("link", { name: /^tools$/i })).toHaveLength(1);
  });

  it("swaps the page and the URL when a link is clicked", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("link", { name: /^chat$/i }));
    expect(window.location.pathname).toBe("/chat");
    expect(screen.queryByRole("textbox", { name: /question/i })).not.toBeInTheDocument();
  });

  it("mounts the requested page on a deep link without passing through assist", () => {
    window.history.replaceState({}, "", "/search");
    render(<App />);
    expect(screen.getByRole("textbox", { name: /search query/i })).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: /question/i })).not.toBeInTheDocument();
  });

  it("goes back to the previous page on popstate", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("link", { name: /^tools$/i }));
    expect(window.location.pathname).toBe("/tools");
    act(() => {
      window.history.replaceState({}, "", "/assist");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(screen.getByRole("textbox", { name: /question/i })).toBeInTheDocument();
  });
});

describe("App search domain", () => {
  it("sends the selected domain in the agent request", async () => {
    (api.fetchSearchDomains as ReturnType<typeof vi.fn>).mockResolvedValue({
      domains: [
        { name: "general", description: "Broad or mixed-topic search; default" },
        { name: "finance", description: "Markets, investments, banking" },
      ],
    });
    mockStreamAgent.mockReturnValue(fakeStream("search"));
    render(<App />);
    const select = await screen.findByLabelText("Domain");
    await userEvent.selectOptions(select, "finance");
    await submitQuery("etf fees");
    await waitFor(() => expect(mockStreamAgent).toHaveBeenCalled());
    expect(mockStreamAgent.mock.calls[0][0]).toMatchObject({
      query: "etf fees",
      domain: "finance",
    });
  });

  it("defaults to general when the user picks nothing", async () => {
    mockStreamAgent.mockReturnValue(fakeStream("search"));
    render(<App />);
    await submitQuery("etf fees");
    await waitFor(() => expect(mockStreamAgent).toHaveBeenCalled());
    expect(mockStreamAgent.mock.calls[0][0]).toMatchObject({ domain: "general" });
  });
});
