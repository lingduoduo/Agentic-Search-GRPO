import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../api";
import { AnswerPanel } from "../AnswerPanel";

// ---------------------------------------------------------------------------
// FeedbackBar — session thumbs with a retrieval/generation target
// ---------------------------------------------------------------------------

describe("AnswerPanel feedback bar", () => {
  afterEach(() => vi.restoreAllMocks());

  it("is absent without a session id", () => {
    render(<AnswerPanel answer="answer" citations={[]} />);
    expect(screen.queryByRole("group", { name: /rate this answer/i })).not.toBeInTheDocument();
  });

  it("is absent while the answer is still streaming", () => {
    render(
      <AnswerPanel
        answer="partial"
        citations={[]}
        sessionId="s1"
        progressSteps={[{ turn: 1, text: "writing answer..." }]}
      />,
    );
    expect(screen.queryByRole("group", { name: /rate this answer/i })).not.toBeInTheDocument();
  });

  it("posts thumbs_up as overall and then thanks the user", async () => {
    const spy = vi.spyOn(api, "submitSessionFeedback").mockResolvedValue({ ok: true });
    render(<AnswerPanel answer="answer" citations={[]} sessionId="s1" />);

    await userEvent.click(screen.getByRole("button", { name: /^helpful$/i }));

    expect(spy).toHaveBeenCalledWith("s1", "thumbs_up", "overall");
    expect(await screen.findByRole("status")).toHaveTextContent(/thanks/i);
    expect(screen.queryByRole("button", { name: /helpful/i })).not.toBeInTheDocument();
  });

  it("asks what was off after a thumbs-down and posts the chosen target", async () => {
    const spy = vi.spyOn(api, "submitSessionFeedback").mockResolvedValue({ ok: true });
    render(<AnswerPanel answer="answer" citations={[]} sessionId="s1" />);

    await userEvent.click(screen.getByRole("button", { name: /not helpful/i }));
    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByRole("group", { name: /what was off/i })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /^sources$/i }));

    expect(spy).toHaveBeenCalledWith("s1", "thumbs_down", "retrieval");
    expect(await screen.findByRole("status")).toHaveTextContent(/thanks/i);
  });

  it("lets the user cancel out of the what-was-off prompt", async () => {
    const spy = vi.spyOn(api, "submitSessionFeedback").mockResolvedValue({ ok: true });
    render(<AnswerPanel answer="answer" citations={[]} sessionId="s1" />);

    await userEvent.click(screen.getByRole("button", { name: /not helpful/i }));
    await userEvent.click(screen.getByRole("button", { name: /^cancel$/i }));

    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByRole("group", { name: /rate this answer/i })).toBeInTheDocument();
  });

  it("maps 'Answer' to generation and 'Both' to overall", async () => {
    const spy = vi.spyOn(api, "submitSessionFeedback").mockResolvedValue({ ok: true });
    const { unmount } = render(<AnswerPanel answer="a1" citations={[]} sessionId="s1" />);
    await userEvent.click(screen.getByRole("button", { name: /not helpful/i }));
    await userEvent.click(screen.getByRole("button", { name: /^answer$/i }));
    expect(spy).toHaveBeenLastCalledWith("s1", "thumbs_down", "generation");
    unmount();

    render(<AnswerPanel answer="a2" citations={[]} sessionId="s1" />);
    await userEvent.click(screen.getByRole("button", { name: /not helpful/i }));
    await userEvent.click(screen.getByRole("button", { name: /^both$/i }));
    expect(spy).toHaveBeenLastCalledWith("s1", "thumbs_down", "overall");
  });

  it("reports a failed post and lets the user retry", async () => {
    const spy = vi
      .spyOn(api, "submitSessionFeedback")
      .mockRejectedValueOnce(new Error("503"))
      .mockResolvedValueOnce({ ok: true });
    render(<AnswerPanel answer="answer" citations={[]} sessionId="s1" />);

    await userEvent.click(screen.getByRole("button", { name: /^helpful$/i }));
    await waitFor(() => expect(screen.getByText(/couldn't send/i)).toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: /^helpful$/i }));
    expect(spy).toHaveBeenCalledTimes(2);
    expect(await screen.findByRole("status")).toHaveTextContent(/thanks/i);
  });
});

describe("AnswerPanel", () => {
  it("renders empty state when answer is empty", () => {
    render(<AnswerPanel answer="" citations={[]} />);
    expect(screen.getByText(/results will appear here/i)).toBeInTheDocument();
  });

  it("renders the answer text", () => {
    render(<AnswerPanel answer="FAISS is a vector library." citations={[]} />);
    expect(screen.getByText(/FAISS is a vector library/)).toBeInTheDocument();
  });

  it("does not render citation row when citations are empty", () => {
    render(<AnswerPanel answer="Some answer." citations={[]} />);
    expect(screen.queryByLabelText(/citations/i)).not.toBeInTheDocument();
  });

  it("renders 'Searched · 5 sources' badge when intent is search", () => {
    render(<AnswerPanel answer="results" citations={[]} intent="search" documentCount={5} />);
    expect(screen.getByText(/searched · 5 sources/i)).toBeInTheDocument();
  });

  it("renders 'Answered · 2 citations' badge when intent is chat", () => {
    render(<AnswerPanel answer="answer" citations={["[D1]", "[D2]"]} intent="chat" />);
    expect(screen.getByText(/answered · 2 citations/i)).toBeInTheDocument();
  });

  it("renders 'Used tools' badge when intent is tool", () => {
    render(<AnswerPanel answer="tool output" citations={[]} intent="tool" />);
    expect(screen.getByText(/used tools/i)).toBeInTheDocument();
  });

  it("renders no badge when intent is undefined", () => {
    render(<AnswerPanel answer="answer" citations={[]} />);
    expect(document.querySelector(".intent-badge")).not.toBeInTheDocument();
  });

  it("renders no badge when answer is empty even if intent is set", () => {
    render(<AnswerPanel answer="" citations={[]} intent="chat" />);
    expect(document.querySelector(".intent-badge")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ProgressLog — live phase
// ---------------------------------------------------------------------------

it("renders live progress log when progressSteps is non-empty", () => {
  render(
    <AnswerPanel
      answer=""
      citations={[]}
      progressSteps={[{ turn: 1, text: "search_routing_tool · 5 docs" }]}
      completedSteps={[]}
    />
  );
  expect(screen.getByText(/agent reasoning/i)).toBeInTheDocument();
  expect(screen.getByText(/search_routing_tool · 5 docs/i)).toBeInTheDocument();
});

it("renders collapsed summary when progressSteps is empty and completedSteps is non-empty", () => {
  render(
    <AnswerPanel
      answer="Done."
      citations={[]}
      progressSteps={[]}
      completedSteps={[
        { turn: 1, text: "search_routing_tool · 5 docs" },
        { turn: 2, text: "writing answer..." },
      ]}
    />
  );
  expect(screen.getByText(/2 turns/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /show reasoning/i })).toBeInTheDocument();
});

it("expands the log when 'show reasoning' is clicked", async () => {
  render(
    <AnswerPanel
      answer="Done."
      citations={[]}
      progressSteps={[]}
      completedSteps={[{ turn: 1, text: "search_routing_tool · 5 docs" }]}
    />
  );
  const button = screen.getByRole("button", { name: /show reasoning/i });
  await userEvent.click(button);
  expect(screen.getByText(/search_routing_tool · 5 docs/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /hide/i })).toBeInTheDocument();
});

it("renders nothing for progress when both arrays are empty", () => {
  render(
    <AnswerPanel
      answer="answer"
      citations={[]}
      progressSteps={[]}
      completedSteps={[]}
    />
  );
  expect(document.querySelector(".progress-log")).not.toBeInTheDocument();
  expect(document.querySelector(".progress-summary")).not.toBeInTheDocument();
});

// ---------------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------------

it("renders bold text from markdown", () => {
  render(
    <AnswerPanel
      answer="See **bold** text here."
      citations={[]}
    />
  );
  expect(screen.getByText("bold")).toBeInTheDocument();
  const strong = document.querySelector("strong");
  expect(strong).not.toBeNull();
  expect(strong?.textContent).toBe("bold");
});

it("renders citation [D1] as an anchor link", () => {
  render(
    <AnswerPanel
      answer="See [D1] for details."
      citations={[]}
    />
  );
  const link = screen.getByRole("link", { name: "[D1]" });
  expect(link).toHaveAttribute("href", "#source-[D1]");
  expect(link).toHaveClass("citation-link");
});

it("renders citation [R1Q1D1] as an anchor link", () => {
  render(
    <AnswerPanel
      answer="See [R1Q1D1] for details."
      citations={[]}
    />
  );
  const link = screen.getByRole("link", { name: "[R1Q1D1]" });
  expect(link).toHaveAttribute("href", "#source-[R1Q1D1]");
  expect(link).toHaveClass("citation-link");
});

it("renders no anchor when answer has no citation pattern", () => {
  render(
    <AnswerPanel
      answer="No citations here."
      citations={[]}
    />
  );
  expect(screen.queryByRole("link")).not.toBeInTheDocument();
});

it("does not render .citation-row div", () => {
  render(
    <AnswerPanel
      answer="answer"
      citations={["[D1]", "[D2]"]}
    />
  );
  expect(document.querySelector(".citation-row")).not.toBeInTheDocument();
});
