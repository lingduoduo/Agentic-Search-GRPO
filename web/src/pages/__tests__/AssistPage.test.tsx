import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AssistPage } from "../AssistPage";
import * as api from "../../api";

describe("AssistPage claim streaming", () => {
  it("renders claim text as it streams in, joined with a space", async () => {
    vi.spyOn(api, "createSession").mockResolvedValue({
      id: "s1",
      messages: [],
    } as never);
    async function* fake() {
      yield { type: "claim", text: "FAISS is a library." } as const;
      yield { type: "claim", text: "It supports GPU search." } as const;
    }
    vi.spyOn(api, "streamAgent").mockImplementation(fake as never);

    render(<AssistPage />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "what is faiss" } });
    fireEvent.click(screen.getByText("Search"));

    await waitFor(() =>
      expect(
        screen.getByText("FAISS is a library. It supports GPU search."),
      ).toBeInTheDocument(),
    );
  });

  it("overwrites the streamed claim text with the authoritative answer event", async () => {
    vi.spyOn(api, "createSession").mockResolvedValue({
      id: "s1",
      messages: [],
    } as never);
    async function* fake() {
      yield { type: "claim", text: "FAISS is a library." } as const;
      // Simulate a dropped claim event: the answer event carries the complete
      // text, which must replace (not append to) what streamed in.
      yield {
        type: "answer",
        text: "FAISS is a library. It also supports GPU search.",
      } as const;
      yield { type: "done", session_id: "s1", citations: [], documents: [] } as const;
    }
    vi.spyOn(api, "streamAgent").mockImplementation(fake as never);

    render(<AssistPage />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "what is faiss" } });
    fireEvent.click(screen.getByText("Search"));

    await waitFor(() =>
      expect(
        screen.getAllByText("FAISS is a library. It also supports GPU search.").length,
      ).toBeGreaterThan(0),
    );
    expect(screen.queryByText(/FAISS is a library\. FAISS is a library\./)).not.toBeInTheDocument();
  });
});


describe("AssistPage search-source taxonomy", () => {
  it("keeps web available when discovery fails", async () => {
    vi.spyOn(api, "fetchSearchDomains").mockRejectedValue(new Error("unavailable"));
    render(<AssistPage />);
    await waitFor(() => expect(api.fetchSearchDomains).toHaveBeenCalled());
    expect(screen.getByLabelText("Search source")).toHaveValue("web");
  });

  it("submits the selected canonical source", async () => {
    vi.spyOn(api, "fetchSearchDomains").mockResolvedValue({domains: [
      {name: "web", description: "General information", typical_sources: ["Web search engines"]},
      {name: "financial", description: "Market data", typical_sources: ["SEC EDGAR"]},
    ]});
    vi.spyOn(api, "createSession").mockResolvedValue({id: "source-session", messages: []} as never);
    async function* fake() { yield {type: "answer", text: "Result"} as const; }
    const stream = vi.spyOn(api, "streamAgent").mockImplementation(fake as never);
    render(<AssistPage />);
    await screen.findByRole("option", {name: "financial"});
    fireEvent.change(screen.getByLabelText("Search source"), {target: {value: "financial"}});
    fireEvent.change(screen.getByLabelText("Question"), {target: {value: "earnings"}});
    fireEvent.click(screen.getByText("Search"));
    await waitFor(() => expect(stream).toHaveBeenCalled());
    expect(stream.mock.calls[0][0].domain).toBe("financial");
  });
});
