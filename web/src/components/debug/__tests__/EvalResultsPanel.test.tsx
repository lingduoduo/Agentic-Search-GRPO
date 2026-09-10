import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { EvalResultsPanel } from "../EvalResultsPanel";
import * as api from "../../../api";

describe("EvalResultsPanel", () => {
  it("renders a card with metric rows per result file", async () => {
    vi.spyOn(api, "getEvalResults").mockResolvedValue({
      results: [
        { name: "beir.json", modified: 1_700_000_000, metrics: { "recall@10": 0.5 } },
      ],
    });
    render(<EvalResultsPanel />);
    await waitFor(() => expect(screen.getByText("beir.json")).toBeInTheDocument());
    expect(screen.getByText("recall@10")).toBeInTheDocument();
    expect(screen.getByText("0.5000")).toBeInTheDocument();
  });

  it("shows retrieval metrics under their own heading, apart from generation", async () => {
    vi.spyOn(api, "getEvalResults").mockResolvedValue({
      results: [
        {
          name: "bamboogle.summary.json",
          modified: 1_700_000_000,
          metrics: { exact_match: 0.4, "retrieval.recall@10": 0.5, num_examples: 125 },
          groups: {
            retrieval: { "retrieval.recall@10": 0.5 },
            generation: { exact_match: 0.4 },
            other: { num_examples: 125 },
          },
        },
      ],
    });
    render(<EvalResultsPanel />);
    await waitFor(() => expect(screen.getByText("bamboogle.summary.json")).toBeInTheDocument());

    const retrieval = screen.getByRole("table", { name: "Retrieval" });
    expect(within(retrieval).getByText("retrieval.recall@10")).toBeInTheDocument();
    expect(within(retrieval).queryByText("exact_match")).not.toBeInTheDocument();
    const generation = screen.getByRole("table", { name: "Generation" });
    expect(within(generation).getByText("exact_match")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Other" })).toBeInTheDocument();
    // Groups render in taxonomy order: retrieval before generation before other.
    const captions = screen.getAllByRole("table").map((t) => t.querySelector("caption")?.textContent);
    expect(captions).toEqual(["Retrieval", "Generation", "Other"]);
    expect(screen.queryByRole("table", { name: "Reward" })).not.toBeInTheDocument();
  });

  it("shows an empty state when there are no results", async () => {
    vi.spyOn(api, "getEvalResults").mockResolvedValue({ results: [] });
    render(<EvalResultsPanel />);
    await waitFor(() =>
      expect(screen.getByText(/no eval results yet/i)).toBeInTheDocument(),
    );
  });
});
