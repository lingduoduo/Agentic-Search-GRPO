import { render, screen, waitFor, within } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { LatencyPanel } from "../LatencyPanel";
import * as api from "../../../api";

const row = {
  method: "POST",
  route: "/api/agent",
  count: 12,
  errors: 0,
  p50_ms: 812.4,
  p95_ms: 2140.9,
  max_ms: 2210,
};

describe("LatencyPanel", () => {
  it("renders one row per route with its percentiles", async () => {
    vi.spyOn(api, "getRouteLatency").mockResolvedValue({ routes: [row] });

    render(<LatencyPanel />);

    await waitFor(() =>
      expect(screen.getByText("POST /api/agent")).toBeInTheDocument(),
    );
    expect(screen.getByText("812.4")).toBeInTheDocument();
    expect(screen.getByText("2140.9")).toBeInTheDocument();
  });

  it("renders the retrieval and generation stage rows apart from the routes", async () => {
    vi.spyOn(api, "getRouteLatency").mockResolvedValue({
      routes: [row],
      stages: {
        retrieval: { count: 12, p50_ms: 4.2, p95_ms: 9.9, max_ms: 12, avg_docs: 5, cache_hit_rate: 0.25 },
        generation: {
          count: 12,
          p50_ms: 800.1,
          p95_ms: 2100.5,
          max_ms: 2200,
          avg_prompt_tokens: 512,
          avg_completion_tokens: 64,
        },
      },
    });

    render(<LatencyPanel />);

    const stages = await screen.findByRole("table", { name: /stage latency/i });
    expect(within(stages).getByText("Retrieval")).toBeInTheDocument();
    expect(within(stages).getByText("Generation")).toBeInTheDocument();
    expect(within(stages).getByText("4.2")).toBeInTheDocument();
    expect(within(stages).getByText("2100.5")).toBeInTheDocument();
    expect(within(stages).getByText(/5\.0 docs · 25% cache hits/)).toBeInTheDocument();
    expect(within(stages).getByText(/512 prompt · 64 completion tokens/)).toBeInTheDocument();
  });

  it("omits the stage table when no request used either stage", async () => {
    vi.spyOn(api, "getRouteLatency").mockResolvedValue({
      routes: [row],
      stages: { retrieval: { count: 0 }, generation: { count: 0 } },
    });

    render(<LatencyPanel />);

    await waitFor(() => expect(screen.getByText("POST /api/agent")).toBeInTheDocument());
    expect(screen.queryByRole("table", { name: /stage latency/i })).not.toBeInTheDocument();
  });

  it("shows an empty state before any request is recorded", async () => {
    vi.spyOn(api, "getRouteLatency").mockResolvedValue({ routes: [] });

    render(<LatencyPanel />);

    await waitFor(() =>
      expect(screen.getByText(/no requests recorded yet/i)).toBeInTheDocument(),
    );
  });

  it("renders a dash instead of crashing on a missing percentile", async () => {
    vi.spyOn(api, "getRouteLatency").mockResolvedValue({
      routes: [{ ...row, p95_ms: null as unknown as number }],
    });

    render(<LatencyPanel />);

    await waitFor(() =>
      expect(screen.getByText("POST /api/agent")).toBeInTheDocument(),
    );
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("falls back to the empty state when the endpoint is unreachable", async () => {
    vi.spyOn(api, "getRouteLatency").mockRejectedValue(new Error("404"));

    render(<LatencyPanel />);

    await waitFor(() =>
      expect(screen.getByText(/no requests recorded yet/i)).toBeInTheDocument(),
    );
  });
});
