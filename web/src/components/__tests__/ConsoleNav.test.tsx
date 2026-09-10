import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { App } from "../../App";

vi.mock("../../api", () => ({
  createSession: vi
    .fn()
    .mockResolvedValue({ id: "s1", messages: [], title: null, user_id: null }),
  runAgent: vi.fn(),
  streamAgent: vi.fn(),
  getAdminSummary: vi.fn().mockRejectedValue(new Error("no admin")),
  getAnalyticsByLLM: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByPersona: vi.fn().mockRejectedValue(new Error()),
  getAnalyticsByFlow: vi.fn().mockRejectedValue(new Error()),
  runDebugRetrieval: vi.fn(),
  getServerHealth: vi.fn().mockResolvedValue({ servers: [] }),
  getEvalResults: vi.fn().mockResolvedValue({ results: [] }),
  getRouteLatency: vi.fn().mockResolvedValue({ routes: [] }),
  getDebugTools: vi.fn().mockResolvedValue({ registered: [], catalog: [] }),
  discoverTools: vi.fn(),
  runQueryTransform: vi.fn(),
  submitSessionFeedback: vi.fn().mockResolvedValue({ ok: true }),
}));

beforeEach(() => vi.clearAllMocks());
afterEach(() => vi.unstubAllEnvs());

describe("Console nav (DEBUG_PANELS gate)", () => {
  it("hides the Console toggle when the flag is off", () => {
    vi.stubEnv("VITE_DEBUG_PANELS", "");
    render(<App />);
    expect(
      screen.queryByRole("button", { name: /console/i }),
    ).not.toBeInTheDocument();
  });

  it("shows the Console toggle and opens the Retrieval Lab when enabled", async () => {
    vi.stubEnv("VITE_DEBUG_PANELS", "1");
    render(<App />);
    const toggle = screen.getByRole("button", { name: /console/i });
    expect(toggle).toBeInTheDocument();
    await userEvent.click(toggle);
    expect(
      screen.getByRole("heading", { name: /retrieval lab/i }),
    ).toBeInTheDocument();
  });
});
