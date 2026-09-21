import { describe, it, expect, vi, beforeEach } from "vitest";
import { runAgent, streamAgent, submitToolApproval } from "../api";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => { mockFetch.mockReset(); });

describe("runAgent", () => {
  it("passes intent field through from response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        session_id: "s1", answer: "hello", citations: [],
        documents: [], messages: [], intent: "search",
      }),
    });
    const result = await runAgent({ query: "find docs" });
    expect(result.intent).toBe("search");
  });
});

describe("submitToolApproval", () => {
  it("posts the approval decision to the pending approval", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "accepted" }),
    });

    await submitToolApproval("a1", "approve");

    expect(mockFetch).toHaveBeenCalledWith("/api/agent/approvals/a1", {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      method: "POST",
      body: JSON.stringify({ decision: "approve" }),
      signal: undefined,
    });
  });
});

describe("streamAgent", () => {
  it("yields error event when response is not ok", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      body: null,
    });
    const gen = streamAgent({ query: "q" });
    await expect(gen.next()).rejects.toThrow("502");
  });

  it("yields claim events", async () => {
    mockFetchStream([
      'data: {"type":"claim","text":"FAISS is a library. [D1]"}\n\n',
      'data: {"type":"answer","text":"FAISS is a library. [D1]"}\n\n',
    ]);
    const types: string[] = [];
    for await (const event of streamAgent({ query: "q" })) types.push(event.type);
    expect(types).toEqual(["claim", "answer"]);
  });

  // The backend sends `: keepalive` on an idle stream so an intermediary does
  // not drop the connection. It must reach the reader as nothing at all.
  it("ignores heartbeat comment frames", async () => {
    mockFetchStream([
      ": keepalive\n\n",
      'data: {"type":"claim","text":"still working"}\n\n',
      ": keepalive\n\n",
      'data: {"type":"answer","text":"done"}\n\n',
    ]);
    const types: string[] = [];
    for await (const event of streamAgent({ query: "q" })) types.push(event.type);
    expect(types).toEqual(["claim", "answer"]);
  });
});

/** Mocks `fetch` to return an SSE body streaming the given raw chunks. */
function mockFetchStream(chunks: string[]) {
  const encoder = new TextEncoder();
  let index = 0;
  mockFetch.mockResolvedValueOnce({
    ok: true,
    body: {
      getReader: () => ({
        read: async () => {
          if (index >= chunks.length) return { done: true, value: undefined };
          const value = encoder.encode(chunks[index]);
          index += 1;
          return { done: false, value };
        },
      }),
    },
  });
}
