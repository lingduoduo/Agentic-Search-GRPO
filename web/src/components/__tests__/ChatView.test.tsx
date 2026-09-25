import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatView } from "../ChatView";
import * as api from "../../api";

describe("ChatView", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("renders the streamed answer", async () => {
    async function* fake() {
      yield { type: "answer", text: "hello there" } as const;
      yield { type: "done", session_id: "s1" } as const;
    }
    vi.spyOn(api, "sendChatMessage").mockImplementation(fake as never);
    render(<ChatView />);
    fireEvent.change(screen.getByLabelText("Chat message"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("hello there")).toBeInTheDocument());
  });

  it("keeps prior turns visible across two submits", async () => {
    const answers = ["first answer", "second answer"];
    let call = 0;
    vi.spyOn(api, "sendChatMessage").mockImplementation((() => {
      const text = answers[call++];
      async function* g() {
        yield { type: "answer", text } as const;
        yield { type: "done", session_id: "s1" } as const;
      }
      return g();
    }) as never);

    render(<ChatView />);
    const input = screen.getByLabelText("Chat message");
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
    vi.spyOn(api, "sendChatMessage").mockImplementation((() => {
      async function* g() { throw new Error("NO_LOCAL_MODEL"); yield undefined as never; }
      return g();
    }) as never);
    render(<ChatView />);
    fireEvent.change(screen.getByLabelText("Chat message"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText(/needs a local model/)).toBeInTheDocument());
  });

  function streamOnce(done: Record<string, unknown>, text = "answer text") {
    return (async function* () {
      yield { type: "answer", text } as const;
      yield { type: "done", session_id: "s1", ...done } as const;
    })();
  }

  async function send(value: string) {
    fireEvent.change(screen.getByLabelText("Chat message"), { target: { value } });
    fireEvent.click(screen.getByText("Send"));
  }

  it("marks a degraded answer with a notice carrying the reason", async () => {
    vi.spyOn(api, "sendChatMessage").mockImplementation(
      (() => streamOnce({ degraded: "model_unavailable" }, "fallback text")) as never,
    );
    render(<ChatView />);
    await send("hi");
    const notice = await screen.findByRole("status");
    expect(notice).toHaveTextContent("⚠ Model unavailable — degraded answer");
    expect(notice).toHaveAttribute("title", "model_unavailable");
  });

  it("shows no notice when done omits degraded", async () => {
    vi.spyOn(api, "sendChatMessage").mockImplementation((() => streamOnce({})) as never);
    render(<ChatView />);
    await send("hi");
    await screen.findByText("answer text");
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("shows no notice when done carries degraded: null", async () => {
    vi.spyOn(api, "sendChatMessage").mockImplementation(
      (() => streamOnce({ degraded: null })) as never,
    );
    render(<ChatView />);
    await send("hi");
    await screen.findByText("answer text");
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("keeps the notice on the degraded turn only", async () => {
    const dones = [{ degraded: "model_unavailable" }, {}];
    const texts = ["degraded one", "normal two"];
    let call = 0;
    vi.spyOn(api, "sendChatMessage").mockImplementation((() => {
      const i = call++;
      return streamOnce(dones[i], texts[i]);
    }) as never);
    render(<ChatView />);
    await send("q1");
    await screen.findByText("degraded one");
    await send("q2");
    await screen.findByText("normal two");
    const notices = screen.getAllByRole("status");
    expect(notices).toHaveLength(1);
    expect(screen.getByText("degraded one").closest(".turn")).toContainElement(notices[0]);
  });
});
