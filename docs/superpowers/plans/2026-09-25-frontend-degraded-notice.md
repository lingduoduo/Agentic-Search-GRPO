# Frontend Degraded-Answer Notice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/chat` and `/tools` show a visible notice on an assistant answer that the backend marked `degraded`, so a degraded answer no longer looks like a normal one.

**Architecture:** The SSE `done` event types gain `degraded?: string | null`; `ChatView` and `ToolAgentView` copy it onto the last assistant `ConversationTurn` (new optional field `degraded`); the shared `Transcript` renders the notice for any assistant turn whose `degraded` is set. One render site serves both views.

**Tech Stack:** React 19, TypeScript, Vite, Vitest + Testing Library (jsdom).

**Spec:** `docs/superpowers/specs/2026-09-25-frontend-degraded-notice-design.md`

## Global Constraints

- Notice text exactly: `⚠ Model unavailable — degraded answer`.
- Notice `title` attribute = the raw reason (e.g. `model_unavailable`).
- Notice has `role="status"`.
- Re-use the `route-pill--degraded` colours (`#fdf0d5` background, `#8a5a00` text); a minimal new class goes next to it in `web/src/styles.css`.
- Rendering unchanged when `degraded` is absent or null. No backend change. No component library.
- Branch `feat/frontend-degraded-notice`; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not run `npm install` (`web/node_modules` is a shared symlink).

## Spec deviations (code differs from the spec's assumptions)

1. **No non-stream response path exists in the frontend.** `sendChatMessage` and `sendToolMessage` (`web/src/api.ts`) always send `stream: true` and yield SSE events; `ChatView`/`ToolAgentView` consume only the stream, and `web/src/types.ts` has no chat/tool JSON response type. So the "non-stream response" type change and test cases have nothing to attach to. Adding a non-stream client path would be new, unrequested code. Instead: only the `done` event types gain `degraded`, and each view's third test case becomes "`done` carries `degraded: null` → no notice" (the null half of the contract).
2. **The notice renders in `Transcript`** (shared by both views) rather than separately in each view: both views already render assistant turns through it, so one render site is the minimal change. The per-view tests still drive it end to end through each view.
3. **New class `.degraded-notice`** instead of re-using `route-pill--degraded` directly: `.route-pill` carries monospace font and pill sizing meant for the page-actions bar; the new class copies only the colours and sits next to `.route-pill--degraded`.

## Review Focus

1. **A degraded turn followed by a normal turn** — the second answer must not show the notice, while the first keeps it (the flag is per turn, not view-wide) — test in Task 1.
2. **`done` omits `degraded`** (the normal backend stream omits the key entirely) — no notice — test in Task 1 and Task 2.
3. **`done` carries `degraded: null`** — no notice (null must not render as a truthy value or as the text "null") — test in Task 1 and Task 2.
4. **`/tools` answer both truncated and degraded** — both the truncation notice and the degraded notice appear; existing truncation tests use `findByRole("status")`, which must stay unambiguous when `degraded` is absent — existing tests stay green in Task 2.
5. **Stream ends with `error` instead of `done`** — no degraded notice (nothing set it) — covered by construction: only the `done` branch writes `degraded`; no separate test.

---

## File Structure

- Modify `web/src/types.ts` — `ChatStreamEvent` / `ToolStreamEvent` `done` gain `degraded?: string | null`; `ConversationTurn` gains `degraded?: string | null`.
- Modify `web/src/components/Transcript.tsx` — render the notice for an assistant turn with `degraded`.
- Modify `web/src/components/ChatView.tsx` — copy `e.degraded` onto the last assistant turn on `done`.
- Modify `web/src/components/ToolAgentView.tsx` — same.
- Modify `web/src/styles.css` — `.degraded-notice` next to `.route-pill--degraded`.
- Tests: `web/src/components/__tests__/ChatView.test.tsx`, `web/src/components/__tests__/ToolAgentView.test.tsx`.

---

### Task 1: Types, Transcript notice, ChatView

**Files:**
- Modify: `web/src/types.ts` (ChatStreamEvent, ToolStreamEvent `done`, ConversationTurn)
- Modify: `web/src/components/Transcript.tsx`
- Modify: `web/src/components/ChatView.tsx` (`done` branch)
- Modify: `web/src/styles.css` (after `.route-pill--degraded`)
- Test: `web/src/components/__tests__/ChatView.test.tsx`

**Interfaces:**
- Produces: `ConversationTurn.degraded?: string | null`; `done` events of `ChatStreamEvent` and `ToolStreamEvent` with `degraded?: string | null`; `Transcript` renders `<p className="degraded-notice" role="status" title={reason}>⚠ Model unavailable — degraded answer</p>` for an assistant turn with truthy `degraded`.

- [x] **Step 1: Write the failing tests** (append inside the `describe("ChatView", ...)` block)

```tsx
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
```

Also add `beforeEach(() => vi.restoreAllMocks());` at the top of the describe (import `beforeEach`) so spies do not leak between the new tests.

- [x] **Step 2: Run to verify they fail**

Run (from `web/`): `npx vitest run src/components/__tests__/ChatView.test.tsx`
Expected: "marks a degraded answer…" and "keeps the notice…" FAIL (no `status` role); the two no-notice tests pass already. Typecheck of the `degraded` spread is not enforced by vitest.

- [x] **Step 3: Implement**

`web/src/types.ts`:

```ts
export type ChatStreamEvent =
  | { type: "answer"; text: string }
  // `degraded` names why the answer is a fallback (e.g. "model_unavailable").
  | { type: "done"; session_id: string; degraded?: string | null }
  | { type: "error"; detail: string };
```

In `ToolStreamEvent`'s `done` member, after `truncated?: boolean;`:

```ts
      // Why the answer is a fallback (e.g. "model_unavailable"); absent/null when not.
      degraded?: string | null;
```

In `ConversationTurn`, after `pending?: boolean;`:

```ts
  degraded?: string | null;
```

`web/src/components/ChatView.tsx` `done` branch:

```tsx
          patchLastAssistant({ pending: false, degraded: e.degraded ?? null });
```

`web/src/components/Transcript.tsx`, after the `turn__content` line:

```tsx
          {turn.role === "assistant" && turn.degraded && (
            <p className="degraded-notice" role="status" title={turn.degraded}>
              ⚠ Model unavailable — degraded answer
            </p>
          )}
```

`web/src/styles.css`, directly after the `.route-pill--degraded` rule:

```css
/* A degraded answer (the model was unavailable) carries the same amber. */
.degraded-notice {
  display: inline-block;
  margin: 6px 0 0;
  border-radius: 999px;
  padding: 3px 9px;
  font-size: 12px;
  font-weight: 750;
  background: #fdf0d5;
  color: #8a5a00;
}
```

- [x] **Step 4: Run to verify they pass**

Run: `npx vitest run src/components/__tests__/ChatView.test.tsx` → all PASS. Then `npm run typecheck` → exit 0.

- [x] **Step 5: Commit**

```bash
git add web/src/types.ts web/src/components/Transcript.tsx web/src/components/ChatView.tsx web/src/styles.css web/src/components/__tests__/ChatView.test.tsx
git commit -m "Chat marks a degraded answer with a notice"
```

### Task 2: ToolAgentView

**Files:**
- Modify: `web/src/components/ToolAgentView.tsx` (`done` branch)
- Test: `web/src/components/__tests__/ToolAgentView.test.tsx`

**Interfaces:**
- Consumes: `ToolStreamEvent` `done.degraded?: string | null`, `ConversationTurn.degraded`, and the Transcript notice from Task 1.

- [x] **Step 1: Write the failing tests** (new describe block at the end of the file)

```tsx
describe("ToolAgentView degraded notice", () => {
  beforeEach(() => vi.restoreAllMocks());

  function streamOnce(done: Record<string, unknown>, text = "search-only list") {
    return (async function* () {
      yield { type: "answer", text } as const;
      yield {
        type: "done",
        session_id: "s1",
        tool_calls: [],
        num_turns: 0,
        truncated: false,
        ...done,
      } as const;
    })();
  }

  async function send() {
    fireEvent.change(screen.getByLabelText("Tool agent message"), {
      target: { value: "find docs" },
    });
    fireEvent.click(screen.getByText("Send"));
  }

  it("marks a degraded answer with a notice carrying the reason", async () => {
    vi.spyOn(api, "sendToolMessage").mockImplementation(
      (() => streamOnce({ degraded: "model_unavailable" })) as never,
    );
    render(<ToolAgentView />);
    await send();
    const notice = await screen.findByRole("status");
    expect(notice).toHaveTextContent("⚠ Model unavailable — degraded answer");
    expect(notice).toHaveAttribute("title", "model_unavailable");
  });

  it("shows no notice when done omits degraded", async () => {
    vi.spyOn(api, "sendToolMessage").mockImplementation((() => streamOnce({})) as never);
    render(<ToolAgentView />);
    await send();
    await screen.findByText("search-only list");
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("shows no notice when done carries degraded: null", async () => {
    vi.spyOn(api, "sendToolMessage").mockImplementation(
      (() => streamOnce({ degraded: null })) as never,
    );
    render(<ToolAgentView />);
    await send();
    await screen.findByText("search-only list");
    expect(screen.queryByRole("status")).toBeNull();
  });
});
```

- [x] **Step 2: Run to verify it fails**

Run: `npx vitest run src/components/__tests__/ToolAgentView.test.tsx`
Expected: "marks a degraded answer…" FAIL (view drops `degraded`).

- [x] **Step 3: Implement** — `ToolAgentView.tsx` `done` branch:

```tsx
          patchLastAssistant((t) => ({ ...t, pending: false, degraded: e.degraded ?? null }));
```

- [x] **Step 4: Run to verify it passes**

Run: `npx vitest run src/components/__tests__/ToolAgentView.test.tsx` → all PASS (including the existing truncation tests).

- [x] **Step 5: Commit**

```bash
git add web/src/components/ToolAgentView.tsx web/src/components/__tests__/ToolAgentView.test.tsx
git commit -m "Tool agent marks a degraded answer with a notice"
```

### Task 3: Mutation check and final verification

- [x] **Step 1:** Remove the notice render block from `Transcript.tsx`; run `npx vitest run src/components/__tests__/ChatView.test.tsx src/components/__tests__/ToolAgentView.test.tsx`. Expected: the three "marks a degraded answer" / "keeps the notice" tests FAIL. Restore with `git checkout web/src/components/Transcript.tsx`; confirm `git diff` is empty.
- [x] **Step 2:** From `web/`: `npm run typecheck` (exit 0) and `npm run test:unit` (all pass). From the worktree root: `git diff --check origin/main...HEAD` (no output).
