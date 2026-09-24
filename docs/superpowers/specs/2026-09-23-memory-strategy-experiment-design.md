# Sliding window vs rolling summary — experiment design

## Question

Every chat surface hands the model the last `MAX_HISTORY_MESSAGES = 40` messages
(`load_working_memory`, `src/internal/memory/working.py`); the search agent trims
further to 6. PR #578 added a rolling summary of the dropped prefix behind
`AGENTIC_SEARCH_MEMORY_COMPRESSION` (default off). Nothing measures either. This
experiment answers: **how much earlier-conversation recall does each strategy
keep, at what prompt and summarizer cost?**

## Workload

Synthetic, seeded conversations, so every probe has an exact answer and the
distance to the fact is controlled.

- 12 conversations × ~80 messages (user/assistant alternating, scripted — no LLM
  writes the transcript).
- 4 facts per conversation, each an attribute with a unique value that appears
  nowhere else (e.g. `my budget is $420`, `my flight is LH452`), planted at
  known message indices spread across the conversation.
- Filler: neutral scripted turns, ~40 tokens per message, so the **full history
  fits the local model's 4,096-token context** — otherwise Ollama silently drops
  the head and "full history" degenerates into a window.
- After the transcript, one probe per fact: `What is my <attribute>?`.
- Scoring: correct iff the normalized answer contains the normalized value. No
  LLM judge.

## Strategies

Each runs the repo's own code; nothing is reimplemented.

| strategy | how |
|---|---|
| `full` | every message (upper bound) |
| `window-N` | `load_working_memory(keep_last=N, cache=None)` |
| `summary-N` | `load_working_memory(keep_last=N, cache=<in-memory>)` + `compress_session` |

N ∈ {6, 10, 20, 40}. Summary runs are paired with windows at equal N.

**Timing fidelity.** Production (`chat_backend.send_chat_message`) loads working
memory, then schedules `compress_session(pending)` concurrently with the answer.
So on turn T the summary covers only what earlier turns compressed, and messages
that fell out of the window since the last compression are in neither the tail
nor the summary for one turn. The harness replays the transcript turn by turn and
compresses after each turn's load (awaited, so runs are deterministic in order),
which reproduces that lag exactly. Probes are asked turn by turn after the
transcript, each probe a user turn, as production would see them.

## Answering

Plain chat (`plain_chat_runner._run_plain_chat`) sends `history + [user
message]` with no system prompt of its own. The harness sends that identical
message list to `llama3.2:3b` through Ollama's OpenAI-compatible endpoint; the
summary, when present, is the leading system message exactly as production
builds it. The same model is the summarizer (production uses the configured
remote LLM; the local 3B model is the only one that runs here), so summary
quality is a stated confound.

## Metrics, per strategy

- recall accuracy, overall and split by whether the fact was inside the window
  at probe time (`in_window`) or had been dropped (`dropped`);
- prompt tokens per probe (Ollama `usage.prompt_tokens`), and any probe at or
  above 4,096 flagged as `context_overflow`;
- summarizer calls and total summarizer seconds;
- answer latency.

## Deliverables

- `examples/measure_memory_strategies.py` (generator, scorer, runner, JSON report
  to `data/eval/memory_strategies.json`).
- Unit tests for the generator (determinism, fact uniqueness, token budget) and
  the scorer, plus a fake-LLM run of the harness proving `summary-N` actually
  sends the summary and `window-N` does not.
- Results in the PR description.

## Out of scope

Fact extraction / long-term memory recall (#579/#580), token-budget windows, and
changing any default. The experiment informs whether to turn the flag on or
change N; it changes neither.
