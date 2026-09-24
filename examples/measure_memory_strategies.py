"""Sliding window vs rolling summary: how much earlier-conversation recall each keeps.

Every chat surface hands its model the last ``MAX_HISTORY_MESSAGES`` messages
(``load_working_memory``); PR #578 added a rolling summary of the dropped prefix
behind ``AGENTIC_SEARCH_MEMORY_COMPRESSION``. This replays seeded synthetic
conversations -- facts planted at known positions, then one probe per fact --
through the repo's own ``load_working_memory`` and ``compress_session``, and
scores each answer exactly.

Replay follows production's order (``chat_backend.send_chat_message``): load
working memory, add the user turn, compress the pending span, answer. So a
summary is always built from earlier turns, and a message that fell out of the
window since the last compression is in neither the tail nor the summary for
one turn.

The answer prompt is what plain chat sends -- history plus the user message, no
system prompt of its own -- so a summary arrives as the leading system message,
exactly as production builds it.

Run (needs Ollama with the model pulled; ~1-1.5 h for the full grid):

    python -m examples.measure_memory_strategies \\
        --llm_model llama3.2:3b --out data/eval/memory_strategies.json

The same local model summarizes and answers. Production summarizes with the
configured remote LLM, so summary quality here is a lower bound.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.internal.cache.interface import InMemoryCache
from src.internal.db.store import AgenticSearchStore
from src.internal.memory.working import compress_session, load_working_memory

# Ollama's default context for llama3.2 via the OpenAI endpoint. A prompt at or
# above it has been silently truncated from the head, so it is flagged.
CONTEXT_TOKENS = 4096

DEFAULT_PAIRS = 40
DEFAULT_FACT_PAIRS = (2, 12, 22, 32)

_DOGS = [
    "Biscuit", "Juniper", "Pickle", "Waffles", "Nimbus", "Tofu", "Marzipan",
    "Clementine", "Pistachio", "Gizmo", "Noodle", "Sprocket", "Quince",
    "Rutabaga", "Zephyr", "Kumquat",
]  # fmt: skip
_HOTELS = [
    "Aurelia", "Belmonte", "Castellan", "Dunmore", "Elsinore", "Fairhaven",
    "Glenrock", "Harrowgate", "Ivybridge", "Kestrel", "Larkspur", "Moorland",
    "Northcliff", "Oakhurst", "Pemberly", "Ravenholt",
]  # fmt: skip


def _values(rng: random.Random, n: int) -> dict[str, list[str]]:
    """``n`` distinct values per attribute, sampled without replacement."""
    letters = "BCDFGHJKMNPQRSTVWXZ"
    flights = [f"{a}{b}{d}" for a in letters for b in letters for d in range(100, 1000)]
    rooms = [f"{a}-{d}" for a in "ABCDEFGH" for d in range(100, 1000)]
    return {
        "budget": [f"${v}" for v in rng.sample(range(100, 1000), n)],
        "flight": rng.sample(flights, n),
        "dog": rng.sample(_DOGS, n),
        "locker": [str(v) for v in rng.sample(range(1000, 10000), n)],
        "hotel": rng.sample(_HOTELS, n),
        "room": rng.sample(rooms, n),
    }


# attribute -> (user statement, assistant acknowledgement, probe question)
_FACTS = {
    "budget": (
        "By the way, my budget for the trip is {v}. Please keep that in mind.",
        "Noted: a trip budget of {v}.",
        "What is my budget for the trip?",
    ),
    "flight": (
        "Quick thing to remember: my flight number is {v}.",
        "Got it, your flight number is {v}.",
        "What is my flight number?",
    ),
    "dog": (
        "Oh, and my dog's name is {v}, in case it comes up.",
        "Lovely, {v} is a great name for a dog.",
        "What is my dog's name?",
    ),
    "locker": (
        "Remind me later: the code for my gym locker is {v}.",
        "I'll remember that your gym locker code is {v}.",
        "What is the code for my gym locker?",
    ),
    "hotel": (
        "For the record, I'm staying at the {v} hotel this week.",
        "Understood, you're at the {v} hotel.",
        "Which hotel am I staying at?",
    ),
    "room": (
        "Note that my meeting tomorrow is in room {v}.",
        "Okay, meeting room {v} it is.",
        "Which room is my meeting in?",
    ),
}

# Filler carries no digits and none of the value pools, so a guess can only be
# credited if the value came from the planted fact.
_FILLER_USER = [
    "I have been trying to cook more at home lately. Any easy ideas for a quick weeknight dinner?",
    "What is a good way to keep houseplants alive when I travel for a few days?",
    "Can you suggest a relaxing way to spend a rainy Sunday afternoon indoors?",
    "I want to start running again. How should a beginner ease back into it?",
    "Do you have tips for staying focused when working from home all day?",
    "What should I keep in mind when choosing a good pair of walking shoes?",
    "How can I make my morning routine a little less rushed and chaotic?",
    "Any recommendations for learning to play the guitar as a total beginner?",
    "What is a simple way to organize a messy kitchen drawer for good?",
    "How do people usually keep a small balcony garden looking nice?",
    "I keep forgetting to drink enough water. Any practical tricks for that?",
    "What makes a good book club pick for a group with mixed tastes?",
]
_FILLER_ASSISTANT = [
    "A stir fry is hard to beat: slice whatever vegetables you have, add a protein, and finish with a simple sauce.",
    "Group the plants together away from direct sun, water them well beforehand, and they usually cope fine.",
    "A warm drink, a long novel, and maybe a slow cooking project make a rainy afternoon feel pleasant.",
    "Start with short easy sessions, alternate walking and jogging, and increase the time gradually each week.",
    "Set clear working hours, take short breaks away from the screen, and keep a separate space for work.",
    "Look for good cushioning, a roomy toe box, and try them on later in the day when feet are larger.",
    "Prepare what you can the night before, wake a little earlier, and keep the first hour free of screens.",
    "Learn a few open chords first, practice a little every day, and pick songs you genuinely enjoy.",
    "Empty it completely, discard what you never use, and add small dividers so everything has a place.",
    "Choose plants suited to the light you get, water in the morning, and trim them regularly.",
    "Keep a bottle within reach, pair a glass of water with each meal, and set gentle reminders.",
    "Something short and discussable works well, ideally a story with a strong central question.",
]


@dataclass(frozen=True)
class Fact:
    index: int  # message index of the user turn that states the fact
    attribute: str
    value: str
    question: str


@dataclass(frozen=True)
class Conversation:
    id: int
    messages: tuple[tuple[str, str], ...]
    facts: tuple[Fact, ...]
    probe_order: tuple[int, ...]  # fact indices, in the order they are probed


def make_dataset(
    n: int,
    *,
    seed: int = 0,
    pairs: int = DEFAULT_PAIRS,
    fact_pairs: tuple[int, ...] = DEFAULT_FACT_PAIRS,
) -> list[Conversation]:
    """``n`` seeded conversations; every fact value is unique across the set."""
    rng = random.Random(seed)
    pools = _values(rng, n)
    conversations = []
    for cid in range(n):
        attributes = rng.sample(sorted(_FACTS), len(fact_pairs))
        planted = dict(zip(fact_pairs, attributes))
        messages: list[tuple[str, str]] = []
        facts: list[Fact] = []
        for pair in range(pairs):
            if pair in planted:
                attribute = planted[pair]
                value = pools[attribute][cid]
                user, assistant, question = _FACTS[attribute]
                facts.append(Fact(len(messages), attribute, value, question))
                messages.append(("user", user.format(v=value)))
                messages.append(("assistant", assistant.format(v=value)))
            else:
                messages.append(("user", rng.choice(_FILLER_USER)))
                messages.append(("assistant", rng.choice(_FILLER_ASSISTANT)))
        order = [f.index for f in facts]
        rng.shuffle(order)
        conversations.append(
            Conversation(cid, tuple(messages), tuple(facts), tuple(order))
        )
    return conversations


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score(answer: str, value: str) -> bool:
    """True when the answer states ``value`` as a whole.

    Case, currency signs, punctuation and spacing are ignored ("lh 452" states
    "LH452"), but the value must be a run of whole answer tokens, so "$4200"
    does not state "$420".
    """
    target = "".join(_tokens(value))
    words = _tokens(answer)
    for start in range(len(words)):
        joined = ""
        for word in words[start:]:
            joined += word
            if joined == target:
                return True
            if len(joined) >= len(target):
                break
    return False


@dataclass(frozen=True)
class Strategy:
    kind: str  # "full" | "window" | "summary"
    window: int | None

    @property
    def name(self) -> str:
        return "full" if self.kind == "full" else f"{self.kind}-{self.window}"


@dataclass(frozen=True)
class ProbeResult:
    strategy: str
    conversation: int
    fact_index: int
    correct: bool
    in_window: bool  # the fact's message was inside the tail the model saw
    in_summary: bool  # the fact's value appeared in the summary it saw
    prompt_tokens: int | None
    latency_s: float
    answer: str = ""


# (messages) -> (answer text, prompt tokens, seconds)
AnswerFn = Callable[[list[dict]], tuple[str, int | None, float]]


async def run_strategy(
    conv: Conversation,
    strategy: Strategy,
    answer_fn: AnswerFn,
    summarizer,
    *,
    stats: dict | None = None,
) -> list[ProbeResult]:
    """Replay one conversation under one strategy; return one result per probe.

    Each run gets its own store and cache, so no summary leaks between runs.
    """
    stats = stats if stats is not None else {}
    stats.setdefault("summarizer_calls", 0)
    stats.setdefault("summarizer_advanced", 0)
    stats.setdefault("summarizer_seconds", 0.0)

    store = AgenticSearchStore(":memory:")
    session_id = store.create_chat_session(title=f"conv-{conv.id}").id
    cache = InMemoryCache() if strategy.kind == "summary" else None
    keep = strategy.window if strategy.window is not None else 10**9
    # Monotonic ids: list_chat_messages orders by created_at then id, and a
    # tight loop can write two messages in the same microsecond.
    seq = itertools.count()

    def add(role: str, content: str) -> None:
        store.add_chat_message(
            session_id, role=role, content=content, message_id=f"m{next(seq):06d}"
        )

    async def turn(user: str):
        working = load_working_memory(store, session_id, keep_last=keep, cache=cache)
        before = len(store.list_chat_messages(session_id))
        add("user", user)
        if cache is not None and working.pending:
            stats["summarizer_calls"] += 1
            started = time.perf_counter()
            advanced = await compress_session(
                session_id, summarizer, pending=working.pending, cache=cache
            )
            stats["summarizer_seconds"] += time.perf_counter() - started
            stats["summarizer_advanced"] += int(advanced)
        return working, before

    for i in range(0, len(conv.messages), 2):
        await turn(conv.messages[i][1])
        add("assistant", conv.messages[i + 1][1])

    facts = {f.index: f for f in conv.facts}
    results = []
    for fact_index in conv.probe_order:
        fact = facts[fact_index]
        working, before = await turn(fact.question)
        messages = [{"role": m.role, "content": m.content} for m in working.messages]
        messages.append({"role": "user", "content": fact.question})
        answer, prompt_tokens, latency = await asyncio.to_thread(answer_fn, messages)
        add("assistant", answer)
        results.append(
            ProbeResult(
                strategy=strategy.name,
                conversation=conv.id,
                fact_index=fact_index,
                correct=score(answer, fact.value),
                in_window=fact_index >= before - keep,
                in_summary=bool(working.summary) and score(working.summary, fact.value),
                prompt_tokens=prompt_tokens,
                latency_s=latency,
                answer=answer,
            )
        )
    return results


def _share(rows: list[ProbeResult]) -> float | None:
    return round(sum(r.correct for r in rows) / len(rows), 3) if rows else None


def aggregate(rows: list[ProbeResult], stats: dict[str, dict]) -> dict:
    report: dict[str, dict] = {}
    for name in dict.fromkeys(r.strategy for r in rows):
        mine = [r for r in rows if r.strategy == name]
        dropped = [r for r in mine if not r.in_window]
        tokens = [r.prompt_tokens for r in mine if r.prompt_tokens is not None]
        report[name] = {
            "probes": len(mine),
            "recall": _share(mine),
            "recall_in_window": _share([r for r in mine if r.in_window]),
            "recall_dropped": _share(dropped),
            "dropped_probes": len(dropped),
            # Of the facts that had left the window, how many the summary kept.
            "summary_retention": (
                round(sum(r.in_summary for r in dropped) / len(dropped), 3)
                if dropped
                else None
            ),
            "prompt_tokens_mean": round(statistics.mean(tokens)) if tokens else None,
            "prompt_tokens_max": max(tokens) if tokens else None,
            "context_overflow": sum(t >= CONTEXT_TOKENS for t in tokens),
            "answer_latency_mean_s": round(
                statistics.mean(r.latency_s for r in mine), 2
            ),
            **stats.get(name, {}),
        }
    return report


def _ollama_answer(base: str, model: str) -> AnswerFn:
    import requests

    session = requests.Session()

    def answer(messages: list[dict]) -> tuple[str, int | None, float]:
        started = time.perf_counter()
        resp = session.post(
            f"{base.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 64,
            },
            timeout=300,
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["choices"][0]["message"]["content"] or ""
        return (
            text,
            body.get("usage", {}).get("prompt_tokens"),
            time.perf_counter() - started,
        )

    return answer


async def _main(args: argparse.Namespace) -> dict:
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    summarizer = OpenAICompatibleLLM(
        LLMConfig(
            model_provider="openai", model_name=args.llm_model, api_base=args.llm_base
        )
    )
    answer_fn = _ollama_answer(args.llm_base, args.llm_model)
    strategies = [Strategy("full", None)]
    for n in args.windows:
        strategies += [Strategy("window", n), Strategy("summary", n)]

    dataset = make_dataset(args.conversations, seed=args.seed)
    rows: list[ProbeResult] = []
    stats: dict[str, dict] = {}
    try:
        for conv in dataset:
            for strategy in strategies:
                results = await run_strategy(
                    conv,
                    strategy,
                    answer_fn,
                    summarizer,
                    stats=stats.setdefault(strategy.name, {}),
                )
                rows += results
                print(
                    f"conv {conv.id} {strategy.name}: "
                    f"{sum(r.correct for r in results)}/{len(results)}",
                    flush=True,
                )
    finally:
        summarizer.close()
    for entry in stats.values():
        entry["summarizer_seconds"] = round(entry["summarizer_seconds"], 1)
    return {
        "config": {
            "model": args.llm_model,
            "conversations": args.conversations,
            "windows": args.windows,
            "seed": args.seed,
        },
        "summary": aggregate(rows, stats),
        "rows": [asdict(r) for r in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--conversations", type=int, default=12)
    parser.add_argument("--windows", type=int, nargs="+", default=[6, 10, 20, 40])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm_base", default="http://localhost:11434/v1")
    parser.add_argument("--llm_model", default="llama3.2:3b")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = asyncio.run(_main(args))
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
