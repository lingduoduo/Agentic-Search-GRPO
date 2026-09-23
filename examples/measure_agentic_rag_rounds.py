"""Census of how many rounds AgenticRAGLoop uses, and why it stops.

The CHAT path pays at least three serial LLM calls per request (enhance,
sufficiency judge, synthesis), with the judge between retrieval and the first
answer token. For a read-only, retryable task a single retrieve-then-answer pass
would do -- if most questions stop after one round. This measures whether they do.

`rounds_used` alone cannot answer that. A run that stops at round 1 because the
judge failed open (timeout or error, treated as "sufficient") is not evidence
that one round was enough, and a run that uses all three rounds may have
retrieved nothing new after the first. So each run is classified from the
control-flow trace the loop already records, not from its round count.

Run, from stored CHAT answers (no LLM needed):

    python -m examples.measure_agentic_rag_rounds --history data/agentic_search.db

Or live, replaying questions through the loop (needs a retrieval server and an
OpenAI-compatible LLM; configured as the web backend configures it):

    python -m examples.measure_agentic_rag_rounds \\
        --questions data/intent_eval_queries.json --label chat \\
        --llm_base http://localhost:11434/v1 --llm_model llama3.2:3b

The judge and gap analysis run under a 5 s timeout, as in serving. A model that
does not fit on the GPU (llama3.1:8b on an 8 GB Mac) times out on nearly every
call, and every run then reads `judge_failed_open` -- measure with a model that
answers in well under that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# The web backend runs AgenticRAGLoop with max_rounds=3 (`_run_agentic_rag`).
MAX_ROUNDS = 3


@dataclass(frozen=True)
class RunSummary:
    rounds_used: int
    stop_reason: str
    docs_after_round_1: int
    docs_final: int
    post_round_1_ms: int | None
    question: str = ""

    @property
    def docs_added_after_round_1(self) -> int:
        return self.docs_final - self.docs_after_round_1


def _ts_ms(event: dict) -> float:
    ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    return ts.timestamp() * 1000


def _round(event: dict) -> int:
    return int(event["details"].get("search_round", 0))


def summarize_trace(
    events: list[dict],
    *,
    rounds_used: int,
    max_rounds: int = MAX_ROUNDS,
    question: str = "",
) -> RunSummary:
    """Classify one run from its control-flow trace."""
    searches = [e for e in events if e["action"] == "vector_db_search"]
    judges = [e for e in events if e["action"] == "sufficiency_check"]
    synth = next((e for e in events if e["action"] == "synthesize"), None)

    if not searches:
        return RunSummary(rounds_used, "no_queries", 0, 0, None, question)

    last_round = max(_round(e) for e in searches)
    last_judge = next((j for j in judges if _round(j) == last_round), None)
    if last_judge is None:
        # The loop skips the judge only on its final permitted round.
        stop_reason = "max_rounds" if last_round >= max_rounds else "no_queries"
    elif last_judge["status"] == "failed" or last_judge["details"].get("fallback"):
        # Traces predating the `fallback` detail mark this only by status.
        stop_reason = "judge_failed_open"
    elif last_judge["details"].get("sufficient"):
        stop_reason = "sufficient"
    else:
        stop_reason = "no_new_followups"

    round_1 = next(e for e in searches if _round(e) == 1)
    docs_after_round_1 = int(round_1["details"].get("document_count", 0))
    docs_final = int(searches[-1]["details"].get("document_count", 0))

    post_round_1_ms = None
    if synth is not None:
        synth_start = _ts_ms(synth) - (synth.get("duration_ms") or 0)
        post_round_1_ms = round(synth_start - _ts_ms(round_1))

    return RunSummary(
        rounds_used,
        stop_reason,
        docs_after_round_1,
        docs_final,
        post_round_1_ms,
        question,
    )


def _pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def aggregate(runs: list[RunSummary]) -> dict:
    stop_reasons = Counter(r.stop_reason for r in runs)
    with_verdict = [r for r in runs if r.stop_reason != "judge_failed_open"]
    one_round = [
        r for r in with_verdict if r.rounds_used == 1 and r.stop_reason == "sufficient"
    ]
    multi = [r for r in runs if r.rounds_used > 1]
    timings = [r.post_round_1_ms for r in runs if r.post_round_1_ms is not None]
    return {
        "runs": len(runs),
        "rounds_used": {
            str(k): v for k, v in sorted(Counter(r.rounds_used for r in runs).items())
        },
        "stop_reasons": dict(stop_reasons),
        "one_round_share": (
            round(len(one_round) / len(with_verdict), 3) if with_verdict else None
        ),
        "multi_round_runs": len(multi),
        "multi_round_zero_new_docs": sum(
            1 for r in multi if r.docs_added_after_round_1 == 0
        ),
        "post_round_1_ms": {
            "median": round(statistics.median(timings)) if timings else None,
            "p90": _pct(timings, 0.9),
        },
    }


def load_history(db_path: Path) -> list[RunSummary]:
    """Summaries of stored CHAT answers that carry a control-flow trace."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT a.metadata_json,
               (SELECT u.content FROM chat_messages u
                 WHERE u.session_id = a.session_id AND u.role = 'user'
                   AND u.created_at <= a.created_at
                 ORDER BY u.created_at DESC LIMIT 1)
          FROM chat_messages a
         WHERE a.role = 'assistant'
           AND json_extract(a.metadata_json, '$.intent') = 'chat'
         ORDER BY a.created_at
        """
    ).fetchall()
    conn.close()
    runs = []
    for metadata_json, question in rows:
        meta = json.loads(metadata_json)
        trace = meta.get("control_flow_trace")
        if trace is None or meta.get("rounds_used") is None:
            continue
        runs.append(
            summarize_trace(
                trace, rounds_used=int(meta["rounds_used"]), question=question or ""
            )
        )
    return runs


def load_questions(path: Path, label: str | None) -> list[str]:
    items = json.loads(path.read_text())
    return [
        item["text"] if isinstance(item, dict) else item
        for item in items
        if label is None or (isinstance(item, dict) and item.get("label") == label)
    ]


async def replay(questions: list[str], args: argparse.Namespace) -> list[RunSummary]:
    from src.agents.core.control_flow_trace import ControlFlowRecorder
    from src.agents.search import AgenticRAGConfig, AgenticRAGLoop
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(
        LLMConfig(
            model_provider="openai",
            model_name=args.llm_model,
            api_key=args.llm_api_key,
            api_base=args.llm_base,
        )
    )
    loop = AgenticRAGLoop(
        AgenticRAGConfig(
            max_rounds=MAX_ROUNDS, topk=args.topk, retrieval_url=args.search_url
        ),
        llm=llm,
    )
    runs = []
    try:
        for i, question in enumerate(questions, 1):
            recorder = ControlFlowRecorder(f"census-{i}")
            result = await loop.run(question, recorder=recorder)
            run = summarize_trace(
                [e.to_dict() for e in recorder.snapshot()],
                rounds_used=result.rounds_used,
                question=question,
            )
            runs.append(run)
            print(
                f"[{i}/{len(questions)}] {run.rounds_used} {run.stop_reason}: {question}",
                flush=True,
            )
    finally:
        llm.close()
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--history", type=Path, help="web store SQLite DB")
    source.add_argument("--questions", type=Path, help="JSON list of questions")
    parser.add_argument("--label", help="keep only questions with this label")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--search_url", default="http://localhost:8001/retrieve")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--llm_base", default="http://localhost:11434/v1")
    parser.add_argument("--llm_model", default="llama3.2:3b")
    parser.add_argument("--llm_api_key")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    if args.history:
        runs = load_history(args.history)
    else:
        questions = load_questions(args.questions, args.label)[: args.limit]
        runs = asyncio.run(replay(questions, args))

    report = {
        "summary": aggregate(runs),
        "runs": [
            asdict(r) | {"docs_added_after_round_1": r.docs_added_after_round_1}
            for r in runs
        ],
    }
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
