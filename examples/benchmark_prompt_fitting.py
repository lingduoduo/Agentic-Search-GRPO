"""Measure what the prompt context budget costs per build.

Drives the real ``AgentLoopBase`` fitting path against a real tokenizer, so the
only stubbed part is the chat template (a ChatML-shaped render) and the message
buffer's shape. Two things are worth measuring separately:

    fit     AgentLoopBase._fit_messages_to_budget -- drops whole oldest messages
            so the prompt stays well-formed. Reports the number of full renders,
            which must not scale with the number of messages dropped.

    slice   the raw ``_crop_prompt_ids`` tail slice this replaced. Faster, and
            produces a prompt whose surviving tail begins mid-message.

    growth  the per-turn cost of re-rendering the whole buffer each turn
            (O(n^2) across a run) against the lower bound of encoding only the
            messages added this turn.

The buffer mirrors SearchAgentLoop's real shape: a ~2KB system instruction, then
per turn one short assistant action and one large ``<information>`` observation.

Examples:
    python -m examples.benchmark_prompt_fitting --target fit
    python -m examples.benchmark_prompt_fitting --target growth --turns 3,5,10
"""

from __future__ import annotations

import argparse
import os
import time

DEFAULT_MODEL = "intfloat/e5-small-v2"


def _tokenizer(model: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(model)

    class ChatMLTokenizer:
        chat_template = "chatml"

        def encode(self, text):
            return hf.encode(text, add_special_tokens=False)

        def decode(self, ids, skip_special_tokens=True):
            return hf.decode(ids)

        def apply_chat_template(
            self, messages, tools=None, add_generation_prompt=True, tokenize=False
        ):
            out = "".join(
                f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
                for m in messages
            )
            if add_generation_prompt:
                out += "<|im_start|>assistant\n"
            return self.encode(out) if tokenize else out

    return ChatMLTokenizer()


def _buffer(turns: int) -> list[dict]:
    """SearchAgentLoop's real shape: big system prompt, action + observation turns."""
    messages = [{"role": "system", "content": "S" * 2000}]
    for _ in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": "<think>reasoning</think><search>q</search>" + "a" * 200,
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "<information>\n" + ("doc text " * 200) + "\n</information>",
            }
        )
    return messages


def _loop(tokenizer, budget: int):
    from src.agents.core.base import AgentLoopBase, AgentLoopConfig

    return AgentLoopBase(
        tokenizer=tokenizer,
        server_manager=None,
        config=AgentLoopConfig(prompt_length=budget),
    )


def run_fit(tokenizer, turn_counts: list[int], budget: int) -> None:
    from src.agents.core.base import _crop_prompt_ids

    header = f"{'turns':>6} {'budget':>7} {'fit(ms)':>9} {'slice(ms)':>10} {'renders':>8} {'dropped':>8}"
    print(header)
    for turns in turn_counts:
        messages = _buffer(turns)

        loop = _loop(tokenizer, budget)
        renders = {"n": 0}
        original = loop._render_prompt_ids

        def counted(msgs, _o=original, _r=renders):
            _r["n"] += 1
            return _o(msgs)

        loop._render_prompt_ids = counted
        start = time.perf_counter()
        loop._build_prompt_ids_sync(messages)
        fit_ms = (time.perf_counter() - start) * 1000

        bare = _loop(tokenizer, budget)
        start = time.perf_counter()
        ids = bare._render_prompt_ids(messages)
        _crop_prompt_ids(ids, bare._encode_system_prefix(messages), budget)
        slice_ms = (time.perf_counter() - start) * 1000

        print(
            f"{turns:>6} {budget:>7} {fit_ms:>9.1f} {slice_ms:>10.1f} "
            f"{renders['n']:>8} {loop.prompt_messages_dropped:>8}"
        )


def run_growth(tokenizer, turn_counts: list[int]) -> None:
    print(
        f"{'turns':>6} {'whole(ms)':>11} {'added-only(ms)':>15} {'saved(ms)':>10} {'ratio':>7}"
    )
    for turns in turn_counts:
        whole = 0.0
        for k in range(1, turns + 1):
            messages = _buffer(k)
            start = time.perf_counter()
            tokenizer.encode(tokenizer.apply_chat_template(messages))
            whole += time.perf_counter() - start

        added = 0.0
        seen = 0
        for k in range(1, turns + 1):
            messages = _buffer(k)
            start = time.perf_counter()
            for message in messages[seen:]:
                tokenizer.encode(
                    f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                )
            added += time.perf_counter() - start
            seen = len(messages)

        whole_ms, added_ms = whole * 1000, added * 1000
        ratio = whole_ms / added_ms if added_ms else float("inf")
        print(
            f"{turns:>6} {whole_ms:>11.1f} {added_ms:>15.1f} "
            f"{whole_ms - added_ms:>10.1f} {ratio:>7.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("fit", "growth"), default="fit")
    parser.add_argument("--turns", default="5,10,20,40")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    turn_counts = [int(t) for t in args.turns.split(",") if t.strip()]
    tokenizer = _tokenizer(args.model)
    if args.target == "fit":
        run_fit(tokenizer, turn_counts, args.budget)
    else:
        run_growth(tokenizer, turn_counts)


if __name__ == "__main__":
    main()
