"""Runner for PlainGenerationLoop — pure LLM generation, no retrieval or tools.

Neutral module (like tool_agent_runner.py) so query_and_chat routers can reuse
it without importing app.py.
"""

from __future__ import annotations

from src.agents.generation.plain import (
    PlainGenerationLoop,
    PlainGenerationLoopConfig,
)


async def _run_plain_chat(
    message: str,
    *,
    manager,
    tokenizer,
    history: list,
    on_turn=None,
    on_token=None,
) -> str:
    """Run one PlainGenerationLoop turn over history + the new user message."""
    loop = PlainGenerationLoop(
        tokenizer=tokenizer,
        server_manager=manager,
        config=PlainGenerationLoopConfig(),
    )
    messages = [{"role": m.role, "content": m.content} for m in history] + [
        {"role": "user", "content": message}
    ]
    output = await loop.run(
        messages,
        sampling_params={"temperature": 0.7, "max_tokens": 256},
        on_turn=on_turn,
        on_token=on_token,
    )
    return output.final_answer or ""
