"""Chat-related configuration constants."""

from __future__ import annotations

import os

# Token sent by the LLM to signal the end of a stream.
STOP_STREAM_PAT: str = os.environ.get("STOP_STREAM_PAT", "")

# Maximum number of LLM tool-calling cycles per chat turn.
# Default 6: covers search → open_url × 2 + fallback answer cycle.
# Override via the MAX_LLM_CYCLES env var for tool-heavy MCP workflows.
MAX_LLM_CYCLES: int = int(os.environ.get("MAX_LLM_CYCLES", "6"))

NUM_RETURNED_HITS: int = int(os.environ.get("NUM_RETURNED_HITS", "50"))
MAX_CHUNKS_FED_TO_CHAT: int = int(os.environ.get("MAX_CHUNKS_FED_TO_CHAT", "25"))

# 1 / (1 + DOC_TIME_DECAY * doc-age-in-years), set to 0 for no decay
DOC_TIME_DECAY: float = float(os.environ.get("DOC_TIME_DECAY", "0.5"))
BASE_RECENCY_DECAY: float = 0.5
FAVOR_RECENT_DECAY_MULTIPLIER: float = 2.0

CONTEXT_CHUNKS_ABOVE: int = int(os.environ.get("CONTEXT_CHUNKS_ABOVE", "1"))
CONTEXT_CHUNKS_BELOW: int = int(os.environ.get("CONTEXT_CHUNKS_BELOW", "1"))

# Weighting between vector and keyword search; 1 = pure vector, 0 = pure keyword.
HYBRID_ALPHA: float = max(0, min(1, float(os.environ.get("HYBRID_ALPHA", "0.5"))))

# Weighting between title and content fields during search (0–1).
TITLE_CONTENT_RATIO: float = max(
    0, min(1, float(os.environ.get("TITLE_CONTENT_RATIO", "0.10")))
)

HARD_DELETE_CHATS: bool = os.environ.get("HARD_DELETE_CHATS", "").lower() == "true"

NUM_INTERNET_SEARCH_RESULTS: int = int(
    os.environ.get("NUM_INTERNET_SEARCH_RESULTS", "10")
)
NUM_INTERNET_SEARCH_CHUNKS: int = int(
    os.environ.get("NUM_INTERNET_SEARCH_CHUNKS", "50")
)

USE_SEMANTIC_KEYWORD_EXPANSIONS_BASIC_SEARCH: bool = (
    os.environ.get("USE_SEMANTIC_KEYWORD_EXPANSIONS_BASIC_SEARCH", "false").lower()
    == "true"
)


PROMPTS_YAML: str = os.environ.get("PROMPTS_YAML", "./data/seeding/prompts.yaml")
PERSONAS_YAML: str = os.environ.get("PERSONAS_YAML", "./data/seeding/personas.yaml")
