"""Search-context, prompt, chat, and answer-generation helpers."""

from .enums import AgentBehavior
from .enums import AnswerStyle
from .enums import QueryType
from .enums import SearchType
from .models import AgentBehaviorConfig
from .models import AnswerClaim
from .models import AnswerDraft
from .models import AnswerGenerationRequest
from .models import AnswerGenerationResult
from .models import ChatMessage
from .models import ContextDocument
from .models import ContextSection
from .models import ClaimVerdict
from .models import EvidenceSource
from .models import EvidenceSnippet
from .models import GroundedGenerationConfig
from .models import LLMClient
from .models import LLMResponse
from .models import LLMTimeoutError
from .models import ModelUnavailableError
from .models import PromptBundle
from .models import SearchContextBundle
from .models import SearchFilters
from .models import SearchRequest
from .models import VerificationResult
from .models import VerificationStatus
from .pipeline import answer_with_retrieval
from .pipeline import generate_answer
from .pipeline import rank_evidence_snippets
from .pipeline import retrieve_context
from .pipeline import synthesize_answer_from_context
from .prompts import build_agent_behavior_prompt
from .prompts import build_answer_prompt
from .prompts import build_chat_prompt
from .prompts import build_corrective_answer_prompt
from .prompts import build_retrieval_prompt
from .prompts import build_structured_answer_prompt
from .safety import CANONICAL_ABSTENTION
from .safety import TIMEOUT_DEGRADED_ANSWER
from .safety import evidence_from_context
from .safety import parse_answer_draft
from .safety import render_verified_answer
from .safety import verify_answer_draft
from .structured_output import SchemaUnsupportedError
from .structured_output import StructuredCompletionMetadata
from .structured_output import StructuredOutputCapability
from .structured_output import StructuredOutputRequest
from .structured_output import answer_draft_json_schema
from .tool_evidence import ToolDescriptor
from .tool_evidence import ToolRegistry
from .tool_evidence import ToolRequest
from .tool_evidence import ToolSafety
from .tool_evidence import ToolSelector
from .tool_evidence import collect_tool_evidence
from .utils import build_context_bundle
from .utils import documents_from_search_results
from .utils import extract_citations
from .utils import merge_adjacent_documents

__all__ = [
    "AgentBehavior",
    "AgentBehaviorConfig",
    "AnswerClaim",
    "AnswerDraft",
    "AnswerGenerationRequest",
    "AnswerGenerationResult",
    "AnswerStyle",
    "ChatMessage",
    "ContextDocument",
    "ContextSection",
    "ClaimVerdict",
    "EvidenceSource",
    "EvidenceSnippet",
    "GroundedGenerationConfig",
    "LLMClient",
    "LLMResponse",
    "LLMTimeoutError",
    "ModelUnavailableError",
    "PromptBundle",
    "QueryType",
    "SearchContextBundle",
    "SearchFilters",
    "SearchRequest",
    "SearchType",
    "SchemaUnsupportedError",
    "StructuredCompletionMetadata",
    "StructuredOutputCapability",
    "StructuredOutputRequest",
    "ToolDescriptor",
    "ToolRegistry",
    "ToolRequest",
    "ToolSafety",
    "ToolSelector",
    "VerificationResult",
    "VerificationStatus",
    "CANONICAL_ABSTENTION",
    "TIMEOUT_DEGRADED_ANSWER",
    "answer_with_retrieval",
    "answer_draft_json_schema",
    "build_agent_behavior_prompt",
    "build_answer_prompt",
    "build_chat_prompt",
    "build_corrective_answer_prompt",
    "build_context_bundle",
    "build_retrieval_prompt",
    "build_structured_answer_prompt",
    "collect_tool_evidence",
    "documents_from_search_results",
    "extract_citations",
    "evidence_from_context",
    "generate_answer",
    "merge_adjacent_documents",
    "parse_answer_draft",
    "rank_evidence_snippets",
    "retrieve_context",
    "render_verified_answer",
    "synthesize_answer_from_context",
    "verify_answer_draft",
]
