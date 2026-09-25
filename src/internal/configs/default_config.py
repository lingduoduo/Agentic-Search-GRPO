#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Default configuration for Agentic Search.

Values here are the hard-coded defaults used when the corresponding
environment variable is absent. They mirror the defaults in app_configs.py
and chat_configs.py and are kept in one place for documentation and
test-fixture use.
"""

from src.shared_configs.intent import (
    DEFAULT_MIN_MODULE_SCORE,
    DEFAULT_MIN_ROUTE_MARGIN,
    DEFAULT_TOP_K,
)

DEFAULT_CONFIG: dict = {
    # -------------------------------------------------------------------------
    # Services — retrieval server, web backend, persistence
    # -------------------------------------------------------------------------
    "AGENTIC_SEARCH_RETRIEVAL_URL": "http://localhost:8001/retrieve",
    "AGENTIC_SEARCH_FETCH_URL": "",
    "AGENTIC_SEARCH_RETRIEVAL_HOST": "0.0.0.0",
    "AGENTIC_SEARCH_RETRIEVAL_PORT": 8001,
    "AGENTIC_SEARCH_WEB_HOST": "0.0.0.0",
    "AGENTIC_SEARCH_WEB_PORT": 8080,
    "AGENTIC_SEARCH_WEB_DB_PATH": ":memory:",
    "AGENTIC_SEARCH_WEB_TOP_K": 5,
    # -------------------------------------------------------------------------
    # Intent model serving
    # -------------------------------------------------------------------------
    "AGENTIC_SEARCH_INTENT_INDEX_PATH": "",
    "AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN": DEFAULT_MIN_ROUTE_MARGIN,
    "AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE": DEFAULT_MIN_MODULE_SCORE,
    # Neighbors averaged per route; see intent_top_k in app_configs.py.
    "AGENTIC_SEARCH_INTENT_TOP_K": DEFAULT_TOP_K,
    "AGENTIC_SEARCH_ROUTE_CLARIFICATION": True,
    "WEB_DOMAIN": "http://localhost:8080",
    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------
    "AGENTIC_SEARCH_AUTH_SECRET": "agentic-search-dev-secret",
    "AGENTIC_SEARCH_JWT_PUBLIC_KEY_URL": "",
    "AGENTIC_SEARCH_ENVIRONMENT": "development",
    "AGENTIC_SEARCH_WORKLOAD_ISSUER": "",
    "AGENTIC_SEARCH_WORKLOAD_AUDIENCE": "",
    "AGENTIC_SEARCH_WORKLOAD_SUBJECTS": "{}",
    "AGENTIC_SEARCH_SUPER_USERS": "[]",
    "AGENTIC_SEARCH_SUPER_API_KEY": "",
    "AUTH_COOKIE_NAME": "fastapiusersauth",
    # -------------------------------------------------------------------------
    # LLM provider
    # -------------------------------------------------------------------------
    "GEN_AI_MODEL_PROVIDER": "openai",
    "GEN_AI_MODEL_VERSION": "gpt-4o-mini",
    "GEN_AI_API_KEY": "",
    "GEN_AI_API_BASE": "",
    "GEN_AI_MAX_INPUT_TOKENS": 8192,
    # -------------------------------------------------------------------------
    # Vector database
    # -------------------------------------------------------------------------
    "DISABLE_VECTOR_DB": "False",
    "MAX_CHUNKS_PER_DOC_BATCH": 512,
    "ENABLE_MULTIPASS_INDEXING": "False",
    # -------------------------------------------------------------------------
    # Chat / agent loop
    # -------------------------------------------------------------------------
    "MAX_LLM_CYCLES": 6,
    "NUM_RETURNED_HITS": 50,
    "MAX_CHUNKS_FED_TO_CHAT": 25,
    "DOC_TIME_DECAY": 0.5,
    "CONTEXT_CHUNKS_ABOVE": 1,
    "CONTEXT_CHUNKS_BELOW": 1,
    # LLM_SOCKET_READ_TIMEOUT: see src/internal/configs/timeouts.toml (llm.socket_read_timeout_seconds, 120)
    "HYBRID_ALPHA": 0.5,
    "TITLE_CONTENT_RATIO": 0.10,
    "HARD_DELETE_CHATS": "False",
    "STOP_STREAM_PAT": "",
    "NUM_INTERNET_SEARCH_RESULTS": 10,
    "NUM_INTERNET_SEARCH_CHUNKS": 50,
    "USE_SEMANTIC_KEYWORD_EXPANSIONS_BASIC_SEARCH": "False",
    # -------------------------------------------------------------------------
    # Permission sync cadence (seconds)
    # -------------------------------------------------------------------------
    "AGENTIC_SEARCH_DEFAULT_PERMISSION_DOC_SYNC_FREQUENCY": 300,
    "AGENTIC_SEARCH_NUM_PERMISSION_WORKERS": 2,
    "CONFLUENCE_PERMISSION_DOC_SYNC_FREQUENCY": 1800,
    "CONFLUENCE_PERMISSION_GROUP_SYNC_FREQUENCY": 1800,
    "JIRA_PERMISSION_DOC_SYNC_FREQUENCY": 1800,
    "JIRA_PERMISSION_GROUP_SYNC_FREQUENCY": 1800,
    "GITHUB_PERMISSION_DOC_SYNC_FREQUENCY": 300,
    "GITHUB_PERMISSION_GROUP_SYNC_FREQUENCY": 300,
    "SLACK_PERMISSION_DOC_SYNC_FREQUENCY": 300,
    "TEAMS_PERMISSION_DOC_SYNC_FREQUENCY": 300,
    "SHAREPOINT_PERMISSION_DOC_SYNC_FREQUENCY": 1800,
    "SHAREPOINT_PERMISSION_GROUP_SYNC_FREQUENCY": 300,
    "GOOGLE_DRIVE_PERMISSION_GROUP_SYNC_FREQUENCY": 300,
    # -------------------------------------------------------------------------
    # Telemetry (PostHog)
    # -------------------------------------------------------------------------
    "POSTHOG_API_KEY": "",
    "POSTHOG_HOST": "https://us.i.posthog.com",
    "POSTHOG_DEBUG_LOGS_ENABLED": "False",
    # -------------------------------------------------------------------------
    # Multi-tenancy and licensing
    # -------------------------------------------------------------------------
    "MULTI_TENANT": "False",
    "AGENTIC_SEARCH_LICENSE_ENFORCEMENT_ENABLED": "False",
    "AGENTIC_SEARCH_CLOUD_DATA_PLANE_URL": "",
    # -------------------------------------------------------------------------
    # Billing (Stripe — optional)
    # -------------------------------------------------------------------------
    "STRIPE_PUBLISHABLE_KEY_OVERRIDE": "",
    "STRIPE_PUBLISHABLE_KEY_URL": "",
    # -------------------------------------------------------------------------
    # Developer flags
    # -------------------------------------------------------------------------
    "DEV_MODE": "False",
}
