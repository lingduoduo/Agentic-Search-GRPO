#!/usr/bin/env bash
# Start the full local dev stack: SerpAPI retrieval server + web backend + Vite frontend
# (from the sibling Agentic-Search-UI checkout).
#
# Usage:
#   bin/run_web_stack.sh
#
# Required env var: SERP_API_KEY (add to .env or export it)
# Optional env vars in .env:
#   SEARCH_AGENT_MODEL   e.g. Qwen/Qwen2.5-1.5B-Instruct  (enables Search Agent mode)
#   SEARCH_AGENT_DEVICE  e.g. mps  (default: mps)
#   TOOL_AGENT_PARSER    json | hermes | llama3 (default: json)
#   SERP_PORT            retrieval server port (default: 8000)
#   WEB_PORT             web backend port (default: 7860)
#   MCP_PORT             MCP server port (default: 8090; only started when MCP_ENABLED=1)
#   FRONTEND_DIR         frontend checkout (default: ../Agentic-Search-UI)
#   MCP_ENABLED          set to 1 to start the MCP server alongside the web stack
#                        Requires: pip install -e ".[mcp]"

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT/.env"
    set +a
fi

if [ -z "${SERP_API_KEY:-}" ]; then
    echo "ERROR: SERP_API_KEY not set (add it to .env or export it)" >&2
    exit 1
fi

SERP_PORT="${SERP_PORT:-8000}"
WEB_PORT="${WEB_PORT:-7860}"
MCP_PORT="${MCP_PORT:-8090}"
MCP_ENABLED="${MCP_ENABLED:-0}"
SEARCH_AGENT_MODEL="${SEARCH_AGENT_MODEL:-}"
SEARCH_AGENT_DEVICE="${SEARCH_AGENT_DEVICE:-mps}"
TOOL_AGENT_PARSER="${TOOL_AGENT_PARSER:-json}"

echo ">> Starting SerpAPI retrieval server on port ${SERP_PORT}..."
PYTHONPATH="$ROOT/src:$ROOT" python3 -m src.internal.servers.web_search.serp --port "$SERP_PORT" &
SERP_PID=$!
MCP_PID=""
trap 'echo ">> Stopping processes..."; kill "$SERP_PID" "$WEB_PID" ${MCP_PID:+"$MCP_PID"} 2>/dev/null; wait 2>/dev/null || true' EXIT

# Wait for retrieval server to be ready (up to 15 s)
for i in $(seq 1 15); do
    if curl -sf "http://localhost:${SERP_PORT}/health" >/dev/null 2>&1; then
        echo ">> Retrieval server ready."
        break
    fi
    [ "$i" -eq 15 ] && { echo "ERROR: retrieval server did not start in 15 s" >&2; exit 1; }
    sleep 1
done

if [ -n "$SEARCH_AGENT_MODEL" ]; then
    echo ">> Search Agent / Tool Agent mode enabled: model=${SEARCH_AGENT_MODEL} device=${SEARCH_AGENT_DEVICE} parser=${TOOL_AGENT_PARSER}"
else
    echo ">> Search Agent / Tool Agent mode disabled (set SEARCH_AGENT_MODEL or SEARCH_AGENT_SERVER_URL in .env to enable)"
fi

echo ">> Starting web backend on port ${WEB_PORT}..."
PYTHONPATH="$ROOT/src:$ROOT" \
SEARCH_AGENT_MODEL="$SEARCH_AGENT_MODEL" \
SEARCH_AGENT_DEVICE="$SEARCH_AGENT_DEVICE" \
TOOL_AGENT_PARSER="$TOOL_AGENT_PARSER" \
AGENTIC_SEARCH_RETRIEVAL_URL="http://localhost:${SERP_PORT}/retrieve" \
    uvicorn src.internal.servers.web.app:app \
        --host 127.0.0.1 --port "$WEB_PORT" &
WEB_PID=$!

if [ "$MCP_ENABLED" = "1" ]; then
    echo ">> Starting MCP server on port ${MCP_PORT}..."
    echo ">> (MCP clients: Claude Desktop, Cursor — see README §MCP Server)"
    PYTHONPATH="$ROOT/src:$ROOT" \
    AGENTIC_SEARCH_WEB_PORT="$WEB_PORT" \
        uvicorn src.internal.mcp_server.api:mcp_app \
            --host 127.0.0.1 --port "$MCP_PORT" &
    MCP_PID=$!
else
    echo ">> MCP server disabled (set MCP_ENABLED=1 in .env to enable; requires pip install -e '[mcp]')"
fi

# The frontend is its own repository (Agentic-Search-UI), cloned next to this one.
FRONTEND_DIR="${FRONTEND_DIR:-$ROOT/../Agentic-Search-UI}"
if [ ! -f "$FRONTEND_DIR/package.json" ]; then
    echo ">> Frontend not found at $FRONTEND_DIR (set FRONTEND_DIR, or clone Agentic-Search-UI there)."
    echo ">> Backend is running on http://127.0.0.1:${WEB_PORT}. Ctrl-C to stop."
    wait
fi
echo ">> Starting frontend dev server..."
echo ">> Open http://127.0.0.1:5173 when ready."
cd "$FRONTEND_DIR" && BACKEND_URL="http://127.0.0.1:${WEB_PORT}" npm run dev
