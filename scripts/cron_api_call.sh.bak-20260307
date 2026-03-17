#!/bin/bash
# Helper: call Memory API endpoints from cron without hardcoding secrets.
# Usage: cron_api_call.sh <endpoint> [json-body]
# Example: cron_api_call.sh /v1/decay '{"agent":"all"}'
set -euo pipefail

# Resolve API key file: env var > ~/.agent-memory > legacy ~/.agent-memory-legacy
if [ -n "${AGENT_MEMORY_API_KEY_FILE:-}" ]; then
  API_KEY_FILE="$AGENT_MEMORY_API_KEY_FILE"
elif [ -f "$HOME/.agent-memory/memory-api-key" ]; then
  API_KEY_FILE="$HOME/.agent-memory/memory-api-key"
elif [ -f "$HOME/.agent-memory-legacy/memory-api-key" ]; then
  API_KEY_FILE="$HOME/.agent-memory-legacy/memory-api-key"
else
  API_KEY_FILE="$HOME/.agent-memory/memory-api-key"
fi
if [ ! -f "$API_KEY_FILE" ]; then
  echo "ERROR: API key file not found: $API_KEY_FILE" >&2
  exit 1
fi
API_KEY=$(cat "$API_KEY_FILE")

ENDPOINT="${1:?Usage: cron_api_call.sh <endpoint> [json-body]}"
BODY="${2:-}"
BASE_URL="${AGENT_MEMORY_BASE_URL:-http://localhost:8787}"

if [ -n "$BODY" ]; then
  curl -sf -X POST \
    -H "X-API-Key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$BODY" \
    "${BASE_URL}${ENDPOINT}"
else
  curl -sf -H "X-API-Key: $API_KEY" "${BASE_URL}${ENDPOINT}"
fi
