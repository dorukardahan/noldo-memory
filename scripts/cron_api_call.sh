#!/bin/bash
# Helper: call Memory API endpoints from cron without hardcoding secrets.
# Usage: cron_api_call.sh <endpoint> [json-body]
# Example: cron_api_call.sh /v1/decay '{"agent":"all"}'
set -euo pipefail

API_KEY_FILE="${AGENT_MEMORY_API_KEY_FILE:-$HOME/.asuman/memory-api-key}"
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
