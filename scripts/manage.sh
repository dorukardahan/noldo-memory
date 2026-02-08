#!/bin/bash
# OpenClaw Memory — Management Script
# Usage: ./scripts/manage.sh {start|stop|restart|status|logs|sync|load|health}

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVICE="${OPENCLAW_MEMORY_SERVICE:-openclaw-memory}"
API_URL="${OPENCLAW_MEMORY_API_URL:-http://127.0.0.1:8787}"
SCRIPTS_DIR="${ROOT_DIR}/scripts"

# Prefer .venv (README), fall back to venv
if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif [ -x "${ROOT_DIR}/venv/bin/python" ]; then
  PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

# Optional env file (recommended)
ENV_FILE="${OPENCLAW_MEMORY_ENV_FILE:-${ROOT_DIR}/.env}"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

DB_PATH="${ASUMAN_MEMORY_DB:-${HOME}/.asuman/memory.sqlite}"

case "${1:-}" in
  start)
    echo "Starting ${SERVICE}..."
    systemctl start "${SERVICE}"
    sleep 1
    systemctl status "${SERVICE}" --no-pager -l
    ;;
  stop)
    echo "Stopping ${SERVICE}..."
    systemctl stop "${SERVICE}"
    echo "Stopped."
    ;;
  restart)
    echo "Restarting ${SERVICE}..."
    systemctl restart "${SERVICE}"
    sleep 1
    systemctl status "${SERVICE}" --no-pager -l
    ;;
  status)
    echo "=== Service Status ==="
    systemctl status "${SERVICE}" --no-pager -l 2>/dev/null || echo "Service not running"
    echo ""

    echo "=== API Health ==="
    curl -s "${API_URL}/v1/health" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "API not reachable"
    echo ""

    echo "=== Database ==="
    if [ -f "${DB_PATH}" ]; then
      SIZE=$(du -h "${DB_PATH}" | cut -f1)
      echo "DB path: ${DB_PATH}"
      echo "DB size: ${SIZE}"
      curl -s "${API_URL}/v1/stats" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Stats unavailable (API not running)"
    else
      echo "Database not created yet (expected at ${DB_PATH})"
    fi
    ;;
  logs)
    LINES="${2:-50}"
    journalctl -u "${SERVICE}" -n "${LINES}" --no-pager -f
    ;;
  sync)
    echo "Running incremental sync..."
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
      echo "ERROR: OPENROUTER_API_KEY not set (set it in env or ${ENV_FILE})" >&2
      exit 1
    fi
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/openclaw_sync.py" "${@:2}"
    ;;
  load)
    echo "Running initial data load..."
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
      echo "ERROR: OPENROUTER_API_KEY not set (set it in env or ${ENV_FILE})" >&2
      exit 1
    fi
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/initial_load.py" "${@:2}"
    ;;
  health)
    curl -s "${API_URL}/v1/health" | python3 -m json.tool
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|sync|load|health}"
    echo ""
    echo "Commands:"
    echo "  start    - Start the memory API service (systemd)"
    echo "  stop     - Stop the memory API service (systemd)"
    echo "  restart  - Restart the memory API service (systemd)"
    echo "  status   - Show service status, API health, and DB stats"
    echo "  logs [N] - Follow journal logs (default: last 50 lines)"
    echo "  sync     - Run incremental session sync"
    echo "  load     - Run initial bulk data load"
    echo "  health   - Quick API health check"
    echo ""
    echo "Sync options (pass after 'sync'):"
    echo "  --full             Re-scan all sessions"
    echo "  --skip-embeddings  Store without vector embeddings"
    echo "  --status           Show sync state"
    echo ""
    echo "Load options (pass after 'load'):"
    echo "  --dry-run          Parse only, don't store"
    echo "  --skip-embeddings  Store without embeddings"
    echo "  --limit N          Process only N session files"
    exit 1
    ;;
esac
