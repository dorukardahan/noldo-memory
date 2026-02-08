#!/bin/bash
# OpenClaw Memory — cron wrapper for session sync
# Runs incremental sync with logging.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Configurable paths (override via environment if desired)
LOGFILE="${OPENCLAW_MEMORY_SYNC_LOG:-/var/log/openclaw-memory-sync.log}"
SCRIPT="${OPENCLAW_MEMORY_SYNC_SCRIPT:-${ROOT_DIR}/scripts/openclaw_sync.py}"

# Prefer .venv (README), fall back to venv
if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif [ -x "${ROOT_DIR}/venv/bin/python" ]; then
  PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
else
  PYTHON_BIN="python3"
fi

# Optional env file (recommended): set OPENROUTER_API_KEY here
ENV_FILE="${OPENCLAW_MEMORY_ENV_FILE:-${ROOT_DIR}/.env}"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set. Set it in the environment or via ${ENV_FILE}" >> "${LOGFILE}"
  exit 1
fi

mkdir -p "$(dirname "${LOGFILE}")" 2>/dev/null || true

echo "--- $(date -Iseconds) --- SYNC START ---" >> "${LOGFILE}"
"${PYTHON_BIN}" "${SCRIPT}" 2>&1 >> "${LOGFILE}"
echo "--- $(date -Iseconds) --- SYNC END ---" >> "${LOGFILE}"
echo "" >> "${LOGFILE}"

# Keep log under 1MB
if [ -f "${LOGFILE}" ]; then
  SIZE=$(stat -f%z "${LOGFILE}" 2>/dev/null || stat -c%s "${LOGFILE}" 2>/dev/null || echo 0)
  if [ "${SIZE}" -gt 1048576 ]; then
    tail -n 500 "${LOGFILE}" > "${LOGFILE}.tmp"
    mv "${LOGFILE}.tmp" "${LOGFILE}"
  fi
fi
