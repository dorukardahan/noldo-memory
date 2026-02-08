#!/bin/bash
# OpenClaw Memory — SQLite backup helper
# Creates a safe online backup (WAL-safe) and prunes older backups.

set -euo pipefail

DB_PATH="${ASUMAN_MEMORY_DB:-${HOME}/.asuman/memory.sqlite}"
BACKUP_DIR="${OPENCLAW_MEMORY_BACKUP_DIR:-${HOME}/.asuman/backups}"
DATE=$(date +%Y-%m-%d)
BACKUP_FILE="${BACKUP_DIR}/memory-${DATE}.sqlite"
KEEP_DAYS="${OPENCLAW_MEMORY_BACKUP_KEEP_DAYS:-7}"

mkdir -p "${BACKUP_DIR}"

if [ ! -f "${DB_PATH}" ]; then
  echo "$(date -Iseconds) ERROR: Database not found at ${DB_PATH}" >&2
  exit 1
fi

# Prefer sqlite3 .backup for a consistent copy while the DB is in use
if command -v sqlite3 &>/dev/null; then
  sqlite3 "${DB_PATH}" ".backup '${BACKUP_FILE}'"
else
  cp "${DB_PATH}" "${BACKUP_FILE}"
fi

echo "$(date -Iseconds) Backup created: ${BACKUP_FILE} ($(du -h "${BACKUP_FILE}" | cut -f1))"

# Prune old backups
find "${BACKUP_DIR}" -name "memory-*.sqlite" -type f -mtime "+${KEEP_DAYS}" -delete 2>/dev/null || true

COUNT=$(find "${BACKUP_DIR}" -name "memory-*.sqlite" -type f | wc -l | tr -d ' ')
echo "$(date -Iseconds) Backups retained: ${COUNT}"
