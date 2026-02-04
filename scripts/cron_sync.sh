#!/bin/bash
# Cron wrapper for Asuman Memory sync
# Runs incremental sync with logging

set -e

LOGFILE="/var/log/asuman-memory-sync.log"
VENV="/opt/asuman/whatsapp-memory/venv/bin"
SCRIPT="/opt/asuman/whatsapp-memory/scripts/openclaw_sync.py"

# Load API key from .env (NEVER hardcode keys in scripts)
if [ -f /opt/asuman/.env ]; then
    export $(grep -v '^#' /opt/asuman/.env | grep OPENROUTER_API_KEY | xargs)
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set. Check /opt/asuman/.env" >> "$LOGFILE"
    exit 1
fi

echo "--- $(date -Iseconds) --- SYNC START ---" >> "$LOGFILE"
${VENV}/python ${SCRIPT} 2>&1 >> "$LOGFILE"
echo "--- $(date -Iseconds) --- SYNC END ---" >> "$LOGFILE"
echo "" >> "$LOGFILE"

# Keep log under 1MB
if [ -f "$LOGFILE" ] && [ $(stat -f%z "$LOGFILE" 2>/dev/null || stat -c%s "$LOGFILE" 2>/dev/null) -gt 1048576 ]; then
    tail -n 500 "$LOGFILE" > "${LOGFILE}.tmp"
    mv "${LOGFILE}.tmp" "$LOGFILE"
fi
