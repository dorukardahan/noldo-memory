#!/bin/bash
# Asuman Memory System — Management Script
# Usage: ./scripts/manage.sh {start|stop|restart|status|logs|sync|load|health}

set -e

SERVICE="asuman-memory"
VENV="/opt/asuman/whatsapp-memory/venv/bin"
SCRIPTS="/opt/asuman/whatsapp-memory/scripts"
API_URL="http://localhost:8787"

case "${1}" in
    start)
        echo "Starting ${SERVICE}..."
        systemctl start ${SERVICE}
        sleep 1
        systemctl status ${SERVICE} --no-pager -l
        ;;
    stop)
        echo "Stopping ${SERVICE}..."
        systemctl stop ${SERVICE}
        echo "Stopped."
        ;;
    restart)
        echo "Restarting ${SERVICE}..."
        systemctl restart ${SERVICE}
        sleep 1
        systemctl status ${SERVICE} --no-pager -l
        ;;
    status)
        echo "=== Service Status ==="
        systemctl status ${SERVICE} --no-pager -l 2>/dev/null || echo "Service not running"
        echo ""
        echo "=== API Health ==="
        curl -s ${API_URL}/v1/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "API not reachable"
        echo ""
        echo "=== Database ==="
        if [ -f /root/.asuman/memory.sqlite ]; then
            SIZE=$(du -h /root/.asuman/memory.sqlite | cut -f1)
            echo "DB size: ${SIZE}"
            curl -s ${API_URL}/v1/stats 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Stats unavailable (API not running)"
        else
            echo "Database not created yet"
        fi
        ;;
    logs)
        LINES=${2:-50}
        journalctl -u ${SERVICE} -n ${LINES} --no-pager -f
        ;;
    sync)
        echo "Running incremental sync..."
        if [ -f /opt/asuman/.env ]; then
            export $(grep -v '^#' /opt/asuman/.env | grep OPENROUTER_API_KEY | xargs)
        fi
        [ -z "$OPENROUTER_API_KEY" ] && echo "ERROR: OPENROUTER_API_KEY not set" && exit 1
        ${VENV}/python ${SCRIPTS}/openclaw_sync.py "${@:2}"
        ;;
    load)
        echo "Running initial data load..."
        if [ -f /opt/asuman/.env ]; then
            export $(grep -v '^#' /opt/asuman/.env | grep OPENROUTER_API_KEY | xargs)
        fi
        [ -z "$OPENROUTER_API_KEY" ] && echo "ERROR: OPENROUTER_API_KEY not set" && exit 1
        ${VENV}/python ${SCRIPTS}/initial_load.py "${@:2}"
        ;;
    health)
        curl -s ${API_URL}/v1/health | python3 -m json.tool
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|sync|load|health}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the memory API service"
        echo "  stop     - Stop the memory API service"
        echo "  restart  - Restart the memory API service"
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
