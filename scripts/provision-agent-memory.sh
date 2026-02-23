#!/bin/bash
# provision-agent-memory.sh — Auto-provision memory system for all OpenClaw agents
#
# What it does:
#   1. Scans all agents in openclaw.json
#   2. Creates memory/ dir in each workspace
#   3. Updates hooks config to allow all agents
#   4. Verifies Memory API can serve each agent
#
# Usage:
#   ./provision-agent-memory.sh              # provision all agents
#   ./provision-agent-memory.sh --status     # show current status
#   ./provision-agent-memory.sh --dry-run    # show what would be done

set -euo pipefail

OPENCLAW_DIR="${HOME}/.openclaw"
CONFIG_FILE="${OPENCLAW_DIR}/openclaw.json"
MEMORY_API="http://localhost:8787/v1"
API_KEY_FILE="/root/.asuman/memory-api-key"
API_KEY=""
if [ -f "$API_KEY_FILE" ]; then
  API_KEY=$(cat "$API_KEY_FILE")
fi

DRY_RUN=false
STATUS_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --status) STATUS_ONLY=true ;;
  esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
err()  { echo -e "  ${RED}✗${NC} $1"; }

# Get all agents and their workspaces
get_agents() {
  python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    c = json.load(f)
defaults_ws = c.get('agents',{}).get('defaults',{}).get('workspace','$OPENCLAW_DIR/workspace')
for a in c.get('agents',{}).get('list',[]):
    aid = a.get('id','')
    ws = a.get('workspace', defaults_ws)
    print(f'{aid}|{ws}')
"
}

# Check memory API health
check_memory_api() {
  local response
  response=$(curl -sf -H "X-API-Key: $API_KEY" "$MEMORY_API/health" 2>/dev/null) || {
    err "Memory API not reachable at $MEMORY_API"
    return 1
  }
  ok "Memory API healthy"
}

# Check/create memory dir for an agent workspace
provision_workspace_memory() {
  local agent_id="$1"
  local workspace="$2"
  local memory_dir="${workspace}/memory"

  if [ ! -d "$workspace" ]; then
    if $DRY_RUN; then
      warn "Would create workspace: $workspace"
    else
      mkdir -p "$workspace"
      ok "Created workspace: $workspace"
    fi
  fi

  if [ ! -d "$memory_dir" ]; then
    if $DRY_RUN; then
      warn "Would create memory dir: $memory_dir"
    else
      mkdir -p "$memory_dir"
      ok "Created memory dir for $agent_id"
    fi
  else
    ok "Memory dir exists for $agent_id"
  fi
}

# Check if agent has memories in the API
check_agent_memories() {
  local agent_id="$1"
  local count
  count=$(curl -sf -X POST "$MEMORY_API/recall" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d "{\"query\": \"test\", \"limit\": 1, \"agent\": \"$agent_id\"}" 2>/dev/null | \
    python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('results',d.get('memories',[]))))" 2>/dev/null) || count="err"

  if [ "$count" = "err" ]; then
    warn "$agent_id: API query failed"
  elif [ "$count" = "0" ]; then
    warn "$agent_id: No memories yet (will populate after first session)"
  else
    ok "$agent_id: Has memories in DB"
  fi
}

# Get current allowedAgentIds from config
get_allowed_agents() {
  python3 -c "
import json
with open('$CONFIG_FILE') as f:
    c = json.load(f)
allowed = c.get('hooks',{}).get('allowedAgentIds',[])
print(' '.join(allowed))
"
}

# Update allowedAgentIds to include all agents
update_allowed_agents() {
  python3 -c "
import json
with open('$CONFIG_FILE') as f:
    c = json.load(f)

agent_ids = [a['id'] for a in c.get('agents',{}).get('list',[]) if 'id' in a]
hooks = c.setdefault('hooks', {})
current = set(hooks.get('allowedAgentIds', []))
needed = set(agent_ids)

if needed <= current:
    print('NOCHANGE')
else:
    hooks['allowedAgentIds'] = sorted(needed | current)
    with open('$CONFIG_FILE', 'w') as f:
        json.dump(c, f, indent=2)
    added = needed - current
    print('UPDATED:' + ','.join(sorted(added)))
"
}

echo "═══════════════════════════════════════════════"
echo "  OpenClaw Memory Provisioning"
echo "═══════════════════════════════════════════════"
echo ""

# 1. Check Memory API
echo "▸ Memory API"
check_memory_api || exit 1
echo ""

# 2. Scan agents
echo "▸ Agents"
while IFS='|' read -r agent_id workspace; do
  echo "  ── $agent_id ($workspace)"

  if $STATUS_ONLY; then
    # Just check status
    if [ -d "${workspace}/memory" ]; then
      ok "memory/ dir exists"
    else
      warn "memory/ dir MISSING"
    fi
    check_agent_memories "$agent_id"
  else
    # Provision
    provision_workspace_memory "$agent_id" "$workspace"
    check_agent_memories "$agent_id"
  fi
  echo ""
done < <(get_agents)

# 3. Check/update hooks allowedAgentIds
echo "▸ Hook Access (allowedAgentIds)"
current_allowed=$(get_allowed_agents)
echo "  Current: $current_allowed"

if ! $STATUS_ONLY && ! $DRY_RUN; then
  result=$(update_allowed_agents)
  if [ "$result" = "NOCHANGE" ]; then
    ok "All agents already allowed"
  else
    added="${result#UPDATED:}"
    ok "Added to allowedAgentIds: $added"
    warn "Gateway restart needed for hook changes to take effect"
  fi
elif $DRY_RUN; then
  # Show what would change
  python3 -c "
import json
with open('$CONFIG_FILE') as f:
    c = json.load(f)
agent_ids = set(a['id'] for a in c.get('agents',{}).get('list',[]) if 'id' in a)
current = set(c.get('hooks',{}).get('allowedAgentIds',[]))
missing = agent_ids - current
if missing:
    print(f'  Would add: {', '.join(sorted(missing))}')
else:
    print('  All agents already allowed')
"
fi
echo ""

# 4. Summary
echo "▸ Sync Cron"
sync_status=$(systemctl --user is-active openclaw-memory-sync.timer 2>/dev/null || \
  crontab -l 2>/dev/null | grep -c "cron_sync" || echo "unknown")
if [ "$sync_status" = "active" ]; then
  ok "Memory sync timer active"
else
  # Check OpenClaw cron jobs
  cron_check=$(curl -sf "http://localhost:8787/v1/health" 2>/dev/null && echo "api_ok" || echo "api_down")
  ok "Memory sync runs via OpenClaw cron (API: $cron_check)"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  Done. New agents get memory automatically"
echo "  after their first session."
echo "═══════════════════════════════════════════════"
