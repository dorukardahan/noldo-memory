# OpenClaw Hooks for Agent Memory

This directory contains installable OpenClaw hooks for integrating with the Agent Memory API.

Each hook ships with:

- `handler.js` - installable, sanitized runtime handler used by `openclaw hooks install`
- `handler.js.example` - mirrored reference file for manual copy workflows

The recommended path is to link this directory as an OpenClaw hook pack so updates stay attached to the repo.

## Hook Overview

| Hook | Trigger event(s) | What it does | Memory API endpoint |
|---|---|---|---|
| `realtime-capture` | `message:received`, `message:sent` | Captures conversational memory into session namespaces while keeping durable lessons/decisions in the workspace-default namespace | `POST /v1/store` |
| `session-end-capture` | `command:new`, `command:reset` | Captures session-transition context, builds QA pairs, writes `last-session.md` with per-session namespace capture | `POST /v1/capture` |
| `bootstrap-context` | `agent:bootstrap` | Recalls session-scoped memories first, then workspace-durable lessons/decisions, injects `SESSION_CONTEXT`; skips heavy recall for cron sessions | `POST /v1/recall` for interactive sessions |
| `after-tool-call` | `after_tool_call` (plugin hook) | Captures high-signal operational command outputs | `POST /v1/store` |
| `post-compaction-restore` | `agent:bootstrap` | Restores `COMPACTION_RECOVERY` bootstrap context when compaction flag exists | No API call (filesystem snapshot restore) |
| `pre-session-save` | `command:new` | Saves critical context snapshot before a reset/new | No API call by default (optional `POST /v1/capture`) |
| `subagent-complete` | `subagent_ended` (plugin hook) | Logs subagent completion/errors and stores failure context for post-mortem | `POST /v1/capture` (error cases) |

## Directory Layout

Each hook directory contains:

- `HOOK.md` - metadata and behavior notes
- `handler.js` - sanitized installable handler
- `handler.js.example` - mirrored example implementation

## Required Environment Variables

Set these where OpenClaw runs:

```bash
export AGENT_MEMORY_API_KEY_FILE="$HOME/.noldomem/memory-api-key"
export OPENCLAW_DIR="$HOME/.openclaw"
export OPENCLAW_WORKSPACE="$HOME/.openclaw/workspace"
```

Optional:

```bash
export OPENCLAW_ENABLE_CROSS_AGENT_RECALL="1"
export OPENCLAW_ALLOW_LEGACY_SNAPSHOT_RESTORE="0"
```

Notes:

- Handlers default to `http://localhost:8787/v1` for Memory API.
- Keep credentials in environment/files, not in `handler.js`.
- The committed `handler.js` files are sanitized and safe to publish. Keep local overrides secret-free.

## Installation

1. Link this directory as an OpenClaw hook pack:

```bash
openclaw hooks install -l /path/to/noldo-memory/hooks
```

2. Restart OpenClaw.

This records the install under `hooks.internal.installs`, adds this directory to
`hooks.internal.load.extraDirs`, and keeps the hook source update-safe.

Fallback manual mode:

1. Copy hooks to your OpenClaw hook directory.
2. For each hook, copy `handler.js.example` to `handler.js`.
3. Restart OpenClaw.

Example:

```bash
mkdir -p "$HOME/.openclaw/hooks"
cp -R hooks/* "$HOME/.openclaw/hooks/"

for d in "$HOME"/.openclaw/hooks/*; do
  if [ -f "$d/handler.js.example" ]; then
    cp "$d/handler.js.example" "$d/handler.js"
  fi
done
```

Config snippet:

```json
{
  "hooks": {
    "enabled": true,
    "internal": {
      "enabled": true,
      "load": {
        "extraDirs": [
          "/path/to/noldo-memory/hooks"
        ]
      }
    }
  }
}
```

## Workspace policy file

Hooks optionally read `workspace/.openclaw/noldo-memory.json`:

```json
{
  "crossWorkspaceRecall": false,
  "sharedNamespaces": [],
  "dailyNotesEnabled": true
}
```

- `crossWorkspaceRecall`: if `true`, bootstrap may recall limited shared context from `main`
- `sharedNamespaces`: additional namespaces to include during bootstrap recall
- `dailyNotesEnabled`: if `false`, skip `workspace/memory/YYYY-MM-DD*.md` injection
