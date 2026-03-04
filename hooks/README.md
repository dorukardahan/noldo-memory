# OpenClaw Hooks for Agent Memory

This directory contains documented example hooks for integrating OpenClaw with the Agent Memory API.

These files are shipped as `handler.js.example` on purpose. Copy or rename to `handler.js` in your OpenClaw hooks directory to activate them.

## Hook Overview

| Hook | Trigger event(s) | What it does | Memory API endpoint |
|---|---|---|---|
| `realtime-capture` | `message:received`, `message:sent` | Captures important user/assistant messages in real time with per-agent routing | `POST /v1/store` |
| `session-end-capture` | `command:new`, `command:reset` | Captures session-transition context, builds QA pairs, writes `last-session.md` | `POST /v1/capture` |
| `bootstrap-context` | `agent:bootstrap` | Recalls recent memories and injects `SESSION_CONTEXT` on session start | `POST /v1/recall` |
| `after-tool-call` | `after_tool_call` (plugin hook) | Captures high-signal operational command outputs | `POST /v1/store` |
| `post-compaction-restore` | `agent:bootstrap` | Restores `COMPACTION_RECOVERY` bootstrap context when compaction flag exists | No API call (filesystem snapshot restore) |
| `pre-session-save` | `command:new` | Saves critical context snapshot before a reset/new | No API call by default (optional `POST /v1/capture`) |
| `subagent-complete` | `subagent_ended` (plugin hook) | Logs subagent completion/errors and stores failure context for post-mortem | `POST /v1/capture` (error cases) |

## Directory Layout

Each hook directory contains:

- `HOOK.md` - metadata and behavior notes
- `handler.js.example` - sanitized example implementation

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
```

Notes:

- Example handlers default to `http://localhost:8787/v1` for Memory API.
- Keep credentials in environment/files, not in `handler.js`.
- Do not commit active `handler.js` files with production secrets.

## Installation

1. Copy hooks to your OpenClaw hook directory, or point `hooks.internal.load.extraDirs` to this repo.
2. For each hook, copy `handler.js.example` to `handler.js`.
3. Restart OpenClaw.

Example:

```bash
mkdir -p "$HOME/.openclaw/hooks"
cp -R hooks/* "$HOME/.openclaw/hooks/"

for d in "$HOME"/.openclaw/hooks/*; do
  cp "$d/handler.js.example" "$d/handler.js"
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
          "/path/to/openclaw/hooks/bootstrap-context",
          "/path/to/openclaw/hooks/pre-session-save",
          "/path/to/openclaw/hooks/post-compaction-restore",
          "/path/to/openclaw/hooks/session-end-capture",
          "/path/to/openclaw/hooks/realtime-capture"
        ]
      }
    }
  }
}
```
