# NoldoMem OpenClaw Plugin

This native OpenClaw plugin exposes NoldoMem as agent tools:

- `noldomem_recall` - search long-term memory
- `noldomem_store` - store important facts, preferences, decisions, and lessons
- `noldomem_pin` - protect critical memories from decay and cleanup
- native typed hooks for operational tool capture, compaction capture, and
  subagent failure capture

It is intentionally separate from OpenClaw `memory-core`. NoldoMem stays a REST
service backed by SQLite/sqlite-vec, while this plugin gives agents explicit
tool access to that service.

The plugin is dependency-free and declares `openclaw.extensions`, so a local
`openclaw plugins install -l ./plugin` uses the current OpenClaw 2026.5.2+
installer path without an extra npm install step.

## Install

From the repo root:

```bash
openclaw plugins install -l ./plugin
```

Then enable it in `openclaw.json`:

```json
{
  "plugins": {
    "allow": ["noldomem"],
    "entries": {
      "noldomem": {
        "enabled": true,
        "hooks": {
          "allowPromptInjection": false,
          "timeoutMs": 5000,
          "timeouts": {
            "after_tool_call": 3000,
            "before_compaction": 10000,
            "subagent_ended": 3000
          }
        },
        "config": {
          "baseUrl": "http://127.0.0.1:8787",
          "apiKeyFile": "~/.noldomem/memory-api-key",
          "enableAutoRecall": false,
          "enableAutoCapture": false,
          "enableOperationalCapture": true,
          "enableCompactionCapture": true,
          "enableSubagentCapture": true
        }
      }
    }
  }
}
```

`hooks.timeoutMs` and `hooks.timeouts` are supported by OpenClaw 2026.5.3+.
They bound optional lifecycle hooks without changing explicit
`noldomem_recall`, `noldomem_store`, or `noldomem_pin` tool calls.

Restart OpenClaw after installing.

## Plugin vs Hook Pack

Use both pieces for the full custom-memory workflow:

- `plugin/` gives agents active tools for recall/store/pin.
- `hooks/` handles lifecycle capture and bootstrap injection, especially
  `agent:bootstrap`, `message:received`, `message:sent`, and `/new` session
  transitions.

Keep `enableAutoRecall=false` unless you explicitly want the native plugin to
run a recall check before prompt build. The hook pack already handles bootstrap
recall and is cheaper for normal operation.

If the agent has an explicit tool allow list, remove OpenClaw's native
`memory_search` and `memory_get` tools when NoldoMem is the intended memory
system. Keep `noldomem_recall`, `noldomem_store`, and `noldomem_pin` allowed.
