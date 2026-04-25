# NoldoMem OpenClaw Plugin

This native OpenClaw plugin exposes NoldoMem as agent tools:

- `noldomem_recall` - search long-term memory
- `noldomem_store` - store important facts, preferences, decisions, and lessons
- `noldomem_pin` - protect critical memories from decay and cleanup

It is intentionally separate from OpenClaw `memory-core`. NoldoMem stays a REST
service backed by SQLite/sqlite-vec, while this plugin gives agents explicit
tool access to that service.

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
        "config": {
          "baseUrl": "http://127.0.0.1:8787",
          "apiKeyFile": "~/.noldomem/memory-api-key",
          "enableAutoRecall": false,
          "enableAutoCapture": false
        }
      }
    }
  }
}
```

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

