# NoldoMem OpenClaw Plugin

This native OpenClaw plugin exposes NoldoMem as agent tools:

- `noldomem_recall` - search long-term memory
- `noldomem_store` - store important facts, preferences, decisions, and lessons
- `noldomem_pin` - protect critical memories from decay and cleanup
- `noldomem_forget` - delete an assertion and its connected revision history
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
            "before_compaction": 30000,
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
On OpenClaw 2026.5.20+, keep `before_compaction` at the host's 30 second
default so NoldoMem's bounded capture request has room to finish. These
timeouts do not change explicit `noldomem_recall`, `noldomem_store`, or
`noldomem_pin` tool calls.
Operational capture also ignores those NoldoMem tools, so a memory tool call
does not recursively trigger another memory write.

Restart OpenClaw after installing.

## Plugin vs Hook Pack

Use the typed plugin for current-turn automatic recall (`enableAutoRecall`) and
capture (`enableAutoCapture`). Both remain opt-in. Declarative prompts can recall
history; trivial acknowledgements skip search. An optional
`recallMinSemanticScore` filters automatic context using a model-calibrated floor;
there is no universal default. Explicit recall remains available in degraded mode.

The older hook pack supplies bootstrap and channel hooks. Enabling it alongside
the same typed plugin capture/injection events can duplicate storage, retrieval
and context. JSONL sync is an archive/legacy path, not a reader for the current
stable host's canonical SQLite sessions. Do not infer live coverage from its
successful scan.

The plugin binds tools to the host's factory context. Missing/mismatched agent
identity fails closed; a model cannot select another agent through tool arguments.
Use a distinct scoped API key per agent as the server-side boundary. Subtask
capture accepts only the same agent's target session. Confirmed `message_sent`
text is labeled delivered; `agent_end` alone is not delivery proof.

See [platform evidence](../docs/platform-memory-alignment-2026-09-09.md) for the
native/coexistence choices and remaining stable-host test limitations.

`noldomem_forget` accepts a recalled `memory_id` for an explicit user forgetting
request. It deletes that assertion and its revision family in the current agent
scope; original transcripts and other stores remain separate.
