---
name: session-end-capture
description: "Capture session content on /new and /reset transitions"
metadata:
  openclaw:
    emoji: "📥"
    events: ["command:new", "command:reset"]
    requires:
      config: ["workspace.dir"]
---

# Session End Capture Hook

Captures session content when a session is explicitly rotated with `/new` or `/reset`.

## Why This Exists

Built-in `session-memory` writes markdown summaries, but this hook also persists
the last dialog turns to Memory API and refreshes workspace snapshot files.

This hook listens to:
- `command:new`
- `command:reset`

This matches OpenClaw's internal hook contract.

## What It Does

1. Reads last 15 messages from the ending session's JSONL file
2. Builds Q&A pairs for proper memory structure
3. Sends to Memory API `/v1/capture` with per-agent + per-session namespace routing
4. Keeps auto-generated behavioral lessons in `default` namespace, which remains workspace-local because the DB is already per-agent/workspace

## Requirements

- Memory API running at `localhost:8787`
