---
name: before-compaction
description: "Capture critical session context to NoldoMem before compaction destroys it"
metadata:
  openclaw:
    emoji: "🛡️"
    events: ["before_compaction"]
    requires:
      config: ["workspace.dir"]
---

# Before Compaction Hook

Fires just before OpenClaw compacts (summarizes) the conversation. Captures
the most important recent messages to NoldoMem so they survive compaction.

## Why This Exists

Compaction summarizes older messages and replaces them with a compact summary.
Any context not saved to memory files or NoldoMem is effectively lost.
This hook ensures critical decisions, preferences, and context are preserved.

## What It Does

1. Receives the full message array from the compaction event
2. Extracts the last N user+assistant messages (configurable, default 30)
3. Filters out low-signal messages (system, cron, subagent metadata)
4. Scores each message for importance (decisions, feedback, errors get priority)
5. Sends top messages to NoldoMem `/v1/capture` as a batch
6. Also writes a `memory/compaction-snapshot.md` with key decisions for file-based recall

## Requirements

- Memory API running at `localhost:8787`
