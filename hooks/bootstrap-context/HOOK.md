---
name: bootstrap-context
description: "Inject memory recall and daily notes at session start"
metadata:
  openclaw:
    emoji: "\U0001F9E0"
    events: ["agent:bootstrap"]
    requires:
      config: ["workspace.dir"]
---

# Bootstrap Context Hook

Injects relevant memory context at the start of every agent session.

## What It Does

1. Calls the Agent Memory API (`/v1/recall`) to fetch session-scoped memories first
2. Falls back to broader agent recall for durable lessons and decisions
3. Reads today's and yesterday's daily notes from `workspace/memory/`
4. Pushes combined context as `SESSION_CONTEXT` bootstrap file

## Workspace policy

If present, `workspace/.openclaw/noldo-memory.json` can tune bootstrap behavior:

- `crossWorkspaceRecall`: opt-in limited recall from the `main` agent/workspace
- `sharedNamespaces`: include extra namespaces during bootstrap
- `dailyNotesEnabled`: disable daily note injection for that workspace

## Requirements

- Memory API running at `localhost:8787`
- Daily notes in `workspace/memory/YYYY-MM-DD*.md` format
