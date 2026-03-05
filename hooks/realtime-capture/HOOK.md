---
name: realtime-capture
description: "Capture user messages and assistant responses to memory in real-time"
metadata:
  openclaw:
    emoji: "⚡"
    events: ["message:received", "message:sent"]
---

# Realtime Capture Hook

Captures high-signal messages as they happen.

## Memory Routing Strategy

- Uses `agentId` from event context when available (OpenClaw-native), with safe fallback parsing.
- Stores conversational memories into a deterministic session namespace derived from `sessionKey`.
- Stores cross-session lessons/decisions in `default` namespace. Because Noldo uses one DB per agent/workspace, this stays workspace-local while remaining reusable across sessions.
