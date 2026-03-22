---
name: message-recall
description: "Mid-conversation recall: inject relevant memories when user asks feature/config/status questions"
metadata:
  openclaw:
    emoji: "🔍"
    events: ["message:received"]
    requires:
      config: ["workspace.dir"]
---

# Message Recall Hook

Triggers on `message:received`. Detects feature/config/status questions and injects
relevant lessons and verified facts into the conversation context.

## What It Does

1. Checks if user message matches question patterns (Turkish + English)
2. Extracts keywords from the question
3. Recalls relevant lessons, verified facts, and incident memories from NoldoMem
4. Injects results as a system message via `event.messages`
5. Timeout: 2 seconds max, graceful skip on timeout

## Question Patterns

- "X var mı?", "X destekleniyor mu?", "X çalışıyor mu?"
- "does X support Y?", "is X running?", "does X have Y?"
- Config, feature, service status questions

## Requirements

- Memory API running at `localhost:8787`
- API key at `~/.noldomem/memory-api-key`
