---
name: claim-scanner
description: "Post-response claim scanner: detect unverified feature/config claims and log them"
metadata:
  openclaw:
    emoji: "🔎"
    events: ["message:sent"]
    requires:
      config: ["workspace.dir"]
---

# Claim Scanner Hook

Triggers on `message:sent`. Scans agent responses for unverified claims
(negative feature claims, positive completion claims, config/credential claims)
and checks if tool evidence exists in the recent tool call history.

## What It Does

1. Scans response for claim patterns (Turkish + English)
2. Checks recent tool calls for verification evidence
3. If no evidence found:
   - Stores `unverified_claim` to NoldoMem
   - Logs to `fabrication-log.json`
4. Does NOT block the response — post-mortem audit only

## Claim Types

- **Negative**: "desteklemiyor", "yok", "not supported", "doesn't have"
- **Positive**: "düzelttim", "güncelledim", "fixed", "updated N files"
- **Config**: hex strings, hash references, credential mentions

## Requirements

- Memory API running at `localhost:8787`
- Shared state module at `../lib/shared-state.js`
