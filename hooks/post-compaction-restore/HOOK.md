---
name: post-compaction-restore
description: "Restore compacted context on next bootstrap when compaction flag exists"
metadata:
  openclaw:
    emoji: "🔄"
    events: ["agent:bootstrap"]
    requires:
      config: ["workspace.dir"]
---

# Post-Compaction Restore Hook

Runs on `agent:bootstrap`. If a recent `memory/compaction-flag.json` exists,
it restores `critical-context-snapshot.json` into bootstrap as `COMPACTION_RECOVERY`.
