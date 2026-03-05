/**
 * post-compaction-restore — Two-phase compaction recovery hook.
 *
 * Phase 1 (after_compaction): Write a compaction flag file with metadata.
 * Phase 2 (agent:bootstrap): If flag is recent (< 2hrs), inject snapshot
 *   as COMPACTION_RECOVERY bootstrap file.
 *
 * Previously this hook was misnamed — it fired on every bootstrap, not
 * specifically after compaction. Now it correctly uses after_compaction
 * for the write phase. [S11, 2026-02-17]
 */

import fs from "node:fs/promises";
import path from "node:path";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH = process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
const ALLOW_LEGACY_SNAPSHOT_RESTORE =
  process.env.OPENCLAW_ALLOW_LEGACY_SNAPSHOT_RESTORE === "1";

async function getApiKey() {
  try {
    return (await fs.readFile(API_KEY_PATH, "utf-8")).trim();
  } catch {
    return "";
  }
}

function getAgentId(workspaceDir) {
  const base = path.basename(workspaceDir);
  if (base === "workspace") return "main";
  if (base.startsWith("workspace-")) return base.replace("workspace-", "");
  return "main";
}

/**
 * Post-restore validation: verify pinned/critical memories are still accessible.
 * Queries memory API for high-importance items and logs coverage.
 */
async function validateMemoryIntegrity(agentId, snapshotMessages) {
  try {
    const apiKey = await getApiKey();
    if (!apiKey) return;

    // Check if pinned memories are accessible
    const res = await fetch(`${MEMORY_API}/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
      body: JSON.stringify({
        query: "important decision preference rule",
        limit: 5,
        agent: agentId,
      }),
      signal: AbortSignal.timeout(10000),
    });
    const data = await res.json();
    const count = data.count || 0;

    if (count === 0) {
      console.warn(
        `[post-compaction-restore] ⚠️ AMNESIA WARNING: no critical memories found for ${agentId}`
      );
    } else {
      console.warn(
        `[post-compaction-restore] ✅ Memory validation OK: ${count} critical memories accessible for ${agentId}`
      );
    }

    // Cross-check: do snapshot topics appear in memory?
    if (snapshotMessages && snapshotMessages.length > 0) {
      const topicQuery = snapshotMessages
        .map((m) => m.text.slice(0, 50))
        .join(" ");
      const topicRes = await fetch(`${MEMORY_API}/recall`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
        body: JSON.stringify({ query: topicQuery.slice(0, 500), limit: 3, agent: agentId }),
        signal: AbortSignal.timeout(10000),
      });
      const topicData = await topicRes.json();
      const topicCount = topicData.count || 0;
      console.warn(
        `[post-compaction-restore] Topic coverage: ${topicCount}/3 matches for recent session topics`
      );
    }
  } catch (err) {
    console.warn(`[post-compaction-restore] validation error: ${err.message}`);
  }
}

const TWO_HOURS_MS = 2 * 60 * 60 * 1000;

async function readRecentMessagesFromTranscript(sessionFile, limit = 8) {
  if (!sessionFile) return [];
  try {
    const raw = await fs.readFile(sessionFile, "utf-8");
    const lines = raw.trim().split("\n");
    const messages = [];
    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry.type !== "message") continue;
        const msg = entry.message;
        const role = msg?.role;
        if (role !== "user" && role !== "assistant") continue;
        const text = Array.isArray(msg?.content)
          ? msg.content.find((c) => c.type === "text")?.text
          : msg?.content;
        if (!text || text.startsWith("/")) continue;
        messages.push({ role, text: text.slice(0, 500) });
      } catch {
        // ignore malformed jsonl line
      }
    }
    return messages.slice(-limit);
  } catch (err) {
    console.warn(`[post-compaction-restore] transcript read failed: ${err.message}`);
    return [];
  }
}

async function consumeCompactionFlag(flagPath, reason) {
  try {
    await fs.unlink(flagPath);
    console.warn(`[post-compaction-restore] consumed compaction flag (${reason})`);
  } catch {
    // best-effort
  }
}

// --- Phase 1: after_compaction — record that compaction happened ---
async function handleCompaction(event) {
  const context = event.context || {};
  const workspaceDir =
    context.workspaceDir || context.cfg?.workspace?.dir || process.env.OPENCLAW_WORKSPACE || `${process.env.HOME}/.openclaw/workspace`;
  const flagPath = path.join(workspaceDir, "memory", "compaction-flag.json");

  const flag = {
    timestamp: new Date().toISOString(),
    messageCount: event.messageCount || 0,
    compactedCount: event.compactedCount || 0,
    tokenCount: event.tokenCount || null,
    sessionFile: event.sessionFile || null,
  };

  try {
    const memoryDir = path.join(workspaceDir, "memory");
    const snapshotPath = path.join(memoryDir, "critical-context-snapshot.json");
    await fs.mkdir(memoryDir, { recursive: true });
    await fs.writeFile(flagPath, JSON.stringify(flag, null, 2), "utf-8");

    if (flag.sessionFile) {
      const recentMessages = await readRecentMessagesFromTranscript(flag.sessionFile, 8);
      if (recentMessages.length > 0) {
        const snapshot = {
          timestamp: new Date().toISOString(),
          sessionKey: event.sessionKey || "compaction-session",
          recentMessages,
        };
        await fs.writeFile(snapshotPath, JSON.stringify(snapshot, null, 2), "utf-8");
      }
    }
    console.warn(
      `[post-compaction-restore] compaction recorded: ${flag.compactedCount} messages compacted`
    );
  } catch (err) {
    console.warn(`[post-compaction-restore] failed to write flag: ${err.message}`);
  }
}

// --- Phase 2: agent:bootstrap — restore if compaction was recent ---
async function handleBootstrap(event) {
  const context = event.context;
  if (!context || !context.bootstrapFiles) return;

  const workspaceDir =
    context.workspaceDir || context.cfg?.workspace?.dir || process.env.OPENCLAW_WORKSPACE || `${process.env.HOME}/.openclaw/workspace`;
  const flagPath = path.join(workspaceDir, "memory", "compaction-flag.json");
  const snapshotPath = path.join(workspaceDir, "memory", "critical-context-snapshot.json");

  // Check if compaction happened recently
  let compactionRecent = false;
  try {
    const flagStat = await fs.stat(flagPath);
    compactionRecent = Date.now() - flagStat.mtimeMs < TWO_HOURS_MS;
    if (!compactionRecent) {
      console.warn("[post-compaction-restore] compaction flag too old, skipping");
      await consumeCompactionFlag(flagPath, "stale");
      return;
    }
  } catch {
    if (!ALLOW_LEGACY_SNAPSHOT_RESTORE) {
      return;
    }
    console.warn("[post-compaction-restore] legacy restore path enabled (no compaction flag)");
  }

  try {
    const stat = await fs.stat(snapshotPath);
    const ageMs = Date.now() - stat.mtimeMs;
    console.warn(
      `[post-compaction-restore] snapshot age: ${Math.round(ageMs / 60000)}min, compaction_recent=${compactionRecent}`
    );

    // Only restore if snapshot is less than 2 hours old
    if (ageMs > TWO_HOURS_MS) {
      console.warn("[post-compaction-restore] snapshot too old, skipping");
      if (compactionRecent) {
        await consumeCompactionFlag(flagPath, "snapshot-stale");
      }
      return;
    }

    const raw = await fs.readFile(snapshotPath, "utf-8");
    const snapshot = JSON.parse(raw);

    if (!snapshot.recentMessages || snapshot.recentMessages.length === 0) {
      console.warn("[post-compaction-restore] no messages in snapshot");
      return;
    }

    const lines = [
      "# Previous Session Context (Auto-Restored)",
      "",
      `> Snapshot from: ${snapshot.timestamp}`,
      `> Session: ${snapshot.sessionKey || "unknown"}`,
      compactionRecent ? "> Trigger: post-compaction recovery" : "> Trigger: session continuity",
      "",
      "**IMPORTANT**: This context was auto-restored from the previous session.",
      "Continue the conversation naturally. Do NOT greet as if starting a new day.",
      "If the user was mid-task, pick up where they left off.",
      "",
      "## Last Messages",
      "",
    ];

    for (const msg of snapshot.recentMessages) {
      lines.push(`**${msg.role}**: ${msg.text}`);
      lines.push("");
    }

    context.bootstrapFiles.push({
      name: "COMPACTION_RECOVERY",
      path: "COMPACTION_RECOVERY",
      content: lines.join("\n"),
    });
    console.warn(
      `[post-compaction-restore] injected COMPACTION_RECOVERY (${snapshot.recentMessages.length} msgs)`
    );
    if (compactionRecent) {
      await consumeCompactionFlag(flagPath, "restored");
    }

    // Post-restore validation: verify memory integrity
    const agentId = getAgentId(workspaceDir);
    await validateMemoryIntegrity(agentId, snapshot.recentMessages);
  } catch (err) {
    console.warn(`[post-compaction-restore] ${err.code || err.message}`);
  }
}

// --- Router: dispatch based on event type ---
const postCompactionRestoreHook = async (event) => {
  // Phase 1: after_compaction event
  if (event.type === "compaction" || event.action === "after_compaction") {
    return handleCompaction(event);
  }

  // Also handle the direct runner call pattern (runAfterCompaction passes data directly)
  if (event.compactedCount !== undefined && event.messageCount !== undefined) {
    return handleCompaction(event);
  }

  // Phase 2: agent:bootstrap
  if (event.type === "agent" && event.action === "bootstrap") {
    return handleBootstrap(event);
  }
};

export default postCompactionRestoreHook;
