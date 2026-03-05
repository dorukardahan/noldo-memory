import fs from "node:fs/promises";
import path from "node:path";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH = process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;

async function getApiKey() {
  try {
    return (await fs.readFile(API_KEY_PATH, "utf-8")).trim();
  } catch {
    return "";
  }
}

/**
 * Pin high-importance memories from recent session to protect from decay.
 * Queries recent memories for this agent and pins those with importance >= 0.7.
 */
async function pinCriticalMemories(agentId = "main") {
  try {
    const apiKey = await getApiKey();
    if (!apiKey) return;

    // Get recent high-importance memories (last 24h)
    const res = await fetch(`${MEMORY_API}/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
      body: JSON.stringify({ query: "important decision preference rule", limit: 10, agent: agentId }),
      signal: AbortSignal.timeout(10000),
    });
    const data = await res.json();
    const results = data.results || [];

    let pinned = 0;
    for (const r of results) {
      if ((r.importance || 0) >= 0.7) {
        try {
          await fetch(`${MEMORY_API}/pin`, {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
            body: JSON.stringify({ id: r.id, agent: agentId }),
            signal: AbortSignal.timeout(3000),
          });
          pinned++;
        } catch { /* non-critical */ }
      }
    }
    if (pinned > 0) {
      console.warn(`[pre-session-save] pinned ${pinned} critical memories for ${agentId}`);
    }
  } catch (err) {
    console.warn(`[pre-session-save] pinCriticalMemories error: ${err.message}`);
  }
}

/**
 * Derive agent ID from workspace directory path.
 *   $HOME/.openclaw/workspace        -> "main"
 *   $HOME/.openclaw/workspace-bureau  -> "bureau"
 */
function getAgentId(workspaceDir) {
  const base = path.basename(workspaceDir);
  if (base === "workspace") return "main";
  if (base.startsWith("workspace-")) return base.replace("workspace-", "");
  return "main";
}

async function resolveSessionFile(sessionFilePath) {
  try {
    await fs.access(sessionFilePath);
    return sessionFilePath;
  } catch {
    // OpenClaw renames session files to *.reset.{timestamp} on /new
    const dir = path.dirname(sessionFilePath);
    const base = path.basename(sessionFilePath);
    const entries = await fs.readdir(dir);
    const resetFile = entries
      .filter((f) => f.startsWith(base + ".reset."))
      .sort()
      .pop(); // most recent
    if (resetFile) {
      console.warn(`[pre-session-save] original gone, using ${resetFile}`);
      return path.join(dir, resetFile);
    }
    return null;
  }
}

async function getRecentMessages(sessionFilePath, count = 5) {
  try {
    const resolved = await resolveSessionFile(sessionFilePath);
    if (!resolved) {
      console.warn(`[pre-session-save] no session file found for ${sessionFilePath}`);
      return [];
    }
    const raw = await fs.readFile(resolved, "utf-8");
    const lines = raw.trim().split("\n");
    const messages = [];

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry.type === "message" && entry.message) {
          const msg = entry.message;
          if ((msg.role === "user" || msg.role === "assistant") && msg.content) {
            const text = Array.isArray(msg.content)
              ? msg.content.find((c) => c.type === "text")?.text
              : msg.content;
            if (text && !text.startsWith("/")) {
              messages.push({ role: msg.role, text: text.slice(0, 500) });
            }
          }
        }
      } catch {
        // skip malformed lines
      }
    }

    return messages.slice(-count);
  } catch (err) {
    console.warn(`[pre-session-save] getRecentMessages error: ${err.message}`);
    return [];
  }
}

async function captureToMemoryAPI(content, agentId = "main") {
  try {
    await fetch(`${MEMORY_API}/capture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "assistant", text: content }],
        agent: agentId,
        metadata: { source: "pre-session-save-hook", timestamp: new Date().toISOString() },
      }),
      signal: AbortSignal.timeout(5000),
    });
  } catch {
    // non-critical
  }
}

const preSessionSaveHook = async (event) => {
  if (event.type !== "command" || event.action !== "new") return;

  console.warn(`[pre-session-save] triggered for ${event.sessionKey}`);

  const context = event.context || {};
  const cfg = context.cfg;
  const workspaceDir =
    context.workspaceDir || cfg?.workspace?.dir || process.env.OPENCLAW_WORKSPACE || `${process.env.HOME}/.openclaw/workspace`;
  const memoryDir = path.join(workspaceDir, "memory");

  await fs.mkdir(memoryDir, { recursive: true });

  const sessionEntry = context.previousSessionEntry || context.sessionEntry || {};
  const sessionFile = sessionEntry.sessionFile;

  console.warn(`[pre-session-save] sessionFile: ${sessionFile || "NONE"}`);

  const snapshot = {
    timestamp: new Date().toISOString(),
    sessionKey: event.sessionKey || "unknown",
    recentMessages: [],
  };

  if (sessionFile) {
    snapshot.recentMessages = await getRecentMessages(sessionFile, 5);
    console.warn(`[pre-session-save] parsed ${snapshot.recentMessages.length} messages`);
  } else {
    console.warn("[pre-session-save] no sessionFile — skipping");
  }

  if (snapshot.recentMessages.length > 0) {
    const summaryParts = snapshot.recentMessages.map((m) => `${m.role}: ${m.text}`);
    const summaryText = `Session snapshot (${snapshot.timestamp}):\n${summaryParts.join("\n")}`;

    const snapshotPath = path.join(memoryDir, "critical-context-snapshot.json");
    await fs.writeFile(snapshotPath, JSON.stringify(snapshot, null, 2), "utf-8");
    console.warn(`[pre-session-save] snapshot written to ${snapshotPath}`);

    // NOTE: Removed captureToMemoryAPI() call here — built-in session-memory
    // hook already captures to memory/YYYY-MM-DD-slug.md on command:new.
    // Having both created duplicate entries. This hook now only writes the
    // snapshot JSON (used by post-compaction-restore). [S4, 2026-02-17]
    // Pin critical memories to protect from decay/gc
    const agentId = getAgentId(workspaceDir);
    await pinCriticalMemories(agentId);
  } else {
    console.warn("[pre-session-save] no messages to save — skipping write");
    // Still pin critical memories even without new messages
    const agentId = getAgentId(workspaceDir);
    await pinCriticalMemories(agentId);
  }
};

export default preSessionSaveHook;
