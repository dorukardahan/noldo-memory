import fs from "node:fs/promises";
import path from "node:path";
import { readFileSync } from "node:fs";
import {
  deriveSessionNamespace,
  readWorkspacePolicy,
  resolveAgentId,
  resolveWorkspaceDir,
} from "../lib/runtime.js";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH =
  process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try {
  _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim();
} catch (e) {
  console.warn("[bootstrap-context] error:", e.message || e);
}

const MAX_MEMORY_CHARS = 8000;
const MAX_DAILY_CHARS = 6000;
const MAX_CROSSAGENT_CHARS = 2000;
const MAX_LAST_SESSION_CHARS = 2000;
const MAX_LESSON_CHARS = 3000;
const ENABLE_CROSS_AGENT_RECALL = process.env.OPENCLAW_ENABLE_CROSS_AGENT_RECALL === "1";

const CRON_NOISE_PATTERNS = [/^\[cron:/i, /steward-engage/i, /steward-post/i, /\/steward-/i];

const LOW_SIGNAL_MEMORY_PATTERNS = [
  /A new session was started via \/new or \/reset/i,
  /\[Subagent Context\]/i,
  /Conversation info \(untrusted metadata\)/i,
  /^User:\s*Conversation info/i,
  /\[System Message\]/i,
  /\[Queued announce messages while agent was busy\]/i,
  /^✅\s*Subagent\s+.+\s+finished/i,
  /\bA cron job\b/i,
  /\bA subagent task\b/i,
  /^System:\s*\[/i,
];

function isCronNoise(text = "") {
  return CRON_NOISE_PATTERNS.some((p) => p.test(text));
}

function isLowSignalMemory(text = "") {
  if (!text) return true;
  if (isCronNoise(text)) return true;
  return LOW_SIGNAL_MEMORY_PATTERNS.some((p) => p.test(text));
}

function truncateSection(text, maxChars) {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars) + `\n... [truncated to ${maxChars} chars]`;
}

function stripLowSignalText(text = "") {
  const raw = String(text || "");
  const withoutBlocks = raw.replace(
    /Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```/gi,
    ""
  );
  const cleanedLines = withoutBlocks
    .split("\n")
    .filter((line) => !LOW_SIGNAL_MEMORY_PATTERNS.some((p) => p.test(line)));
  return cleanedLines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

async function recall(agentId, query, options = {}) {
  if (!_memoryApiKey) return [];
  try {
    const payload = {
      query,
      limit: options.limit ?? 20,
      min_score: options.minScore ?? 0.01,
      agent: agentId,
    };
    if (options.namespace) payload.namespace = options.namespace;
    if (options.memoryType) payload.memory_type = options.memoryType;

    const res = await fetch(`${MEMORY_API}/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(options.timeoutMs ?? 5000),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.results || data.memories || [];
  } catch (e) {
    console.warn("[bootstrap-context] error:", e.message || e);
    return [];
  }
}

function dedupeMemories(items) {
  const deduped = [];
  const seen = new Set();
  for (const item of items || []) {
    const text = item?.text || item?.content || "";
    if (isLowSignalMemory(text)) continue;
    const key = text.slice(0, 120);
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(item);
  }
  return deduped;
}

async function fetchRecentMemories(agentId, sessionNamespace) {
  if (!_memoryApiKey) return [];

  const queries = [
    "recent decisions and configuration changes",
    "last conversation summary and active work",
    "credential updates, config file changes, environment variable modifications",
  ];

  const sessionScoped = [];
  if (sessionNamespace && sessionNamespace !== "default") {
    for (const q of queries) {
      const scoped = await recall(agentId, q, {
        namespace: sessionNamespace,
        limit: 20,
        minScore: 0.0,
        timeoutMs: 4000,
      });
      sessionScoped.push(...scoped);
    }
  }

  const dedupedSession = dedupeMemories(sessionScoped);
  if (dedupedSession.length >= 6) {
    return dedupedSession;
  }

  const global = [];
  for (const q of queries) {
    const query = agentId !== "main" ? `${agentId}: ${q}` : q;
    const results = await recall(agentId, query, { limit: 20, minScore: 0.01, timeoutMs: 5000 });
    global.push(...results);
  }

  return dedupeMemories([...dedupedSession, ...global]);
}

async function fetchSharedNamespaceMemories(agentId, sharedNamespaces = []) {
  const items = [];
  for (const namespace of sharedNamespaces) {
    const results = await recall(agentId, "shared operational context and reusable decisions", {
      namespace,
      limit: 6,
      minScore: 0.0,
      timeoutMs: 4000,
    });
    items.push(...results);
  }
  return dedupeMemories(items);
}

async function fetchDecisionMemories(agentId) {
  if (!_memoryApiKey) return null;
  return recall(agentId, "recent decisions, architectural choices, configuration changes", {
    limit: 8,
    minScore: 0.01,
    timeoutMs: 4000,
  });
}

async function fetchLessons(agentId) {
  if (!_memoryApiKey) return [];
  const results = await recall(agentId, "behavioral lessons", {
    limit: 20,
    minScore: 0.0,
    memoryType: "lesson",
    timeoutMs: 4000,
  });

  const sorted = results.sort((a, b) => {
    const imp = (Number(b.importance) || 0) - (Number(a.importance) || 0);
    if (imp !== 0) return imp;
    return (Number(b.created_at) || 0) - (Number(a.created_at) || 0);
  });

  const deduped = [];
  for (const item of sorted) {
    const text = stripLowSignalText(item.text || item.content || "");
    if (!text) continue;
    if (isLowSignalMemory(text)) continue;
    const isDuplicate = deduped.some((existing) => {
      const prev = stripLowSignalText(existing.text || existing.content || "");
      return prev.includes(text) || text.includes(prev);
    });
    if (!isDuplicate) deduped.push(item);
    if (deduped.length >= 5) break;
  }

  return deduped;
}

async function readLastSessionSummary(workspaceDir) {
  const filePath = path.join(workspaceDir, "memory", "last-session.md");
  try {
    const content = await fs.readFile(filePath, "utf-8");
    const cleaned = stripLowSignalText(content);
    if (cleaned) {
      return truncateSection(cleaned, MAX_LAST_SESSION_CHARS);
    }
  } catch (e) {
    if (e?.code !== "ENOENT") {
      console.warn("[bootstrap-context] error:", e.message || e);
    }
  }
  return null;
}

async function readDailyNotes(memoryDir) {
  const now = new Date();
  const today = now.toISOString().split("T")[0];
  const yesterday = new Date(now - 86400000).toISOString().split("T")[0];
  const parts = [];

  for (const date of [today, yesterday]) {
    try {
      const files = await fs.readdir(memoryDir);
      const matches = files.filter((f) => f.startsWith(date) && f.endsWith(".md"));
      for (const file of matches) {
        const content = await fs.readFile(path.join(memoryDir, file), "utf-8");
        const cleaned = stripLowSignalText(content);
        if (cleaned) parts.push(`## ${file}\n${cleaned}`);
      }
    } catch (e) {
      if (e?.code !== "ENOENT") {
        console.warn("[bootstrap-context] error:", e.message || e);
      }
    }
  }
  return parts.join("\n\n");
}

const bootstrapContextHook = async (event) => {
  if (event.type !== "agent" || event.action !== "bootstrap") return;

  const context = event.context;
  if (!context || !context.bootstrapFiles) return;

  const workspaceDir = resolveWorkspaceDir(event);
  const agentId = resolveAgentId(event, workspaceDir);
  const sessionNamespace = deriveSessionNamespace(event?.sessionKey);
  const memoryDir = path.join(workspaceDir, "memory");
  const policy = await readWorkspacePolicy(workspaceDir);

  console.warn(
    `[bootstrap-context] hook fired (agent=${agentId} namespace=${sessionNamespace})`
  );

  const sections = [];

  const lessons = await fetchLessons(agentId);
  if (lessons.length > 0) {
    console.warn(`[bootstrap-context] fetched ${lessons.length} lessons for ${agentId}`);
    const lessonLines = ["# Critical Lessons (Behavioral Memory)\n"];
    let lessonChars = 0;
    for (const l of lessons) {
      const text = stripLowSignalText(l.text || l.content || "");
      if (!text || isLowSignalMemory(text)) continue;
      const line = `- ${text.slice(0, 400)}`;
      lessonChars += line.length;
      if (lessonChars > MAX_LESSON_CHARS) break;
      lessonLines.push(line);
    }
    lessonLines.push("");
    sections.push(...lessonLines);
  }

  const memories = await fetchRecentMemories(agentId, sessionNamespace);
  if (memories && memories.length > 0) {
    console.warn(
      `[bootstrap-context] fetched ${memories.length} memories for ${agentId} namespace=${sessionNamespace}`
    );
    const memLines = ["# Recent Memory Recall\n"];
    let memChars = 0;
    for (const mem of memories) {
      const text = mem.text || mem.content || JSON.stringify(mem);
      if (isLowSignalMemory(text)) continue;
      const cat = mem.category ? ` [${mem.category}]` : "";
      const strength = mem.strength ? ` (strength: ${mem.strength})` : "";
      const line = `- ${text.slice(0, 300)}${cat}${strength}`;
      memChars += line.length;
      if (memChars > MAX_MEMORY_CHARS) break;
      memLines.push(line);
    }
    sections.push(...memLines);
  } else {
    console.warn(`[bootstrap-context] no memories for ${agentId}`);
  }

  const sharedNamespaceMemories = await fetchSharedNamespaceMemories(
    agentId,
    policy.sharedNamespaces || []
  );
  if (sharedNamespaceMemories.length > 0) {
    const sharedLines = ["\n# Shared Namespace Recall\n"];
    let sharedChars = 0;
    for (const mem of sharedNamespaceMemories.slice(0, 5)) {
      const text = mem.text || mem.content || "";
      if (isLowSignalMemory(text)) continue;
      const namespace = mem.namespace ? ` [${mem.namespace}]` : "";
      const line = `- ${text.slice(0, 220)}${namespace}`;
      sharedChars += line.length;
      if (sharedChars > MAX_CROSSAGENT_CHARS) break;
      sharedLines.push(line);
    }
    if (sharedLines.length > 1) {
      sections.push(...sharedLines);
    }
  }

  const allowCrossWorkspaceRecall =
    ENABLE_CROSS_AGENT_RECALL || policy.crossWorkspaceRecall === true;
  if (
    allowCrossWorkspaceRecall &&
    _memoryApiKey &&
    agentId !== "main" &&
    (!memories || memories.length < 5)
  ) {
    try {
      const mainRes = await fetch(`${MEMORY_API}/recall`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
        body: JSON.stringify({
          query: `recent decisions and context for ${agentId} tasks`,
          limit: 5,
          agent: "main",
        }),
        signal: AbortSignal.timeout(3000),
      });
      if (mainRes.ok) {
        const mainData = await mainRes.json();
        const mainMemories = mainData.results || [];
        if (mainMemories.length > 0) {
          console.warn(`[bootstrap-context] cross-agent: ${mainMemories.length} from main for ${agentId}`);
          const crossLines = ["\n# Main Agent Context\n"];
          let crossChars = 0;
          for (const mem of mainMemories.slice(0, 3)) {
            const t = mem.text || "";
            if (isLowSignalMemory(t)) continue;
            const line = `- ${t.slice(0, 200)} [from main]`;
            crossChars += line.length;
            if (crossChars > MAX_CROSSAGENT_CHARS) break;
            crossLines.push(line);
          }
          sections.push(...crossLines);
        }
      }
    } catch (e) {
      console.warn("[bootstrap-context] error:", e.message || e);
    }
  }

  const decisions = await fetchDecisionMemories(agentId);
  if (decisions && decisions.length > 0) {
    const existingTexts = new Set((memories || []).map((m) => (m.text || "").slice(0, 100)));
    const newDecisions = decisions.filter((d) => !existingTexts.has((d.text || "").slice(0, 100)));
    if (newDecisions.length > 0) {
      console.warn(`[bootstrap-context] fetched ${newDecisions.length} unique decisions for ${agentId}`);
      const decLines = ["\n# Recent Decisions & Changes\n"];
      for (const dec of newDecisions.slice(0, 5)) {
        const text = dec.text || dec.content || "";
        if (isLowSignalMemory(text)) continue;
        decLines.push(`- ${text.slice(0, 250)}`);
      }
      sections.push(...decLines);
    }
  }

  // Recent config/credential changes — remind agent to check memory before asking user
  if (_memoryApiKey) {
    try {
      const configRecall = await recall(agentId, "config change credential update .env modified propagation", {
        limit: 5,
        minScore: 0.3,
        timeoutMs: 3000,
      });
      const configChanges = (configRecall || []).filter(
        (m) => /config.change|credential|\.env|güncellen|update|propagat/i.test(m.text || "")
      );
      if (configChanges.length > 0) {
        const ccLines = ["\n# Recent Config/Credential Changes (CHECK BEFORE ASKING USER)\n"];
        ccLines.push("> ⚠️ If a config issue arises, check these memories FIRST before asking the user.\n");
        for (const cc of configChanges.slice(0, 5)) {
          const text = (cc.text || "").slice(0, 300);
          ccLines.push(`- ${text}`);
        }
        sections.push(...ccLines);
        console.warn(`[bootstrap-context] injected ${configChanges.length} config change reminders for ${agentId}`);
      }
    } catch (e) {
      console.warn("[bootstrap-context] config recall error:", e.message || e);
    }
  }

  const lastSession = await readLastSessionSummary(workspaceDir);
  if (lastSession) {
    console.warn(`[bootstrap-context] read last-session.md (${lastSession.length} chars)`);
    sections.push("\n# Last Session Summary\n");
    sections.push(lastSession);
  }

  if (policy.dailyNotesEnabled !== false) {
    const dailyNotes = await readDailyNotes(memoryDir);
    if (dailyNotes) {
      const trimmedNotes = truncateSection(dailyNotes, MAX_DAILY_CHARS);
      console.warn(
        `[bootstrap-context] read daily notes (${dailyNotes.length} -> ${trimmedNotes.length} chars)`
      );
      sections.push("\n# Recent Daily Notes\n");
      sections.push(trimmedNotes);
    }
  }

  if (sections.length === 0) {
    console.warn("[bootstrap-context] no content to inject");
    return;
  }

  const content = sections.join("\n");
  context.bootstrapFiles.push({
    name: "SESSION_CONTEXT",
    path: "SESSION_CONTEXT",
    content,
  });
  console.warn(`[bootstrap-context] injected SESSION_CONTEXT (${content.length} chars, agent=${agentId})`);
};

export default bootstrapContextHook;
