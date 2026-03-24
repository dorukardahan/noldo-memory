/**
 * session-end-capture hook - Capture session content on explicit session resets.
 */

import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { atomicWrite } from "../lib/util.js";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH =
  process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try {
  _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim();
} catch (e) {
  console.warn("[session-end-capture] error:", e.message || e);
}

const OPENCLAW_DIR = process.env.OPENCLAW_DIR || `${process.env.HOME}/.openclaw`;

const SUGGESTION_PATTERNS = [/önerim/i, /ekleyelim/i, /\bsuggest\b/i, /\brecommend\b/i];
const VERIFICATION_TOOL_PATTERNS = [/\bexec\b/i, /\bread\b/i, /\bweb_fetch\b/i];
const LOW_SIGNAL_PATTERNS = [
  /A new session was started via \/new or \/reset/i,
  /Conversation info \(untrusted metadata\)/i,
  /\[Subagent Context\]/i,
  /^User:\s*Conversation info/i,
  /\[System Message\]/i,
  /\[Queued announce messages while agent was busy\]/i,
  /\bA cron job\b/i,
  /\bA subagent task\b/i,
  /^System:\s*\[System Message\]/i,
  /^System:\s*\[(?:Queued|Internal|Subagent)\b/i,
];

function isSuggestionText(text = "") {
  return SUGGESTION_PATTERNS.some((p) => p.test(text));
}

function hasVerificationSignal(entry = {}) {
  const haystack = JSON.stringify(entry || {});
  return VERIFICATION_TOOL_PATTERNS.some((p) => p.test(haystack));
}

function normalizeMessageText(raw = "") {
  let text = String(raw || "").replace(/\r\n/g, "\n");
  // Strip OpenClaw Slack envelope prefix
  text = text.replace(
    /^System:\s*\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s+GMT[+-]\d+\]\s*Slack message(?:\s+edited)?\s+in\s+#\S+(?:\s+from\s+[^:]+)?[.:]\s*/i,
    ""
  );
  // Strip OpenClaw runtime context preamble
  text = text.replace(
    /^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+GMT[+-]\d+\]\s*OpenClaw runtime context \(internal\):[\s\S]*?(?=\n\n|$)/gi,
    ""
  );
  text = text.replace(/Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```/gi, "");
  text = text.replace(/\[Subagent Context\][\s\S]*?(?=\n\n|$)/gi, "");
  text = text.replace(/^\s*\[[^\]]+\]\s*\[System Message\].*$/gim, "");
  text = text.replace(/\n{3,}/g, "\n\n").trim();
  return text;
}

function isLowSignalText(text = "") {
  if (!text) return true;
  return LOW_SIGNAL_PATTERNS.some((p) => p.test(text));
}

function parseAgentIdFromSessionKey(sessionKey = "") {
  const raw = String(sessionKey || "").trim().toLowerCase();
  const parts = raw.split(":").filter(Boolean);
  if (parts.length >= 3 && parts[0] === "agent" && parts[1]) {
    return parts[1];
  }
  return null;
}

function parseAgentIdFromWorkspaceDir(workspaceDir = "") {
  const base = path.basename(workspaceDir || "");
  if (base === "workspace") return "main";
  if (base.startsWith("workspace-")) return base.replace("workspace-", "");
  return null;
}

function deriveSessionNamespace(sessionKey = "") {
  const raw = String(sessionKey || "").trim().toLowerCase();
  if (!raw) return "default";
  const hash = crypto.createHash("sha1").update(raw).digest("hex").slice(0, 16);
  return `session-${hash}`;
}

function deriveAgentId(event = {}) {
  const fromContext = String(event?.context?.agentId || "")
    .trim()
    .toLowerCase();
  if (fromContext) return fromContext;
  const fromSessionKey = parseAgentIdFromSessionKey(event?.sessionKey);
  if (fromSessionKey) return fromSessionKey;
  const fromWorkspace = parseAgentIdFromWorkspaceDir(event?.context?.workspaceDir || "");
  if (fromWorkspace) return fromWorkspace;
  return "main";
}

function resolveWorkspaceDir(event = {}, agentId = "main") {
  const workspaceFromContext = event?.context?.workspaceDir;
  if (typeof workspaceFromContext === "string" && workspaceFromContext.trim()) {
    return workspaceFromContext.trim();
  }
  if (agentId === "main") {
    return process.env.OPENCLAW_WORKSPACE || `${process.env.HOME}/.openclaw/workspace`;
  }
  return `${process.env.HOME}/.openclaw/workspace-${agentId}`;
}

function getSessionIdCandidates(raw) {
  if (!raw) return [];
  const value = String(raw);
  const parts = value.split(":").filter(Boolean);
  const last = parts.length > 0 ? parts[parts.length - 1] : value;
  return Array.from(new Set([value, last].filter(Boolean)));
}

function findSessionFile(sessionRef, agentId = "main") {
  const candidates = getSessionIdCandidates(sessionRef);
  if (candidates.length === 0) return null;

  const searchDirs = [
    path.join(OPENCLAW_DIR, "agents", agentId, "sessions"),
    path.join(OPENCLAW_DIR, "sessions"),
  ];

  for (const sessionsDir of searchDirs) {
    try {
      const files = readdirSync(sessionsDir);
      for (const f of files) {
        if (!f.endsWith(".jsonl")) continue;
        if (candidates.some((candidate) => f.includes(candidate))) {
          return path.join(sessionsDir, f);
        }
      }
    } catch {
      // non-critical
    }
  }
  return null;
}

async function pickFirstExisting(paths) {
  for (const candidate of paths) {
    if (!candidate) continue;
    try {
      await fs.access(candidate);
      return candidate;
    } catch {
      // continue
    }
  }
  return null;
}

async function resolveSessionFileFromEvent(event, agentId) {
  const context = event?.context || {};
  const previousSessionEntry = context?.previousSessionEntry || {};
  const sessionEntry = context?.sessionEntry || {};

  const directCandidates = [previousSessionEntry?.sessionFile, sessionEntry?.sessionFile].filter(
    (value) => typeof value === "string" && value.trim()
  );

  const existingDirect = await pickFirstExisting(directCandidates);
  if (existingDirect) return existingDirect;

  const refs = [previousSessionEntry?.sessionId, sessionEntry?.sessionId, event?.sessionKey];
  for (const ref of refs) {
    const found = findSessionFile(ref, agentId);
    if (found) return found;
  }

  return null;
}

async function detectUnverifiedSuggestion(sessionFilePath) {
  try {
    const raw = await fs.readFile(sessionFilePath, "utf-8");
    const lines = raw.trim().split("\n").filter(Boolean);
    const entries = lines
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);

    let hasPriorVerification = false;
    for (const entry of entries) {
      if (hasVerificationSignal(entry)) {
        hasPriorVerification = true;
        continue;
      }

      if (entry?.type !== "message") continue;
      const role = entry?.message?.role;
      if (role !== "assistant") continue;

      const content = entry?.message?.content;
      const text = Array.isArray(content) ? content.find((c) => c.type === "text")?.text : content;
      const normalized = normalizeMessageText(text || "");
      if (!normalized || isLowSignalText(normalized)) continue;

      if (isSuggestionText(normalized) && !hasPriorVerification) {
        return true;
      }
    }
  } catch (e) {
    console.warn("[session-end-capture] error:", e.message || e);
  }
  return false;
}

async function getSessionMessages(sessionFilePath, count = 15) {
  try {
    const raw = await fs.readFile(sessionFilePath, "utf-8");
    const lines = raw.trim().split("\n");
    const messages = [];

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry.type !== "message") continue;
        const msg = entry.message;
        if (!msg || !msg.content) continue;

        const role = msg.role;
        if (role !== "user" && role !== "assistant") continue;

        const text = Array.isArray(msg.content) ? msg.content.find((c) => c.type === "text")?.text : msg.content;

        const normalized = normalizeMessageText(text || "");
        if (!normalized || normalized.startsWith("/") || isLowSignalText(normalized)) continue;
        if (normalized.length <= 10) continue;

        messages.push({ role, text: normalized.slice(0, 1000) });
      } catch {
        // skip malformed lines
      }
    }

    return messages.slice(-count);
  } catch (e) {
    console.warn("[session-end-capture] error:", e.message || e);
    return [];
  }
}

function buildQAPairs(messages) {
  const pairs = [];
  for (let i = 0; i < messages.length - 1; i++) {
    if (messages[i].role === "user" && messages[i + 1].role === "assistant") {
      pairs.push({
        role: "qa_pair",
        text: `User: ${messages[i].text}\nAssistant: ${messages[i + 1].text}`,
      });
      i++;
    } else {
      pairs.push(messages[i]);
    }
  }
  const lastMsg = messages[messages.length - 1];
  if (lastMsg && (pairs.length === 0 || pairs[pairs.length - 1].text !== lastMsg.text)) {
    pairs.push(lastMsg);
  }
  return pairs;
}

async function checkPatternEscalation(agent, tag, apiKey) {
  try {
    const resp = await fetch(`${MEMORY_API}/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
      body: JSON.stringify({ query: tag, agent, limit: 10, memory_type: "lesson" }),
      signal: AbortSignal.timeout(10000),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    const sameTag = (data.results || []).filter((r) => {
      const text = String(r?.text || r?.content || "");
      if (r?.category === tag) return true;
      if (Array.isArray(r?.tags) && r.tags.includes(tag)) return true;
      return text.includes(`[tag=${tag}]`);
    });

    if (sameTag.length >= 3) {
      const ruleText =
        `AUTO-RULE (${sameTag.length} occurrences): Pattern "${tag}" detected repeatedly. ` +
        `Original lessons: ${sameTag
          .slice(0, 3)
          .map((r) => (r.text || "").substring(0, 80))
          .join(" | ")}`;

      await fetch(`${MEMORY_API}/store`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
        body: JSON.stringify({
          text: ruleText,
          category: "rule",
          importance: 0.98,
          agent,
          source: "auto_escalation",
        }),
        signal: AbortSignal.timeout(10000),
      });
      console.warn(`[session-end-capture] auto-escalated pattern "${tag}" (${sameTag.length} lessons)`);
    }
  } catch (e) {
    console.warn("[session-end-capture] pattern escalation error:", e.message || e);
  }
}

// ── MAST P0: Session Handoff & ATS helpers ──

const MOOD_FRUSTRATED = [
  /sinirlendim/i, /kızdığım/i, /aptal/i, /neden yapmadın/i, /neden yapmıyorsun/i,
  /tekrar mı/i, /yine mi/i, /frustrated/i, /annoyed/i, /useless/i,
];
const MOOD_POSITIVE = [
  /tamam güzel/i, /harika/i, /süper/i, /mükemmel/i, /great/i, /perfect/i, /awesome/i, /güzel olmuş/i,
];

function detectUserMood(messages) {
  const userMsgs = messages.filter((m) => m.role === "user");
  const last3 = userMsgs.slice(-3);
  const combined = last3.map((m) => m.text).join(" ");

  if (MOOD_FRUSTRATED.some((p) => p.test(combined))) return "frustrated";
  if (MOOD_POSITIVE.some((p) => p.test(combined))) return "positive";
  return "neutral";
}

function extractUnfinishedWork(messages) {
  const items = [];
  // Look for TODO-like patterns in assistant messages
  const assistantMsgs = messages.filter((m) => m.role === "assistant");
  for (const msg of assistantMsgs.slice(-5)) {
    const todoMatches = msg.text.match(/(?:- \[ \]|TODO|bekliyor|kalan|unfinished|remaining)[^\n]{5,80}/gi);
    if (todoMatches) items.push(...todoMatches.map((m) => m.trim().slice(0, 150)));
  }
  return items.slice(0, 5);
}

function extractFilesModified(messages) {
  // NOTE: This extracts files *mentioned* in conversation, not necessarily modified.
  // Heuristic only — used for handoff context, not for verification.
  const files = new Set();
  for (const msg of messages) {
    const matches = msg.text.match(/(?:\/(?:opt|root|home|etc|var|tmp)\/[^\s\])"',]{5,80})/g);
    if (matches) {
      for (const m of matches) {
        if (m.match(/\.\w{1,5}$/)) files.add(m); // only files with extensions
      }
    }
  }
  return [...files].slice(0, 10);
}

const sessionEndCaptureHook = async (event) => {
  if (event?.type !== "command" || (event?.action !== "new" && event?.action !== "reset")) {
    return;
  }

  const agentId = deriveAgentId(event || {});
  const sessionNamespace = deriveSessionNamespace(event?.sessionKey);
  const sessionFile = await resolveSessionFileFromEvent(event, agentId);
  if (!sessionFile) {
    console.warn(
      `[session-end-capture] no session file found (sessionKey=${event?.sessionKey || "unknown"} agent=${agentId})`
    );
    return;
  }

  const sessionRef =
    event?.context?.previousSessionEntry?.sessionId ||
    event?.context?.sessionEntry?.sessionId ||
    event?.sessionKey ||
    "unknown";
  console.warn(`[session-end-capture] fired session=${sessionRef} agent=${agentId}`);

  const messages = await getSessionMessages(sessionFile, 15);
  if (messages.length === 0) {
    console.warn(`[session-end-capture] no usable messages for ${sessionRef}`);
    return;
  }

  const pairs = buildQAPairs(messages);

  if (_memoryApiKey) {
    try {
      const captureRes = await fetch(`${MEMORY_API}/capture`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
        body: JSON.stringify({
          messages: pairs.map((p) => ({ role: p.role || "assistant", text: p.text })),
          agent: agentId,
          namespace: sessionNamespace,
        }),
        signal: AbortSignal.timeout(30000),
      });

      if (!captureRes.ok) {
        console.warn(`[session-end-capture] capture failed: status=${captureRes.status}`);
      } else {
        const payload = await captureRes.json().catch(() => ({}));
        console.warn(
          `[session-end-capture] capture ok: stored=${payload?.stored ?? "?"} merged=${payload?.merged ?? "?"} total=${payload?.total ?? "?"} namespace=${payload?.namespace ?? sessionNamespace}`
        );
      }
    } catch (err) {
      console.warn(`[session-end-capture] capture failed: ${err.message}`);
    }
  }

  if (_memoryApiKey) {
    try {
      const shouldStoreAutoLesson = await detectUnverifiedSuggestion(sessionFile);
      if (shouldStoreAutoLesson) {
        const res = await fetch(`${MEMORY_API}/store`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
          body: JSON.stringify({
            text: "[Auto-Lesson][tag=verification] Suggested without verifiable check before recommendation.",
            category: "lesson",
            importance: 0.9,
            agent: agentId,
            source: "session_capture",
          }),
          signal: AbortSignal.timeout(10000),
        });
        if (!res.ok) {
          console.warn(`[session-end-capture] auto-lesson store failed: status=${res.status}`);
        } else {
          await checkPatternEscalation(agentId, "verification", _memoryApiKey);
        }
      }
    } catch (err) {
      console.warn(`[session-end-capture] auto-lesson failed: ${err.message}`);
    }
  }

  try {
    const workspaceDir = resolveWorkspaceDir(event, agentId);
    const memoryDir = path.join(workspaceDir, "memory");
    await fs.mkdir(memoryDir, { recursive: true });

    const lines = [
      "# Last Session Summary",
      "",
      `- **Session ID:** ${sessionRef}`,
      `- **Agent:** ${agentId}`,
      `- **Captured:** ${new Date().toISOString()}`,
      "",
      "## Conversation",
      "",
    ];
    for (const m of messages.slice(-10)) {
      lines.push(`**${m.role}:** ${m.text.slice(0, 300)}`);
      lines.push("");
    }
    await fs.writeFile(path.join(memoryDir, "last-session.md"), lines.join("\n"), "utf-8");

    const snapshot = {
      timestamp: new Date().toISOString(),
      sessionKey: String(event?.sessionKey || sessionRef),
      recentMessages: messages.slice(-8).map((m) => ({ role: m.role, text: m.text.slice(0, 500) })),
    };
    atomicWrite(
      path.join(memoryDir, "critical-context-snapshot.json"),
      JSON.stringify(snapshot, null, 2)
    );

    // ── MAST P0: Session Handoff ──
    try {
      const last3 = messages.slice(-3).map((m) => `${m.role}: ${m.text.slice(0, 150)}`).join(" | ");
      const handoff = {
        timestamp: new Date().toISOString(),
        session_key: String(event?.sessionKey || sessionRef),
        agent: agentId,
        summary: last3.slice(0, 500),
        unfinished_work: extractUnfinishedWork(messages),
        next_steps: [], // populated by active task context if available
        files_modified: extractFilesModified(messages),
        user_mood: detectUserMood(messages),
      };
      atomicWrite(
        path.join(memoryDir, "session-handoff.json"),
        JSON.stringify(handoff, null, 2)
      );
      console.warn(`[session-end-capture] handoff written (mood=${handoff.user_mood}, files=${handoff.files_modified.length})`);
    } catch (e) {
      console.warn(`[session-end-capture] handoff write error: ${e.message}`);
    }

    // ── MAST P0: Update Active Tasks with session context ──
    try {
      // Use readATS/writeATS for atomic operations (Fix: race condition from review)
      const { readATS: readATSSync, writeATS: writeATSSync } = await import("../lib/ats.js");
      const ats = readATSSync(workspaceDir);
      let modified = false;
      for (const task of ats.tasks) {
        if (task.status === "in_progress") {
          task.context.last_session_summary = messages.slice(-3).map((m) => m.text.slice(0, 200)).join(" | ").slice(0, 600);
          task.updated_at = new Date().toISOString();
          modified = true;
        }
      }
      if (modified) {
        writeATSSync(workspaceDir, ats);
        console.warn(`[session-end-capture] ATS tasks updated with session context`);
      }
    } catch (e) {
      if (e?.code !== "ENOENT") console.warn(`[session-end-capture] ATS update error: ${e.message}`);
    }

  } catch (err) {
    console.warn(`[session-end-capture] write failed: ${err.message}`);
  }
};

export default sessionEndCaptureHook;
