/**
 * before-compaction hook — Save critical context to NoldoMem before compaction.
 *
 * OpenClaw fires this just before summarizing/compacting the conversation.
 * We capture the most important recent messages so they survive compaction.
 *
 * Event shape (from OpenClaw source):
 *   { messageCount: number, messages: array, sessionFile: string }
 * Context shape:
 *   { sessionKey: string }
 *
 * Created: 2026-04-02 [Memory Audit Faz 1]
 */

import { readFileSync } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveAgentId, resolveWorkspaceDir } from "../lib/runtime.js";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH =
  process.env.AGENT_MEMORY_API_KEY_FILE ||
  `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try {
  _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim();
} catch (e) {
  console.warn("[before-compaction] API key read error:", e.message || e);
}

// How many recent messages to extract
const MAX_MESSAGES = 30;
// Max chars per message to store
const MAX_MSG_CHARS = 2000;
// Min message length to consider
const MIN_MSG_LENGTH = 15;

// Low-signal patterns to skip
const LOW_SIGNAL_PATTERNS = [
  /A new session was started via \/new or \/reset/i,
  /Conversation info \(untrusted metadata\)/i,
  /\[Subagent Context\]/i,
  /\[System Message\]/i,
  /\[Queued announce messages while agent was busy\]/i,
  /^\[cron:/i,
  /steward-engage/i,
  /HEARTBEAT_OK/i,
  /^System:\s*\[(?:Queued|Internal|Subagent)\b/i,
  /A cron job/i,
  /A subagent task/i,
];

// High-importance markers
const DECISION_MARKERS = [
  /\bkarar\b/i, /yapalım/i, /yapacağız/i, /anlaştık/i,
  /\bdecided\b/i, /\blet'?s do\b/i, /\bagree\b/i,
  /onaylıyorum/i, /\bapproved?\b/i,
];

const ERROR_MARKERS = [
  /\bhata\b/i, /\bbug\b/i, /\bfix\b/i, /\bsorun\b/i,
  /\berror\b/i, /\bfailed?\b/i, /\bcrash\b/i,
];

const PREFERENCE_MARKERS = [
  /\bterci[hk]\b/i, /\bprefer\b/i, /\bher zaman\b/i,
  /\basla\b/i, /\balways\b/i, /\bnever\b/i,
];

function isLowSignal(text) {
  return LOW_SIGNAL_PATTERNS.some((p) => p.test(text));
}

function scoreMessage(text) {
  if (!text || text.length < MIN_MSG_LENGTH) return 0;
  if (isLowSignal(text)) return 0;

  let score = 0.4;
  if (DECISION_MARKERS.some((p) => p.test(text))) score = Math.max(score, 0.9);
  if (ERROR_MARKERS.some((p) => p.test(text))) score += 0.15;
  if (PREFERENCE_MARKERS.some((p) => p.test(text))) score += 0.2;
  if (text.length > 100) score += 0.05;
  if (text.length > 300) score += 0.05;
  return Math.min(1.0, score);
}

function extractTextFromContent(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const textBlock = content.find((c) => c?.type === "text");
    return textBlock?.text || "";
  }
  return "";
}

function cleanText(raw) {
  let text = String(raw || "");
  // Strip OpenClaw Slack envelope
  text = text.replace(
    /^System:\s*\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s+GMT[+-]\d+\]\s*Slack message(?:\s+edited)?\s+in\s+#\S+(?:\s+from\s+[^:]+)?[.:]\s*/i,
    ""
  );
  // Strip metadata blocks
  text = text.replace(
    /Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```/gi,
    ""
  );
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

async function captureToNoldoMem(messages, agentId) {
  if (!_memoryApiKey || messages.length === 0) return;

  try {
    const res = await fetch(`${MEMORY_API}/capture`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": _memoryApiKey,
      },
      body: JSON.stringify({
        messages,
        agent: agentId,
        source: "before-compaction",
        namespace: "default",
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (!res.ok) {
      console.warn(
        `[before-compaction] capture failed: status=${res.status} agent=${agentId}`
      );
    } else {
      const result = await res.json().catch(() => ({}));
      console.warn(
        `[before-compaction] captured ${messages.length} messages for ${agentId} (stored: ${result.stored || "?"})`
      );
    }
  } catch (e) {
    console.warn("[before-compaction] capture error:", e.message || e);
  }
}

async function writeCompactionSnapshot(workspaceDir, topMessages, agentId) {
  if (!workspaceDir || topMessages.length === 0) return;

  const memoryDir = path.join(workspaceDir, "memory");
  try {
    await fs.mkdir(memoryDir, { recursive: true });
  } catch {}

  const now = new Date();
  const dateStr = now.toISOString().split("T")[0];
  const timeStr = now.toISOString().split("T")[1].split(".")[0];

  const lines = [
    `# Pre-Compaction Snapshot (${dateStr} ${timeStr} UTC)`,
    "",
    `Agent: ${agentId}`,
    `Messages captured: ${topMessages.length}`,
    "",
    "## Key Context",
    "",
  ];

  for (const msg of topMessages) {
    const prefix = msg.role === "user" ? "User" : "Assistant";
    const truncated =
      msg.content.length > 500
        ? msg.content.slice(0, 500) + "..."
        : msg.content;
    lines.push(`**${prefix}:** ${truncated}`);
    lines.push("");
  }

  const filePath = path.join(memoryDir, `${dateStr}.md`);

  try {
    // Append to existing daily notes, don't overwrite
    const existing = await fs.readFile(filePath, "utf-8").catch(() => "");
    const separator = existing ? "\n\n---\n\n" : "";
    await fs.writeFile(filePath, existing + separator + lines.join("\n"), "utf-8");
    console.warn(
      `[before-compaction] snapshot appended to ${filePath}`
    );
  } catch (e) {
    console.warn("[before-compaction] snapshot write error:", e.message || e);
  }
}

const beforeCompactionHook = async (event, context) => {
  const messages = event?.messages;
  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    console.warn("[before-compaction] no messages in event, skipping");
    return;
  }

  const agentId = resolveAgentId(event, context?.workspaceDir || "");
  const workspaceDir = resolveWorkspaceDir(event, context);

  console.warn(
    `[before-compaction] hook fired (agent=${agentId}, messageCount=${messages.length})`
  );

  // Extract recent user/assistant messages
  const extracted = [];
  const recentMessages = messages.slice(-MAX_MESSAGES * 2); // Take more, filter down

  for (const msg of recentMessages) {
    if (!msg || typeof msg !== "object") continue;

    const role = msg.role;
    if (role !== "user" && role !== "assistant") continue;

    const rawText = extractTextFromContent(msg.content);
    const text = cleanText(rawText);
    if (!text || text.length < MIN_MSG_LENGTH) continue;

    const score = scoreMessage(text);
    if (score <= 0) continue;

    extracted.push({
      role,
      content: text.slice(0, MAX_MSG_CHARS),
      score,
    });
  }

  // Sort by score descending, take top messages
  extracted.sort((a, b) => b.score - a.score);
  const topMessages = extracted.slice(0, MAX_MESSAGES);

  if (topMessages.length === 0) {
    console.warn("[before-compaction] no high-signal messages found");
    return;
  }

  console.warn(
    `[before-compaction] extracted ${topMessages.length} messages (top score: ${topMessages[0]?.score?.toFixed(2)})`
  );

  // Capture to NoldoMem API
  const captureMessages = topMessages.map((m) => ({
    role: m.role,
    content: m.content,
  }));

  await Promise.all([
    captureToNoldoMem(captureMessages, agentId),
    writeCompactionSnapshot(workspaceDir, topMessages, agentId),
  ]);
};

import { resilientHandler } from "../lib/resilient-import.js";
export default resilientHandler(beforeCompactionHook, "before-compaction");
