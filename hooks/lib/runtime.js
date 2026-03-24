import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

const DEFAULT_POLICY = Object.freeze({
  crossWorkspaceRecall: false,
  sharedNamespaces: [],
  dailyNotesEnabled: true,
});

/**
 * Resolve sessionKey from event — unified across all hooks (H-9 fix).
 * Checks event.sessionKey, event.context.sessionKey, and ctx.sessionKey.
 */
export function resolveSessionKey(event = {}, ctx = {}) {
  const key = String(
    event?.sessionKey || event?.context?.sessionKey || ctx?.sessionKey || ""
  ).trim();
  return key || "";
}

export function resolveWorkspaceDir(event = {}) {
  const context = event?.context || {};
  return (
    context.workspaceDir ||
    context.cfg?.workspace?.dir ||
    process.env.OPENCLAW_WORKSPACE ||
    `${process.env.HOME}/.openclaw/workspace`
  );
}

export function resolveAgentId(event = {}, workspaceDir = "") {
  const fromContext = String(event?.context?.agentId || "")
    .trim()
    .toLowerCase();
  if (fromContext) return fromContext;

  const sessionKey = String(event?.sessionKey || "")
    .trim()
    .toLowerCase();
  const parts = sessionKey.split(":").filter(Boolean);
  if (parts.length >= 3 && parts[0] === "agent" && parts[1]) {
    return parts[1];
  }

  const base = path.basename(workspaceDir || "");
  if (base === "workspace") return "main";
  if (base.startsWith("workspace-")) return base.replace("workspace-", "");
  return "main";
}

export function deriveSessionNamespace(sessionKey = "") {
  const raw = String(sessionKey || "").trim().toLowerCase();
  if (!raw) return "default";
  const hash = crypto.createHash("sha1").update(raw).digest("hex").slice(0, 16);
  return `session-${hash}`;
}

/**
 * Strip OpenClaw channel envelope prefixes from message content.
 * OpenClaw wraps inbound messages with metadata headers:
 *   "System: [2026-03-10 14:30:00 GMT+3] Slack message in #channel from User: content"
 * This function extracts only the user's actual content.
 */
export function stripChannelEnvelope(text = "") {
  let cleaned = String(text || "");
  // Slack envelope: System: [timestamp] Slack message [edited] in #channel [from User]: content
  cleaned = cleaned.replace(
    /^System:\s*\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s+GMT[+-]\d+\]\s*Slack message(?:\s+edited)?\s+in\s+#\S+(?:\s+from\s+[^:]+)?[.:]\s*/i,
    ""
  );
  // OpenClaw runtime context preamble (cron/subagent delivered messages)
  cleaned = cleaned.replace(
    /^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+GMT[+-]\d+\]\s*OpenClaw runtime context \(internal\):[\s\S]*?(?=\n\n|$)/gi,
    ""
  );
  return cleaned;
}

export async function readWorkspacePolicy(workspaceDir = "") {
  const policyPath = path.join(workspaceDir, ".openclaw", "noldo-memory.json");
  try {
    const raw = await fs.readFile(policyPath, "utf-8");
    const parsed = JSON.parse(raw);
    const sharedNamespaces = Array.isArray(parsed?.sharedNamespaces)
      ? parsed.sharedNamespaces
          .map((value) => (typeof value === "string" ? value.trim() : ""))
          .filter(Boolean)
      : [];

    return {
      crossWorkspaceRecall: parsed?.crossWorkspaceRecall === true,
      sharedNamespaces: Array.from(new Set(sharedNamespaces)),
      dailyNotesEnabled:
        typeof parsed?.dailyNotesEnabled === "boolean"
          ? parsed.dailyNotesEnabled
          : DEFAULT_POLICY.dailyNotesEnabled,
    };
  } catch (err) {
    if (err?.code !== "ENOENT") {
      console.warn(`[noldo-policy] failed to read workspace policy: ${err.message || err}`);
    }
    return { ...DEFAULT_POLICY };
  }
}
