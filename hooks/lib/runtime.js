import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

const DEFAULT_POLICY = Object.freeze({
  crossWorkspaceRecall: false,
  sharedNamespaces: [],
  dailyNotesEnabled: true,
});

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
