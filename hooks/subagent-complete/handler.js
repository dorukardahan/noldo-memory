/**
 * subagent-complete hook — Log and track sub-agent completions.
 *
 * Consumes the subagent:complete internal hook event (PR #20268).
 * Logs completions, tracks runtime, and captures errors to memory.
 *
 * Created: 2026-02-19
 */

import { appendFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const LOG_DIR = process.env.SUBAGENT_LOG_DIR || "./logs";
const LOG_FILE = path.join(LOG_DIR, "subagent-completions.log");
const MEMORY_API = "http://localhost:8787/v1";

import { readFileSync } from "node:fs";
const API_KEY_PATH = process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try { _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim(); } catch (e) { console.warn("[subagent-complete] error:", e.message || e); }

try { mkdirSync(LOG_DIR, { recursive: true }); } catch (e) { console.warn("[subagent-complete] error:", e.message || e); }

function ts() {
  return new Date().toISOString().replace("T", " ").slice(0, 19);
}

function getAgentFromSession(childSessionKey) {
  // agent:main:subagent:bureau -> bureau
  const parts = (childSessionKey || "").split(":");
  return parts[parts.length - 1] || "unknown";
}

const subagentCompleteHook = async (event) => {
  if (event.type !== "subagent" || event.action !== "complete") return;

  const ctx = event.context || {};
  const {
    runId,
    childSessionKey,
    label,
    task,
    outcome,
    startedAt,
    endedAt,
    runtimeMs,
  } = ctx;

  const agent = label || getAgentFromSession(childSessionKey);
  const status = outcome?.status || "unknown";
  const duration = runtimeMs ? `${(runtimeMs / 1000).toFixed(1)}s` : "?";
  const error = outcome?.error || "";

  // Log to file
  const logLine = `[${ts()}] ${status === "ok" ? "OK" : "ERR"} agent=${agent} runId=${runId} duration=${duration}${error ? ` error="${error}"` : ""} task="${(task || "").slice(0, 100)}"\n`;

  try {
    appendFileSync(LOG_FILE, logLine);
  } catch (err) {
    console.warn(`[subagent-complete] log write failed: ${err.message}`);
  }

  console.warn(`[subagent-complete] ${agent} → ${status} (${duration})`);

  // On error, capture to memory for post-mortem
  if (status === "error" && error) {
    try {
      await fetch(`${MEMORY_API}/capture`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": _memoryApiKey,
        },
        body: JSON.stringify({
          messages: [
            {
              role: "system",
              text: `Sub-agent ${agent} failed: ${error}. Task: ${(task || "").slice(0, 500)}. Runtime: ${duration}. RunId: ${runId}`,
            },
          ],
          agent: getAgentFromSession(childSessionKey),
        }),
        signal: AbortSignal.timeout(5000),
      });
    } catch (e) {
      console.warn("[subagent-complete] error:", e.message || e);
      // Non-critical — don't block on memory capture failure
    }
  }
};

export default subagentCompleteHook;
