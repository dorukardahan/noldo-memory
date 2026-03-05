/**
 * after-tool-call hook — Capture important tool outputs to memory.
 *
 * Fires on every tool call completion. Only captures exec commands that
 * match operationally important patterns (deploy, git, config, errors).
 * Skips routine read-only commands to avoid noise.
 *
 * Created: 2026-02-17 [S6 — Memory Improvement Plan]
 */

import path from "node:path";

const MEMORY_API = "http://localhost:8787/v1";
import { readFileSync } from "node:fs";
const API_KEY_PATH = process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try { _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim(); } catch (e) { console.warn("[after-tool-call] error:", e.message || e); }


// Command patterns that indicate important operations
const IMPORTANT_CMD_PATTERNS = [
  /systemctl\s+(restart|stop|start|enable|disable)/i,
  /docker\s+(compose|build|push|pull|up|down|restart|stop|start)/i,
  /git\s+(push|merge|tag|commit|checkout\s+-b)/i,
  /apt(?:-get)?\s+(install|upgrade|remove|purge)/i,
  /npm\s+(publish|install|update|uninstall)/i,
  /pip3?\s+install/i,
  /curl\s+.*-X\s*(POST|PUT|DELETE|PATCH)/i,
  /ufw\s+(allow|deny|enable|disable|delete)/i,
  /iptables\s+-[AID]/i,
  /certbot/i,
  /(?:cp|mv|rm)\s+.*(?:\.env|\.conf|\.service|\.json|\.yaml|\.yml)/i,
  /sed\s+-i/i,
  /crontab\s+-[re]/i,
  /chmod|chown/i,
  /mkswap|swapon|swapoff/i,
  /resize2fs|growpart/i,
];

const CAPTURE_COMMANDS = [
  /^(git\s|gh\s|npm\s|pip\s|curl\s|ls\s|cat\s|grep\s|find\s|wc\s)/,
  /^(python3?\s-m\spytest)/,
  /^(ruff\s|black\s|mypy\s)/,
];

function getAgentId(workspaceDir) {
  const base = path.basename(workspaceDir || "");
  if (base === "workspace") return "main";
  if (base.startsWith("workspace-")) return base.replace("workspace-", "");
  return "main";
}

function shouldCapture(toolName, toolInput) {
  if (toolName !== "exec") return false;

  const cmd = (toolInput?.command || "").trim();
  if (!cmd || cmd.length < 3) return false;

  return CAPTURE_COMMANDS.some((p) => p.test(cmd));
}

function scoreToolOutput(toolInput, toolOutput) {
  const cmd = (toolInput?.command || "").toLowerCase();
  let importance = 0.50;

  // Service operations
  if (/systemctl|service\s|docker/.test(cmd)) importance = 0.80;
  // Deploy/release
  if (/deploy|push|publish|release/.test(cmd)) importance = 0.90;
  // Config changes
  if (/\.env|\.conf|\.service|config/.test(cmd)) importance = 0.75;
  // Git operations
  if (/git\s+(push|merge|tag)/.test(cmd)) importance = 0.80;
  // Security operations
  if (/ufw|iptables|certbot|chmod|chown/.test(cmd)) importance = 0.80;
  // Error in output
  if (/error|fail|denied|crash|killed|refused/i.test(toolOutput || ""))
    importance = Math.max(importance, 0.85);

  return importance;
}

const afterToolCallHook = async (event, ctx) => {
  // Event shape varies — handle both possible structures
  const toolName = event.toolName;
  const toolInput = event.params || {};
  const toolResult = event.result || event.error || "";
  const toolOutput =
    typeof toolResult === "string" ? toolResult : JSON.stringify(toolResult);

  if (!toolName) return;
  if (!shouldCapture(toolName, toolInput)) return;

  const workspaceDir = ctx?.workspaceDir || process.env.OPENCLAW_WORKSPACE || `${process.env.HOME}/.openclaw/workspace`;
  const agentId = getAgentId(workspaceDir);

  const importance = scoreToolOutput(toolInput, toolOutput);
  const cmd = (toolInput.command || toolName).slice(0, 200);

  // Truncate output intelligently — keep first and last portions
  let output = toolOutput;
  if (output.length > 2000) {
    const head = output.slice(0, 1200);
    const tail = output.slice(-600);
    output = `${head}\n...[truncated ${output.length - 1800} chars]...\n${tail}`;
  }

  const memoryText = `[Tool: ${toolName}] Command: ${cmd}\nOutput: ${output}`;

  try {
    const res = await fetch(`${MEMORY_API}/store`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
      body: JSON.stringify({
        text: memoryText.slice(0, 3000),
        category: "tool_output",
        importance,
        agent: agentId,
      }),
      signal: AbortSignal.timeout(5000),
    });

    if (res.ok) {
      console.warn(
        `[after-tool-call] captured: ${cmd.slice(0, 80)} (imp=${importance}, agent=${agentId})`
      );
    } else {
      console.warn(
        `[after-tool-call] API error ${res.status}: ${cmd.slice(0, 60)}`
      );
    }
  } catch (err) {
    console.warn(`[after-tool-call] capture failed: ${err.message}`);
  }
};

export default afterToolCallHook;
