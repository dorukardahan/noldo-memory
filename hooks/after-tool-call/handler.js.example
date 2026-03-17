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

// Only capture mutating/meaningful dev commands, not read-only exploration
const CAPTURE_COMMANDS = [
  /^git\s+(?:push|commit|merge|tag|checkout\s+-b)/,
  /^gh\s+(?:pr\s+(?:create|merge|close)|issue\s+(?:create|close)|release)/,
  /^npm\s+(?:publish|install|update|uninstall)/,
  /^pip3?\s+install/,
  /^curl\s+.*-X\s*(?:POST|PUT|DELETE|PATCH)/i,
  /^python3?\s+-m\s+pytest/,
  /^(?:ruff|black|mypy)\s/,
];

function getAgentId(workspaceDir) {
  const base = path.basename(workspaceDir || "");
  if (base === "workspace") return "main";
  if (base.startsWith("workspace-")) return base.replace("workspace-", "");
  return "main";
}

// Config/credential file patterns — structured fact capture for edit/write tools
// Narrowed to known config basenames to avoid noise from package.json, test fixtures, etc.
const CONFIG_FILE_PATTERNS = [
  /\.env(?:\.\w+)?$/i,                    // .env, .env.local, .env.production
  /(?:^|\/)config\.\w+$/i,               // config.json, config.yaml, etc.
  /(?:^|\/)settings\.\w+$/i,             // settings.json, settings.yaml
  /(?:^|\/)docker-compose\.\w+$/i,       // docker-compose.yml
  /\.(?:conf|service|ini|cfg)$/i,        // nginx.conf, systemd .service, .ini, .cfg
  /credential/i,                          // anything with "credential" in path
  /secret/i,                              // anything with "secret" in path
];

function isConfigFile(filePath) {
  return CONFIG_FILE_PATTERNS.some((p) => p.test(filePath || ""));
}

function shouldCapture(toolName, toolInput) {
  // Capture edit/write to config/credential files
  if (toolName === "edit" || toolName === "write") {
    const filePath = toolInput?.file_path || toolInput?.path || "";
    return isConfigFile(filePath);
  }

  if (toolName !== "exec") return false;

  const cmd = (toolInput?.command || "").trim();
  if (!cmd || cmd.length < 3) return false;

  // Capture both standard dev commands AND important operational commands
  return CAPTURE_COMMANDS.some((p) => p.test(cmd)) ||
         IMPORTANT_CMD_PATTERNS.some((p) => p.test(cmd));
}

// Classify exec command type for memory_type preassignment
// Order matches Python classifier: incident > config > ops > deploy
// Only preassign for high-confidence cases; null = let server classifier decide
function classifyExecType(cmd) {
  // Config file mutations — high confidence
  if (/(?:sed\s+-i|cp|mv|rm|edit|write)\s.*(?:\.env|\.conf|\.service|config)/i.test(cmd)) return "config_change";
  // Service lifecycle — high confidence
  if (/systemctl\s+(?:restart|stop|start|enable|disable)/i.test(cmd)) return "operational_event";
  if (/docker\s+(?:compose\s+)?(?:up|down|restart|stop|start|build)/i.test(cmd)) return "operational_event";
  // Deploy actions — high confidence
  if (/git\s+push|npm\s+publish/i.test(cmd)) return "deployment";
  // Security ops — high confidence
  if (/ufw\s+(?:allow|deny|enable|disable)|iptables|certbot/i.test(cmd)) return "operational_event";
  return null; // let server-side classifier decide for ambiguous commands
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

  // Structured config change capture for edit/write tools
  if ((toolName === "edit" || toolName === "write") && isConfigFile(toolInput?.file_path || toolInput?.path || "")) {
    const filePath = toolInput?.file_path || toolInput?.path || "";
    const fileName = path.basename(filePath);
    // Store ONLY the fact that a config file changed — NEVER store content/secrets
    const configMemory = `[Config Change] File: ${filePath} modified via ${toolName}. Agent: ${agentId}. Check propagation: were all related config files updated?`;
    try {
      await fetch(`${MEMORY_API}/store`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
        body: JSON.stringify({
          text: configMemory,
          category: "config_change",
          importance: 0.85,
          agent: agentId,
          source: "after-tool-call-hook",
          memory_type: "config_change",
        }),
        signal: AbortSignal.timeout(5000),
      });
      console.warn(`[after-tool-call] config change captured: ${fileName} (agent=${agentId})`);
    } catch (err) {
      console.warn(`[after-tool-call] config capture failed: ${err.message}`);
    }
    return;
  }

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
        memory_type: classifyExecType(toolInput?.command || "") || undefined,
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
