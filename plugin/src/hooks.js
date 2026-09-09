/**
 * NoldoMem lifecycle hooks — auto-recall and auto-capture.
 *
 * Auto-recall: before_prompt_build — inject relevant memories before every response
 * Auto-capture: agent_end — capture important user messages after each turn
 */

import {
  looksLikePromptInjection,
  hasSafeRecallMetadata,
  formatRelevantMemoriesContext,
} from "./sanitize.js";

import { resolveAgentId } from "./scope.js";

function extractUserText(prompt) {
  // The host supplies the current user prompt separately from system context.
  if (typeof prompt !== "string") return null;
  return prompt.trim().slice(0, 1950);
}

const SKIP_PATTERNS = [
  /^HEARTBEAT/i,
  /^\[cron:/i,
  /^NO_REPLY$/,
  /^\/\w+/,  // slash commands
  /A new session was started/i,
  /Pre-compaction memory flush/i,
];

function shouldSkipRecall(text) {
  if (!text || text.length < 3) return true;
  if (/^(?:hi|hello|hey|ok|okay|yes|no|thanks(?:,? that is all)?|thank you|done|continue|merhaba|tamam|evet|hayır|teşekkürler)[\s!.?,]*$/iu.test(text)) return true;
  return SKIP_PATTERNS.some((p) => p.test(text));
}

// Capture heuristics — only capture high-signal user messages
const CAPTURE_TRIGGERS = [
  /\b(remember|hatırla|kaydet|note|not al)\b/i,
  /\b(karar|decided|decision|agreed|anlaştık|yapalım)\b/i,
  /\b(prefer|tercih|always|her zaman|never|asla)\b/i,
  /\b(important|önemli|critical|kritik)\b/i,
  /\b(rule|kural|policy|politika)\b/i,
];

function boundedCaptureText(text) {
  // Preserve the existing UTF-16 bound without sending a cut surrogate pair.
  return text.slice(0, 2000).replace(/[\uD800-\uDBFF]$/u, "");
}

function shouldCapture(text) {
  if (!text || text.length < 15) return false;
  if (looksLikePromptInjection(text)) return false;
  text = boundedCaptureText(text);
  if (SKIP_PATTERNS.some((p) => p.test(text))) return false;
  // Capture if explicitly trigger-worthy or moderately long with substance
  return (
    CAPTURE_TRIGGERS.some((p) => p.test(text)) ||
    (text.length > 80 && !text.startsWith("```"))
  );
}

function extractUserTextsFromMessages(messages) {
  if (!messages || !Array.isArray(messages)) return [];
  const texts = [];
  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    if (msg.role !== "user") continue;
    const content = msg.content;
    if (typeof content === "string") {
      texts.push(content);
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (block?.type === "text" && typeof block.text === "string") {
          texts.push(block.text);
        }
      }
    }
  }
  return texts;
}

function nativeMediaKind(text) {
  // Stable host text derivatives, not evidence that a raw attachment was read.
  // A forged marker can only downgrade trust to derived, never promote it.
  const kinds = [...text.matchAll(/(?:^|\n)\[(Audio|Image|Video)(?: \d+\/\d+)?\]\n(?:User text:\n[\s\S]*?\n)?(?:Transcript|Description):\n/gu)]
    .map(match => match[1] === "Video" ? "mixed" : match[1].toLowerCase());
  return kinds.length ? (new Set(kinds).size === 1 ? kinds[0] : "mixed") : null;
}

const SECRET_PATTERNS = [
  /((?:["'])?(?:api[_-]?key|token|secret|password|passwd|pwd)(?:["'])?\s*[:=]\s*)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}\]]+)/gi,
  /\bBearer\s+[A-Za-z0-9._~+/=-]{12,}/gi,
  /\bsk-[A-Za-z0-9_-]{12,}/g,
];

const OPERATIONAL_TOOL_PATTERNS = [
  /\bsystemctl\b/i,
  /\bdocker(?:\s+compose)?\b/i,
  /\bgit\s+(?:commit|merge|push|pull|tag|checkout|switch)\b/i,
  /\b(openclaw|clawhub)\s+(?:update|plugins|hooks|install|publish)\b/i,
  /\b(?:npm|pnpm|yarn|pip|uv|apt)\s+(?:install|add|update|upgrade)\b/i,
  /\b(error|failed|traceback|exception|timeout|oom|sigkill)\b/i,
];

// Canonical ids plus the qualification forms known to be emitted by OpenClaw.
// This is deliberately explicit: suffix matching suppresses unrelated plugins.
const NOLDOMEM_TOOL_NAMES = new Set([
  "noldomem_recall", "noldomem_store", "noldomem_pin", "noldomem_forget",
  "plugin:noldomem_forget", "noldomem/noldomem_forget", "memory.noldomem_forget",
  "plugin:noldomem_recall", "plugin:noldomem_store", "plugin:noldomem_pin",
  "noldomem/noldomem_recall", "noldomem/noldomem_store", "noldomem/noldomem_pin",
  "memory.noldomem_recall", "memory.noldomem_store", "memory.noldomem_pin",
]);

function isNoldoMemToolName(toolName) {
  return typeof toolName === "string" && NOLDOMEM_TOOL_NAMES.has(toolName.trim().toLowerCase());
}

function redactOperationalText(text) {
  let out = String(text || "");
  out = out.replace(SECRET_PATTERNS[0], (_match, prefix) => `${prefix}<redacted>`);
  for (const pattern of SECRET_PATTERNS.slice(1)) out = out.replace(pattern, "<redacted>");
  return out;
}

function sanitizeStructuredValue(value, active = new WeakSet()) {
  if (typeof value === "string") {
    const candidate = value.trim();
    if (!((candidate.startsWith("{") && candidate.endsWith("}")) ||
          (candidate.startsWith("[") && candidate.endsWith("]")))) return value;
    try {
      const parsed = JSON.parse(candidate);
      if (!Array.isArray(parsed) &&
          (parsed === null || Object.getPrototypeOf(parsed) !== Object.prototype)) return value;
      return JSON.stringify(sanitizeStructuredValue(parsed, active));
    } catch {
      return value;
    }
  }
  if (value == null || typeof value === "boolean" || typeof value === "number") return value;
  if (typeof value !== "object") return "<redacted>";
  if (active.has(value)) return "<redacted>";
  const prototype = Object.getPrototypeOf(value);
  if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) return "<redacted>";
  active.add(value);
  try {
    if (Array.isArray(value)) {
      return value.map((item) => sanitizeStructuredValue(item, active));
    }
    const sanitized = {};
    for (const [key, item] of Object.entries(value)) {
      sanitized[key] = /^(?:api[_-]?key|token|secret|password|passwd|pwd)$/i.test(key)
        ? "<redacted>"
        : sanitizeStructuredValue(item, active);
    }
    return sanitized;
  } finally {
    active.delete(value);
  }
}

function toCompactText(value, maxChars = 1200) {
  if (value == null) return "";
  let text = "";
  try {
    const sanitized = sanitizeStructuredValue(value);
    text = typeof sanitized === "string" ? sanitized : JSON.stringify(sanitized);
  } catch {
    text = "<redacted>";
  }
  text = redactOperationalText(text).replace(/\s+/g, " ").trim();
  return text.length > maxChars ? `${text.slice(0, maxChars)}...` : text;
}

function shouldCaptureOperationalTool(event) {
  // Avoid recursively capturing NoldoMem's own explicit memory tool calls.
  if (isNoldoMemToolName(event?.toolName)) return false;

  const haystack = [
    event?.toolName,
    toCompactText(event?.params, 500),
    toCompactText(event?.error, 500),
    toCompactText(event?.result, 800),
  ].join(" ");
  return OPERATIONAL_TOOL_PATTERNS.some((pattern) => pattern.test(haystack));
}

function extractMessageText(message) {
  if (!message || typeof message !== "object") return "";
  const content = message.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (part?.type === "text" && typeof part.text === "string" ? part.text : ""))
      .filter(Boolean)
      .join("\\n");
  }
  return "";
}

function selectCompactionMessages(messages) {
  if (!Array.isArray(messages)) return [];
  const picked = [];
  for (const message of messages.slice(-60)) {
    const role = message?.role;
    if (role !== "user" && role !== "assistant") continue;
    const text = redactOperationalText(extractMessageText(message)).trim();
    if (!shouldCapture(text)) continue;
    picked.push({ role, text: boundedCaptureText(text) });
    if (picked.length >= 20) break;
  }
  return picked;
}

export function registerAutoRecall(api, client, cfg) {
  api.on("before_prompt_build", async (event, ctx) => {
    const userQuery = extractUserText(event.prompt);
    if (shouldSkipRecall(userQuery)) return;

    const agent = resolveAgentId(ctx);
    if (!agent) return;

    try {
      const data = await client.recall({
        query: userQuery,
        limit: cfg.recallLimit,
        agent,
        namespace: cfg.defaultNamespace,
        max_tokens: cfg.recallMaxTokens,
        ...(cfg.recallMinSemanticScore != null ? { min_semantic_score: cfg.recallMinSemanticScore } : {}),
      });

      const results = (data.results || []).filter((r) =>
        hasSafeRecallMetadata(r) && !looksLikePromptInjection(r.text || r.content || "")
      );

      if (results.length === 0) return;

      const context = formatRelevantMemoriesContext(
        results.map((r) => ({
          category: r.memory_type || r.category || "other",
          text: `[id=${r.id} valid_from=${r.valid_from ?? "unknown"} valid_to=${r.valid_to ?? "open"} evidence=${JSON.stringify(r.evidence || {})}] ${(r.text || r.content || "").slice(0, 500)}`,
        }))
      );

      return { prependContext: context };
    } catch (err) {
      // Silently skip — don't block the agent response
      if (err.name !== "AbortError") {
        console.warn(`[noldomem-plugin] auto-recall failed: ${err.message || err}`);
      }
    }
  });
}

export function registerAutoCapture(api, client, cfg) {
  api.on("agent_end", async (event, ctx) => {
    if (!event.success) return;

    const agent = resolveAgentId(ctx);
    if (!agent) return;
    const messages = Array.isArray(event.messages) ? event.messages : [];
    const latestUser = messages.findLastIndex((message) => message?.role === "user");
    const texts = extractUserTextsFromMessages(latestUser < 0 ? [] : [messages[latestUser]]);
    const blocks = latestUser >= 0 ? messages[latestUser]?.content : [];
    const media = Array.isArray(blocks) && blocks.some((block) =>
      ["image", "image_url", "input_audio", "audio", "file", "document"].includes(block?.type));
    const candidates = texts.filter((text) => shouldCapture(text) ||
      ((media || nativeMediaKind(text)) && text.length >= 15 && !looksLikePromptInjection(text))).slice(0, cfg.captureMaxItems);

    for (const text of candidates) {
      const derivativeKind = nativeMediaKind(text);
      const derived = media || derivativeKind !== null;
      try {
        await client.store({
          text: boundedCaptureText(text),
          agent,
          source: "plugin-auto-capture",
          session_id: ctx.sessionKey || ctx.sessionId,
          evidence: { role: "user", assertion: derived ? "derived" : "reported", delivery: "received",
            modality: derivativeKind || (media ? "mixed" : "text"), representation: derived ? "extracted_text" : "text" },
          namespace: cfg.defaultNamespace,
        });
      } catch (err) {
        console.warn(
          `[noldomem-plugin] auto-capture failed: ${err.message || err}`
        );
      }
    }
  });

  // agent_end confirms generation, not channel delivery. Only this host event
  // can label outgoing text as delivered. Media-only payloads remain a host gap.
  api.on("message_sent", async (event, ctx) => {
    if (event?.success !== true || typeof event.content !== "string") return;
    const agent = resolveAgentId(ctx);
    if (!agent || !shouldCapture(event.content)) return;
    try {
      await client.store({
        text: boundedCaptureText(event.content), agent, source: "plugin-message-sent",
        namespace: cfg.defaultNamespace, session_id: ctx.sessionKey,
        category: "assistant", memory_type: "conversation",
        evidence: { event_id: event.messageId, role: "assistant", assertion: "derived", delivery: "delivered" },
      });
    } catch {
      api.logger?.warn("noldomem: delivered-text capture unavailable");
    }
  });
}

export function registerNativeLifecycleCapture(api, client, cfg) {
  if (cfg.enableOperationalCapture) {
    api.on("after_tool_call", async (event, ctx) => {
      if (!shouldCaptureOperationalTool(event)) return;
      const agent = resolveAgentId(ctx);
      if (!agent) return;
      const params = toCompactText(event?.params, 700);
      const result = event?.error ? toCompactText(event.error, 1000) : toCompactText(event?.result, 1000);
      const text = [
        `Tool call: ${event?.toolName || "unknown"}`,
        event?.durationMs ? `Duration: ${event.durationMs}ms` : "",
        params ? `Params: ${params}` : "",
        result ? `Result: ${result}` : "",
      ]
        .filter(Boolean)
        .join("\\n");

      try {
        await client.store({
          text: text.slice(0, 2400),
          agent,
          source: "plugin-after-tool-call",
          session_id: ctx.sessionKey || ctx.sessionId,
          evidence: { role: "tool", assertion: "derived", delivery: "generated" },
          namespace: cfg.defaultNamespace,
          category: event?.error ? "error" : "tool",
          importance: event?.error ? 0.85 : 0.65,
        });
      } catch (err) {
        console.warn(`[noldomem-plugin] after_tool_call capture failed: ${err.message || err}`);
      }
    });
  }

  if (cfg.enableCompactionCapture) {
    api.on("before_compaction", async (event, ctx) => {
      const messages = selectCompactionMessages(event?.messages);
      if (messages.length === 0) return;
      const agent = resolveAgentId(ctx);
      if (!agent) return;
      try {
        await client.capture({
          messages: messages.map((message) => ({ ...message, session: ctx.sessionKey || ctx.sessionId,
            evidence: { role: message.role, assertion: message.role === "user" ? "reported" : "derived",
              delivery: message.role === "user" ? "received" : "generated" } })),
          agent,
          source: "plugin-before-compaction",
          namespace: cfg.defaultNamespace,
        });
      } catch (err) {
        console.warn(`[noldomem-plugin] before_compaction capture failed: ${err.message || err}`);
      }
    });
  }

  if (cfg.enableSubagentCapture) {
    api.on("subagent_ended", async (event, ctx) => {
      if (!event?.outcome || event.outcome === "ok") return;
      const agent = resolveAgentId(ctx);
      if (!agent) return;
      if (resolveAgentId({ sessionKey: event.targetSessionKey }) !== agent) return;
      const text = [
        `Subagent ended with outcome: ${event.outcome}`,
        event.targetSessionKey ? `Target session: ${event.targetSessionKey}` : "",
        event.reason ? `Reason: ${event.reason}` : "",
        event.error ? `Error: ${redactOperationalText(event.error)}` : "",
      ]
        .filter(Boolean)
        .join("\\n");
      try {
        await client.store({
          text: boundedCaptureText(text),
          agent,
          source: "plugin-subagent-ended",
          session_id: event.targetSessionKey,
          evidence: { role: "tool", assertion: "derived", delivery: "generated" },
          namespace: cfg.defaultNamespace,
          category: "subagent",
          importance: 0.75,
        });
      } catch (err) {
        console.warn(`[noldomem-plugin] subagent_ended capture failed: ${err.message || err}`);
      }
    });
  }
}
