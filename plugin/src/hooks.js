/**
 * NoldoMem lifecycle hooks — auto-recall and auto-capture.
 *
 * Auto-recall: before_prompt_build — inject relevant memories before every response
 * Auto-capture: agent_end — capture important user messages after each turn
 */

import {
  looksLikePromptInjection,
  escapeForPrompt,
  formatRelevantMemoriesContext,
} from "./sanitize.js";

function resolveAgentId(ctx) {
  const sk = ctx?.sessionKey || "";
  const match = sk.match(/^agent:([^:]+)/);
  return match ? match[1] : "main";
}

function extractUserText(prompt) {
  // The prompt in before_prompt_build contains the full system prompt + user message
  // We want to extract the user's actual question for recall
  // Take last 500 chars as a heuristic for the user query portion
  if (!prompt || prompt.length < 10) return null;
  // If prompt is short enough, use it directly
  if (prompt.length <= 500) return prompt;
  // Otherwise take the tail which is more likely the user's actual input
  return prompt.slice(-500);
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
  if (!text || text.length < 10) return true;
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

function shouldCapture(text) {
  if (!text || text.length < 15 || text.length > 2000) return false;
  if (looksLikePromptInjection(text)) return false;
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

// Patterns that suggest the user is asking about past context, decisions, or memory
const RECALL_TRIGGER_PATTERNS = [
  /\b(hatırla|remember|recall|daha önce|earlier|previously|geçen sefer|last time)\b/i,
  /\b(karar|decision|kararlaştır|agreed|anlaştık)\b/i,
  /\b(ne yapmıştık|what did we|nerede kaldık|where were we)\b/i,
  /\b(tercih|preference|always|her zaman|never|asla)\b/i,
  /\b(config|credential|deploy|push|commit|migration)\b/i,
  /\b(lesson|ders|kural|rule|öğren)\b/i,
  /\b(durum|status|ne oldu|what happened|sorun|problem|issue|bug)\b/i,
  /\?/, // Questions are good recall candidates
];

function shouldTriggerRecall(text) {
  if (!text || text.length < 15) return false;
  return RECALL_TRIGGER_PATTERNS.some((p) => p.test(text));
}

export function registerAutoRecall(api, client, cfg) {
  api.on("before_prompt_build", async (event, ctx) => {
    const userQuery = extractUserText(event.prompt);
    if (shouldSkipRecall(userQuery)) return;

    // Only trigger recall for messages that look like they need memory context
    // This avoids 6-7s embedding latency on every single message
    if (!shouldTriggerRecall(userQuery)) return;

    const agent = resolveAgentId(ctx);

    try {
      // Use AbortSignal with generous timeout — embedding on CPU takes ~6s
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 15000);

      const data = await client.recall({
        query: userQuery,
        limit: cfg.recallLimit,
        agent,
        namespace: cfg.defaultNamespace,
        max_tokens: cfg.recallMaxTokens,
      });

      clearTimeout(timer);

      const results = (data.results || []).filter(
        (r) => !looksLikePromptInjection(r.text || r.content || "")
      );

      if (results.length === 0) return;

      const context = formatRelevantMemoriesContext(
        results.map((r) => ({
          category: r.memory_type || r.category || "other",
          text: (r.text || r.content || "").slice(0, 500),
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
    const texts = extractUserTextsFromMessages(event.messages);
    const candidates = texts.filter(shouldCapture).slice(0, cfg.captureMaxItems);

    for (const text of candidates) {
      try {
        await client.store({
          text: text.slice(0, 2000),
          agent,
          source: "plugin-auto-capture",
          namespace: cfg.defaultNamespace,
        });
      } catch (err) {
        console.warn(
          `[noldomem-plugin] auto-capture failed: ${err.message || err}`
        );
      }
    }
  });
}
