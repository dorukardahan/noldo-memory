/**
 * NoldoMem lifecycle hooks — auto-recall and auto-capture.
 *
 * Auto-recall: before_prompt_build — inject relevant memories before every response
 * Auto-capture: user evidence and source-qualified generated media episodes
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

function boundedCaptureText(text, limit = 2000) {
  // Preserve the existing UTF-16 bound without sending a cut surrogate pair.
  return text.slice(0, limit).replace(/[\uD800-\uDBFF]$/u, "");
}

function isShortEventFact(text) {
  // Keep short declared event details without requiring a remember command or
  // padding. This is a bounded heuristic, not general fact extraction.
  if (/[?？]/u.test(text) || /\b(?:if|maybe|perhaps|might|could|would|whether|belki|muhtemelen|olabilir|galiba|sanırım|eğer)\b/iu.test(text)) return false;
  if (/(?:^|\s)(?:mı|mi|mu|mü)(?:\s|[.!]|$)/iu.test(text)) return false;
  const english = /^(?:my|our|the)\s+[^.!?\n]{0,100}\b(?:booking|reservation|appointment|meeting|guide|organizer|contact)\s+(?:is|are|was|were|will be|moved to|changed to)\s+\S/iu;
  const turkish = /^(?:rezervasyonum(?:uz)?|randevum(?:uz)?|toplantım(?:ız)?|rehberim(?:iz)?)\s+\S/iu;
  return english.test(text) || turkish.test(text);
}

function isQuestionOnly(text) {
  // Reject recognizable information-seeking questions before length/keyword
  // admission. Preserve mixed declarative turns and explicit memory requests.
  const sentences = text.trim().split(/(?<=[.!?？])\s+|\n+/u).filter(Boolean);
  return sentences.length > 0 && sentences.every(sentence => {
    if (!/[?？]$/u.test(sentence.trim())) return false;
    // A short topic phrase is not a declarative sentence: "For the visit, who...?".
    // Keep prefixes containing a clause, so a supplied change is not discarded.
    const topic = /^(?:for|about|regarding|concerning|as for)\s+([^,;:.!?]{1,120}),\s*/iu.exec(sentence);
    if (topic) {
      if (/\b(?:i|we|you|he|she|they|that|which|who|is|are|was|were|has|have|had|will|would|can|could|must|should)\b/iu.test(topic[1])) return false;
      sentence = sentence.slice(topic[0].length);
    }
    return /^(?:what|when|where|which|who|whose|why|how|ne|nerede|hangi|kim|kimin|neden|nasıl)\s/iu.test(sentence);
  });
}

function shouldCapture(text) {
  if (!text || text.length < 15) return false;
  if (looksLikePromptInjection(text)) return false;
  text = boundedCaptureText(text);
  if (SKIP_PATTERNS.some((p) => p.test(text))) return false;
  if (isQuestionOnly(text)) return false;
  // Capture if explicitly trigger-worthy or moderately long with substance
  return (
    CAPTURE_TRIGGERS.some((p) => p.test(text)) ||
    isShortEventFact(text) ||
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

function omitEmptyAudioPlaceholder(text) {
  // Exact stable-host failure output is not a transcript. Preserve separately
  // supplied user text and other successful media sections in the same turn.
  return text.replace(
    /(?:^|\n)\[Audio(?: \d+\/\d+)?\]\n(?:User text:\n([\s\S]*?)\n)?Transcript:\n\[Voice note could not be transcribed because the audio attachment was too small\](?=\s*$|\n\n\[)/gu,
    (_match, userText) => userText ? `\n${userText}\n` : "",
  ).trim();
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
  "noldomem_relearn_source", "plugin:noldomem_relearn_source",
  "noldomem/noldomem_relearn_source", "memory.noldomem_relearn_source",
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

async function readGatewayMediaHistory(api, cfg, ctx) {
  const { callGatewayFromCli } = await import("openclaw/plugin-sdk/gateway-runtime");
  return callGatewayFromCli("chat.history", {
    port: String(api.config.gateway.port), timeout: String(cfg.requestTimeoutMs || 15000), json: true,
  }, {agentId: resolveAgentId(ctx), sessionKey: ctx.sessionKey, limit: 20},
  {progress: false, scopes: ["operator.read"], sharedStateMode: "read-only"});
}

function captureRunKey(ctx) {
  const agent = resolveAgentId(ctx);
  return agent && typeof ctx.runId === "string" && ctx.runId &&
    typeof ctx.sessionKey === "string" && ctx.sessionKey.startsWith(`agent:${agent}:`) &&
    typeof ctx.sessionId === "string" && ctx.sessionId
    ? JSON.stringify([agent, ctx.sessionKey, ctx.sessionId, ctx.runId]) : null;
}

export function registerAutoCapture(api, client, cfg, readHistory = readGatewayMediaHistory) {
  // Only a hint to inspect the exact admitted source, never evidence that an
  // image belongs to this user turn: imagesCount can include context images.
  // Keep identities/count presence only, bounded even if completion never fires.
  const imageRuns = new Set();
  const documentRuns = new Map();
  api.on("before_prompt_build", (event, ctx) => {
    const key = captureRunKey(ctx);
    if (!key) return;
    documentRuns.delete(key);
    if (cfg.autoCaptureSource === "preprocessed" ||
        api.config?.plugins?.entries?.noldomem?.hooks?.allowConversationAccess !== true ||
        typeof event.prompt !== "string" || event.prompt.length > 32000) return;
    const documents = [];
    for (const match of event.prompt.matchAll(/<file name="([^"\n]{1,200})"(?: mime="[^"\n]*")?>\n([\s\S]*?)\n<\/file>/gu)) {
      const prepared = preprocessedFileText(match[0]);
      if (prepared.extracted && prepared.text.length >= 15 && !looksLikePromptInjection(prepared.text)) {
        documents.push({name: match[1], text: boundedCaptureText(prepared.text)});
      }
      if (documents.length >= cfg.captureMaxItems) break;
    }
    if (!documents.length) return;
    // Retain bounded derivatives only, never the composite system/model prompt.
    documentRuns.set(key, documents);
    if (documentRuns.size > 128) documentRuns.delete(documentRuns.keys().next().value);
  });
  api.on("llm_input", (event, ctx) => {
    const key = captureRunKey(ctx);
    if (!key) return;
    imageRuns.delete(key);
    if (event.runId !== ctx.runId || event.sessionId !== ctx.sessionId ||
        !Number.isInteger(event.imagesCount) || event.imagesCount <= 0) return;
    imageRuns.add(key);
    if (imageRuns.size > 128) imageRuns.delete(imageRuns.values().next().value);
  });
  if (cfg.autoCaptureSource === "preprocessed") {
    registerPreprocessedCapture(api, client, cfg);
  }
  api.on("agent_end", async (event, ctx) => {
    const runKey = captureRunKey(ctx);
    const hadImages = imageRuns.delete(runKey);
    const documents = documentRuns.get(runKey) || [];
    documentRuns.delete(runKey);
    if (!event.success) return;

    const agent = resolveAgentId(ctx);
    if (!agent) return;
    const messages = Array.isArray(event.messages) ? event.messages : [];
    const latestUser = messages.findLastIndex((message) => message?.role === "user");
    const sourceMessage = latestUser >= 0 ? messages[latestUser] : undefined;
    const texts = extractUserTextsFromMessages(cfg.autoCaptureSource === "preprocessed" || latestUser < 0 ? [] : [messages[latestUser]])
      .map(text => preprocessedFileText(omitEmptyAudioPlaceholder(text)));
    const blocks = latestUser >= 0 ? messages[latestUser]?.content : [];
    const media = Array.isArray(blocks) && blocks.some((block) =>
      ["image", "image_url", "input_audio", "audio", "file", "document"].includes(block?.type));
    const candidates = texts.filter(({text, extracted}) => shouldCapture(text) ||
      ((media || extracted || nativeMediaKind(text)) && text.length >= 15 && !looksLikePromptInjection(text))).slice(0, cfg.captureMaxItems);

    for (const {text, extracted} of candidates) {
      const kinds = new Set([nativeMediaKind(text), extracted ? "document" : null].filter(Boolean));
      const derivativeKind = kinds.size > 1 ? "mixed" : [...kinds][0];
      const derived = media || Boolean(derivativeKind);
      try {
        await client.store({
          text: boundedCaptureText(text),
          agent,
          source: "plugin-auto-capture",
          session_id: ctx.sessionKey || ctx.sessionId,
          evidence: { role: "user", assertion: derived ? "derived" : "reported", delivery: "received",
            // Binary presence does not prove the adjacent text came from it.
            modality: derivativeKind || "text", representation: derivativeKind ? "extracted_text" : "text",
            ...completionSourceEvidence(sourceMessage, derivativeKind) },
          namespace: cfg.defaultNamespace,
        });
      } catch (err) {
        console.warn(
          `[noldomem-plugin] auto-capture failed: ${err.message || err}`
        );
      }
    }
    let responseSource = sourceMessage;
    // Stable Codex finalization persists enriched media in chat.history but
    // supplies its earlier snapshot to agent_end. Recover metadata only via the
    // public local Gateway API, with an explicit conversation-access grant.
    const gateway = api.config?.gateway;
    if ((hadImages || documents.length) && cfg.captureMaxItems > candidates.length &&
        api.config?.plugins?.entries?.noldomem?.hooks?.allowConversationAccess === true &&
        gateway?.mode === "local" && gateway.bind === "loopback" &&
        Number.isInteger(gateway.port) && gateway.port > 0 && gateway.port <= 65535 &&
        (!event.runId || event.runId === ctx.runId) &&
        sourceMessage?.idempotencyKey === `${ctx.runId}:user` &&
        sourceMessage.__openclaw?.media === undefined &&
        sourceMessage.__openclaw?.mediaImageLayout === undefined) {
      try {
        const history = await readHistory(api, cfg, ctx);
        if (history?.sessionKey === ctx.sessionKey && history.sessionId === ctx.sessionId) {
          const sources = Array.isArray(history.messages) ? history.messages.filter(message =>
            message?.role === "user" && message.idempotencyKey === sourceMessage.idempotencyKey) : [];
          const stored = sources.length === 1 ? sources[0] : null;
          if (stored && stored.timestamp === sourceMessage.timestamp && stored.display !== false &&
              stored.excludeFromContext !== true &&
              (!stored.provenance?.kind || stored.provenance.kind === "external_user") &&
              JSON.stringify(stored.content) === JSON.stringify(sourceMessage.content)) {
            responseSource = {...sourceMessage, __openclaw: {...sourceMessage.__openclaw,
              media: stored.__openclaw?.media, mediaImageLayout: stored.__openclaw?.mediaImageLayout}};
          }
        }
      } catch {
        api.logger?.warn("noldomem: source media metadata unavailable");
      }
    }
    // The official prepared prompt can contain PDF text that the Codex completion
    // snapshot omits. Bind each derivative to that exact turn's durable attachment;
    // never recover text from upstreamUserText or infer a source from a filename alone.
    let documentCount = 0;
    const sourceMedia = responseSource?.__openclaw?.media;
    if (api.config?.plugins?.entries?.noldomem?.hooks?.allowConversationAccess === true &&
        !texts.some(part => part.extracted) && Array.isArray(sourceMedia) && sourceMessage?.idempotencyKey === `${ctx.runId}:user` &&
        (!event.runId || event.runId === ctx.runId)) {
      for (const document of documents) {
        if (candidates.length + documentCount >= cfg.captureMaxItems) break;
        const matching = sourceMedia.filter(fact => isManagedMediaReference(fact?.url) &&
          fact.url.slice("media://inbound/".length) === document.name &&
          (fact.kind === "document" || fact.kind === "file" || fact.contentType === "application/pdf"));
        if (matching.length !== 1) continue;
        // hydrationSuppressed prevents a later binary re-attachment; it does not
        // invalidate text actually extracted in this admitted turn.
        try {
          await client.store({text: document.text, agent, namespace: cfg.defaultNamespace,
            source: "plugin-document-derivative", session_id: ctx.sessionKey || ctx.sessionId,
            evidence: {...completionSourceEvidence(sourceMessage), role: "user", assertion: "derived",
              delivery: "received", modality: "document", representation: "extracted_text",
              reference: matching[0].url}});
          documentCount++;
        } catch { api.logger?.warn("noldomem: document derivative capture unavailable"); }
      }
    }
    const response = mediaResponseCandidate(event, ctx, responseSource, messages.slice(latestUser + 1));
    if (response && candidates.length + documentCount < cfg.captureMaxItems) {
      try {
        await client.store({...response, agent, namespace: cfg.defaultNamespace,
          session_id: ctx.sessionKey || ctx.sessionId});
      } catch {
        api.logger?.warn("noldomem: media-response capture unavailable");
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

function mediaResponseCandidate(event, ctx, sourceMessage, tail) {
  // Channel delivery remains owned by message_sent. Do not add a second
  // automatic writer for a known external-channel response.
  if ([ctx.channel, ctx.messageProvider].some(value => value && value !== "webchat")) return null;
  // The stable recorder binds an admitted chat input to this run. An image
  // count, old history row, or an unbound assistant answer cannot establish it.
  if (typeof ctx.runId !== "string" || !ctx.runId ||
      (event.runId && event.runId !== ctx.runId) || !(ctx.sessionKey || ctx.sessionId) ||
      sourceMessage?.idempotencyKey !== `${ctx.runId}:user` ||
      sourceMessage.display === false || sourceMessage.excludeFromContext === true ||
      (sourceMessage.provenance?.kind && sourceMessage.provenance.kind !== "external_user")) return null;
  const meta = sourceMessage.__openclaw;
  // Presence-check the host's actual submitted context only. Never persist or
  // extract from it. If NoldoMem context was supplied, this answer may merely
  // echo another source; do not relabel that as a fresh media episode.
  if (typeof meta?.upstreamUserText !== "string" ||
      /<relevant-memories>|&lt;relevant-memories&gt;/u.test(meta.upstreamUserText)) return null;
  const media = meta?.media;
  const layout = meta?.mediaImageLayout;
  if (!Array.isArray(media) || !Array.isArray(layout?.slots)) return null;
  const hasCurrentImage = layout.slots.some(slot => {
    if (!Number.isInteger(slot?.factIndex) || slot.factIndex < 0 ||
        !["inline", "offloaded"].includes(slot.kind) ||
        (Array.isArray(layout.suppressedFactIndexes) && layout.suppressedFactIndexes.includes(slot.factIndex))) return false;
    const fact = media[slot.factIndex];
    return typeof fact?.contentType === "string" && fact.contentType.startsWith("image/") && !fact.hydrationSuppressed &&
      (isManagedMediaReference(fact.url) ||
        (typeof fact.path === "string" && !/^[a-z][a-z\d+.-]*:/iu.test(fact.path) && safeMediaReference(fact.path)));
  });
  if (!hasCurrentImage) return null;
  // Do not turn recalled/revised/forgotten memory tool output into a fresh
  // source. Preserve the original memory-tool echo boundary for this writer.
  if (tail.some(message => isNoldoMemToolName(message?.toolName) ||
      (Array.isArray(message?.content) && message.content.some(block =>
        ["toolCall", "tool_use"].includes(block?.type) && isNoldoMemToolName(block.name))))) return null;
  const last = tail.at(-1);
  if (last?.role !== "assistant" || last.stopReason !== "stop" || last.errorMessage) return null;
  const assistantText = typeof last.content === "string" ? last.content :
    Array.isArray(last.content) ? last.content.filter(block => block?.type === "text" && typeof block.text === "string")
      .map(block => block.text).join("\n") : "";
  const requestText = extractUserTextsFromMessages([sourceMessage]).join("\n");
  if (assistantText.trim().length < 15 || looksLikePromptInjection(assistantText) ||
      looksLikePromptInjection(requestText)) return null;
  const evidence = completionSourceEvidence(sourceMessage, media.length === 1 ? "image" : "mixed");
  if (!evidence.event_id) return null;
  // This is an episode of what the model said about a media-bearing input,
  // not an assertion that each sentence was extracted from any particular file.
  return {
    text: boundedCaptureText(`Assistant response to media input (unverified interpretation):\n${boundedCaptureText(assistantText.trim(), 1500)}\nUser request:\n${boundedCaptureText(requestText.trim(), 300)}`),
    category: "assistant", memory_type: "conversation", source: "plugin-media-response",
    evidence: {...evidence, role: "assistant", assertion: "inferred", delivery: "generated",
      modality: media.length === 1 ? "image" : "mixed", representation: "text"},
  };
}

function completionSourceEvidence(message, derivativeKind) {
  // Stable UserTurnTranscriptRecorder retains these source fields on the
  // prepared row supplied to agent_end. The composite upstream prompt is not
  // an attachment extraction or source identity and is never stored here.
  const evidence = {};
  if (typeof message?.idempotencyKey === "string" && message.idempotencyKey.length > 0 &&
      message.idempotencyKey.length <= 200 && !/[\r\n]/u.test(message.idempotencyKey)) {
    evidence.event_id = message.idempotencyKey;
  }
  // Host row timestamps are milliseconds. This is an observation time, not a
  // supplied event date or revision-validity boundary.
  if (typeof message?.timestamp === "number" && Number.isFinite(message.timestamp) && message.timestamp >= 0) {
    evidence.observed_at = message.timestamp / 1000;
  }
  const media = message?.__openclaw?.media;
  if (!derivativeKind || derivativeKind === "mixed" || !Array.isArray(media) || media.length !== 1) return evidence;
  const fact = media[0];
  if (!fact || fact.hydrationSuppressed) return evidence;
  const reference = isManagedMediaReference(fact.url) ? fact.url :
    typeof fact.path === "string" && !/^[a-z][a-z\d+.-]*:/iu.test(fact.path) ? safeMediaReference(fact.path) : null;
  const matches = fact.kind === derivativeKind ||
    (derivativeKind === "document" && fact.kind === "file") ||
    (typeof fact.contentType === "string" && (fact.contentType.startsWith(`${derivativeKind}/`) ||
      (derivativeKind === "document" && /^(?:text\/|application\/(?:pdf|msword|vnd\.))/u.test(fact.contentType))));
  if (matches) {
    if (reference) evidence.reference = reference;
  }
  return evidence;
}

function registerPreprocessedCapture(api, client, cfg) {
  // Internal hooks do not inherit the typed conversation-access gate. Enforce
  // the same explicit grant here instead of using this API to bypass it.
  if (api.config?.plugins?.entries?.noldomem?.hooks?.allowConversationAccess !== true ||
      api.config?.hooks?.internal?.enabled === false || typeof api.registerHook !== "function") {
    api.logger?.warn("noldomem: preprocessed capture requires internal hooks and explicit conversation access");
    return;
  }
  api.registerHook("message:preprocessed", async (event) => {
    if (event?.type !== "message" || event.action !== "preprocessed") return;
    const agent = resolveAgentId({ sessionKey: event.sessionKey });
    if (!agent) return;
    const ctx = event.context || {};
    const transcript = typeof ctx.transcript === "string" ? ctx.transcript.trim() : "";
    const body = typeof ctx.bodyForAgent === "string" && ctx.bodyForAgent.trim()
      ? ctx.bodyForAgent : transcript || (typeof ctx.body === "string" ? ctx.body : "");
    const prepared = preprocessedFileText(omitEmptyAudioPlaceholder(
      transcript && !body.includes(transcript) ? `${body}\n${transcript}` : body));
    const text = prepared.text;
    const kinds = new Set([nativeMediaKind(text), transcript ? "audio" : null,
      prepared.extracted ? "document" : null].filter(Boolean));
    const derivativeKind = kinds.size > 1 ? "mixed" : [...kinds][0];
    if (!shouldCapture(text) && !(derivativeKind && text.length >= 15 &&
        !shouldSkipRecall(text) && !looksLikePromptInjection(text))) return;
    // Only the staged single-source fact is attributable to this derivative.
    // Pending/original URLs and multi-attachment lists are never guessed or fetched.
    const media = !ctx.mediaStagingPending && Array.isArray(ctx.media) ? ctx.media : [];
    const evidence = {
      // Prepared text may also contain unlabelled host link-understanding output.
      // Never promote that composite into a direct user assertion.
      role: "user", assertion: "derived", delivery: "received",
      modality: derivativeKind || "text", representation: derivativeKind ? "extracted_text" : "text",
    };
    if (typeof ctx.messageId === "string" && ctx.messageId.length <= 200) evidence.event_id = ctx.messageId;
    // FinalizedMsgContext timestamps are milliseconds; do not synthesize a missing event time.
    if (typeof ctx.timestamp === "number" && Number.isFinite(ctx.timestamp) && ctx.timestamp >= 0) {
      evidence.observed_at = ctx.timestamp / 1000;
    }
    if (derivativeKind && media.length === 1 &&
        (media[0]?.kind === derivativeKind || (derivativeKind === "document" && media[0]?.kind === "file") ||
         (typeof media[0]?.contentType === "string" &&
          media[0].contentType.startsWith(`${derivativeKind}/`)))) {
      const reference = safeMediaReference(media[0]?.url || media[0]?.path);
      if (reference) evidence.reference = reference;
    }
    try {
      await client.store({ text: boundedCaptureText(text), agent,
        source: "plugin-preprocessed", session_id: event.sessionKey,
        namespace: cfg.defaultNamespace, evidence });
    } catch {
      api.logger?.warn("noldomem: preprocessed capture unavailable");
    }
  }, { name: "noldomem-preprocessed-capture", description: "Capture existing inbound text derivatives with source evidence" });
}

function preprocessedFileText(body) {
  let extracted = false;
  // Remove only the leading transport placeholder before unwrapping contents.
  // A document may itself contain the same literal marker (e.g. a manual).
  const text = body.replace(/^<media:document>\s*(?=<file name=")/u, "")
    .replace(/<file name="[^"\n]*"(?: mime="[^"\n]*")?>\n([\s\S]*?)\n<\/file>/gu,
    (_block, content) => {
      // Only successful extraction has the host's matching untrusted envelope.
      // Failure/path-only/rendered-image markers are not document contents.
      const match = content.trim().match(/^<<<EXTERNAL_UNTRUSTED_CONTENT id="([a-f0-9]{16})">>>\nSource: [^\n]+\n---\n([\s\S]*)\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id="\1">>>$/u);
      if (!match) return "";
      extracted = true;
      // Remove random wrapper IDs for stable deduplication, not the trust boundary:
      // the extracted text is still screened, stored as derived and injected untrusted.
      return match[2];
    }).trim();
  return { text, extracted };
}

function isManagedMediaReference(value) {
  return typeof value === "string" && /^media:\/\/inbound\/[a-zA-Z0-9][a-zA-Z0-9._-]{0,199}$/u.test(value) &&
    !value.includes("..") && !/[\r\n]/u.test(value);
}

function safeMediaReference(value) {
  if (typeof value !== "string" || value.length > 1000 || /[\r\n]/u.test(value)) return null;
  if (isManagedMediaReference(value)) return value;
  if (/^https?:\/\//iu.test(value)) {
    try {
      const url = new URL(value);
      if (url.username || url.password) return null;
      url.search = "";
      url.hash = "";
      return url.href;
    } catch { return null; }
  }
  return /^[a-z][a-z\d+.-]*:/iu.test(value) ? null : value;
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
