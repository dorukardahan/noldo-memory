/**
 * message-recall hook — Mid-conversation recall for feature/config/status questions.
 *
 * When user asks "does X support Y?", "X var mı?", "X çalışıyor mu?" etc.,
 * this hook recalls relevant lessons and verified facts from NoldoMem and
 * injects them into the conversation via event.messages.
 *
 * Timeout: 2 seconds max, graceful skip.
 * Max results: 3, max 500 chars each.
 *
 * Created: 2026-03-22 [Fabrication Prevention System]
 */

import { readFileSync } from "node:fs";
import {
  resolveAgentId,
  resolveWorkspaceDir,
  stripChannelEnvelope,
} from "../lib/runtime.js";
import { readVerifiedFacts } from "../lib/shared-state.js";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH =
  process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try {
  _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim();
} catch (e) {
  console.warn("[message-recall] error:", e.message || e);
}

const RECALL_TIMEOUT_MS = 2000;
const MAX_RESULTS = 3;
const MAX_CHARS_PER_RESULT = 500;

// ── Question Detection Patterns ──

// Turkish question patterns
const TR_QUESTION_PATTERNS = [
  /(?:var\s+mı|mevcut\s+mu|destekl(?:iyor|eniyor)\s+mu|çalışıyor\s+mu|aktif\s+mi|açık\s+mı)/i,
  /(?:yapabiliyor\s+mu|kullanıyor\s+mu|kullanılabili(?:r|yor)\s+mu)/i,
  /(?:nerede|nedir|nasıl|ne\s+zaman|hangi)/i,
  /(?:biliyor\s+mu|hatırlıyor\s+mu)/i,
  /(?:ne\s+(?:durumda|halde)|durumu\s+ne)/i,
];

// English question patterns
const EN_QUESTION_PATTERNS = [
  /(?:does\s+\w+\s+(?:support|have|include|provide|offer))/i,
  /(?:is\s+\w+\s+(?:running|active|enabled|supported|available|working))/i,
  /(?:can\s+\w+\s+(?:do|handle|process|run|use))/i,
  /(?:where\s+is|what\s+is|how\s+(?:does|do|to|is))/i,
  /(?:do\s+we\s+have|are\s+there\s+any)/i,
];

// Config/feature/service specific patterns
const SPECIFICITY_PATTERNS = [
  /(?:MCP|hook|plugin|endpoint|API|config|service|cron|database|port)/i,
  /(?:OpenClaw|NoldoMem|Slack|Telegram|Signal|Docker|systemd|nginx)/i,
  /(?:feature|özellik|capability|destekle|support)/i,
  /(?:password|şifre|credential|secret|key|token)/i,
  /(?:fleet|steward|dashboard|monitor)/i,
];

// Noise patterns — skip these even if they match question patterns
const SKIP_PATTERNS = [
  /^(?:evet|hayır|tamam|ok|yes|no)\s*$/i,
  /^\s*$/,
  /^HEARTBEAT/i,
  /^\[cron:/i,
];

function isQuestion(text) {
  if (!text || text.length < 8) return false;
  if (SKIP_PATTERNS.some((p) => p.test(text))) return false;

  const isQuestionMark = text.includes("?");
  const matchesTR = TR_QUESTION_PATTERNS.some((p) => p.test(text));
  const matchesEN = EN_QUESTION_PATTERNS.some((p) => p.test(text));
  const matchesSpecific = SPECIFICITY_PATTERNS.some((p) => p.test(text));

  // Must match at least one question pattern OR have a question mark + specificity
  return matchesTR || matchesEN || (isQuestionMark && matchesSpecific);
}

// ── Keyword Extraction ──

const STOP_WORDS = new Set([
  "bir", "bu", "şu", "o", "ve", "ile", "için", "mi", "mu", "mı", "mü",
  "var", "yok", "ne", "nasıl", "nerede", "neden", "hangi", "kaç",
  "the", "a", "an", "is", "are", "does", "do", "have", "has", "can",
  "what", "where", "how", "when", "which", "who", "in", "on", "at",
  "to", "for", "with", "from", "by", "of", "it", "this", "that",
  "we", "you", "they", "our", "your", "my", "its",
  "mevcut", "destekliyor", "çalışıyor", "kullanıyor", "aktif",
  "running", "working", "supported", "enabled", "available",
]);

function extractKeywords(text) {
  const words = text
    .replace(/[?!.,;:'"()\[\]{}]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP_WORDS.has(w.toLowerCase()))
    .map((w) => w.toLowerCase());

  // Dedupe and take top 5
  return [...new Set(words)].slice(0, 5);
}

// ── Recall ──

async function recallForQuestion(agentId, keywords) {
  if (!_memoryApiKey || keywords.length === 0) return [];

  const query = keywords.join(" ");

  // Single global deadline for entire recall (both calls must finish within budget)
  const deadline = AbortSignal.timeout(RECALL_TIMEOUT_MS);

  // Run lesson + general recall in parallel with shared deadline
  const [lessonResult, generalResult] = await Promise.allSettled([
    fetch(`${MEMORY_API}/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
      body: JSON.stringify({
        query,
        limit: MAX_RESULTS,
        min_score: 0.20,
        agent: agentId,
        memory_type: "lesson",
      }),
      signal: deadline,
    }).then((r) => r.ok ? r.json() : { results: [] }),
    fetch(`${MEMORY_API}/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
      body: JSON.stringify({
        query,
        limit: MAX_RESULTS,
        min_score: 0.25,
        agent: agentId,
      }),
      signal: deadline,
    }).then((r) => r.ok ? r.json() : { results: [] }),
  ]);

  const results = [];
  if (lessonResult.status === "fulfilled") {
    results.push(...(lessonResult.value.results || []).map((r) => ({ ...r, _source: "lesson" })));
  }
  if (generalResult.status === "fulfilled") {
    results.push(...(generalResult.value.results || []).map((r) => ({ ...r, _source: "general" })));
  }

  return results.slice(0, MAX_RESULTS);
}

// ── Verified Facts Lookup ──

function lookupVerifiedFacts(workspaceDir, keywords) {
  const facts = readVerifiedFacts(workspaceDir);
  if (!facts.facts || Object.keys(facts.facts).length === 0) return [];

  const matches = [];
  for (const [key, fact] of Object.entries(facts.facts)) {
    if (fact.stale) continue; // Skip expired facts
    const factText = `${key} ${fact.claim || ""}`.toLowerCase();
    const matchCount = keywords.filter((kw) => factText.includes(kw)).length;
    if (matchCount > 0) {
      matches.push({ key, fact, matchCount });
    }
  }

  return matches
    .sort((a, b) => b.matchCount - a.matchCount)
    .slice(0, 2)
    .map((m) => m.fact);
}

// ── Main Hook ──

const messageRecallHook = async (event) => {
  if (event.type !== "message" || event.action !== "received") return;

  const rawContent = String(event?.context?.content || "");
  const content = stripChannelEnvelope(rawContent).trim();

  if (!isQuestion(content)) return;

  const workspaceDir = resolveWorkspaceDir(event);
  const agentId = resolveAgentId(event, workspaceDir);

  console.warn(`[message-recall] question detected (agent=${agentId}): ${content.slice(0, 80)}`);

  const keywords = extractKeywords(content);
  if (keywords.length === 0) return;

  console.warn(`[message-recall] keywords: ${keywords.join(", ")}`);

  // Parallel: NoldoMem recall + verified facts lookup
  const [memories, verifiedFacts] = await Promise.all([
    recallForQuestion(agentId, keywords).catch(() => []),
    Promise.resolve(lookupVerifiedFacts(workspaceDir, keywords)),
  ]);

  const parts = [];

  // Verified facts first (highest confidence)
  if (verifiedFacts.length > 0) {
    parts.push("**Verified Facts:**");
    for (const fact of verifiedFacts) {
      const status = fact.verified ? "✅" : "❌";
      parts.push(`${status} ${(fact.claim || "").slice(0, MAX_CHARS_PER_RESULT)} (source: ${(fact.source || "unknown").slice(0, 100)})`);
    }
  }

  // Memory recall results
  if (memories.length > 0) {
    parts.push("**Related Memories:**");
    for (const mem of memories) {
      const text = (mem.text || mem.content || "").slice(0, MAX_CHARS_PER_RESULT);
      const source = mem._source === "lesson" ? "📝 Lesson" : "💭 Memory";
      parts.push(`${source}: ${text}`);
    }
  }

  if (parts.length === 0) return;

  // Inject as system message
  const recallMessage = [
    "⚡ **Mid-Conversation Recall** (auto-injected by message-recall hook)",
    "Bu bilgileri yanıt vermeden ÖNCE kontrol et. Fabrication'ı önlemek için buraya bak:",
    "",
    ...parts,
    "",
    "Eğer bu bilgiler soruyla ilgiliyse, bunları dikkate al. İlgili değilse yoksay.",
  ].join("\n");

  if (event.messages && Array.isArray(event.messages)) {
    event.messages.push(recallMessage);
    console.warn(`[message-recall] injected ${parts.length} results for agent=${agentId}`);
  } else {
    console.warn("[message-recall] event.messages not available, cannot inject");
  }
};

import { resilientHandler } from "../lib/resilient-import.js";
export default resilientHandler(messageRecallHook, "message-recall");
