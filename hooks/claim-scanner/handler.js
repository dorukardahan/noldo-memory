/**
 * claim-scanner hook — Post-response claim scanner.
 *
 * Scans agent responses for unverified claims:
 * - Negative claims: "not supported", "doesn't exist", "yok", "desteklemiyor"
 * - Positive claims: "fixed", "updated", "düzelttim", "tamamlandı"
 * - Config/credential claims: hash references, password mentions
 *
 * Cross-checks against recent tool calls. If no verification evidence,
 * logs to fabrication-log.json and stores unverified_claim to NoldoMem.
 *
 * Does NOT block responses — post-mortem audit only.
 * Timeout: 3 seconds max.
 *
 * Created: 2026-03-22 [Fabrication Prevention System]
 */

import { readFileSync } from "node:fs";
import {
  resolveAgentId,
  resolveWorkspaceDir,
  stripChannelEnvelope,
} from "../lib/runtime.js";
import {
  hasRecentVerificationTool,
  appendFabricationIncident,
} from "../lib/shared-state.js";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH =
  process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try {
  _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim();
} catch (e) {
  console.warn("[claim-scanner] error:", e.message || e);
}

const SCAN_TIMEOUT_MS = 3000;

// ── Claim Detection Patterns ──

const NEGATIVE_CLAIM_PATTERNS = [
  // Turkish
  { pattern: /(?:destekle(?:miyor|nmez|nmedi)|yok|mevcut\s+değil|bulunmuyor)/i, type: "feature_claim" },
  { pattern: /(?:yapamıyor|kullanılamaz|çalışmıyor|aktif\s+değil)/i, type: "feature_claim" },
  { pattern: /(?:mümkün\s+değil|imkansız|olmaz)/i, type: "feature_claim" },
  // English
  { pattern: /(?:not\s+support(?:ed)?|doesn'?t\s+(?:have|support|exist|include))/i, type: "feature_claim" },
  { pattern: /(?:not\s+available|not\s+(?:possible|implemented|enabled))/i, type: "feature_claim" },
  { pattern: /(?:there\s+is\s+no|there\s+are\s+no|no\s+(?:support|way)\s+for)/i, type: "feature_claim" },
  { pattern: /(?:cannot|can'?t\s+(?:do|handle|use|run))/i, type: "feature_claim" },
];

const POSITIVE_CLAIM_PATTERNS = [
  // Turkish
  { pattern: /(?:düzelttim|güncelledim|tamamlandı|hallettim|yaptım|ekledim)/i, type: "completion_claim" },
  { pattern: /(?:fix(?:le)?dim|deploy\s+ettim|push\s+ettim|commit\s+ettim)/i, type: "completion_claim" },
  // English
  { pattern: /(?:I\s+(?:fixed|updated|completed|resolved|patched|deployed|pushed))/i, type: "completion_claim" },
  { pattern: /(?:has\s+been\s+(?:fixed|updated|resolved|deployed|patched))/i, type: "completion_claim" },
  { pattern: /(?:successfully\s+(?:updated|deployed|fixed|patched|completed))/i, type: "completion_claim" },
];

const CONFIG_CLAIM_PATTERNS = [
  // Hex hash references (SHA, MD5, etc.)
  { pattern: /\b[0-9a-f]{32,64}\b/i, type: "config_claim" },
  // Password/credential mentions with specific values
  { pattern: /(?:password|şifre|parola)\s*(?:is|=|:)\s*\S+/i, type: "credential_claim" },
  { pattern: /(?:token|api[_-]?key|secret)\s*(?:is|=|:)\s*\S+/i, type: "credential_claim" },
];

// Skip patterns — don't flag these contexts
const SKIP_CONTEXTS = [
  /```[\s\S]*?```/g,        // Code blocks
  /`[^`]+`/g,               // Inline code
  /^\s*#/gm,                // Headers
  /^\s*\|.*\|/gm,           // Table rows
  /HEARTBEAT_OK/i,
  /NO_REPLY/i,
];

// Low-signal response patterns — don't scan short/trivial responses
const LOW_SIGNAL_RESPONSE = [
  /^(?:evet|hayır|tamam|ok|yes|no|done|✅|👍)\s*$/i,
  /^HEARTBEAT/i,
  /^NO_REPLY$/,
];

function cleanForScanning(text) {
  let cleaned = text;
  // Remove code blocks and inline code before scanning
  cleaned = cleaned.replace(/```[\s\S]*?```/g, " ");
  cleaned = cleaned.replace(/`[^`]+`/g, " ");
  return cleaned;
}

function detectClaims(text) {
  if (!text || text.length < 20) return [];
  if (LOW_SIGNAL_RESPONSE.some((p) => p.test(text.trim()))) return [];

  const cleaned = cleanForScanning(text);
  const claims = [];

  for (const { pattern, type } of NEGATIVE_CLAIM_PATTERNS) {
    const match = cleaned.match(pattern);
    if (match) {
      // Extract surrounding context (±50 chars)
      const idx = cleaned.indexOf(match[0]);
      const start = Math.max(0, idx - 50);
      const end = Math.min(cleaned.length, idx + match[0].length + 50);
      const context = cleaned.slice(start, end).trim();
      claims.push({ type, direction: "negative", match: match[0], context });
    }
  }

  for (const { pattern, type } of POSITIVE_CLAIM_PATTERNS) {
    const match = cleaned.match(pattern);
    if (match) {
      const idx = cleaned.indexOf(match[0]);
      const start = Math.max(0, idx - 50);
      const end = Math.min(cleaned.length, idx + match[0].length + 50);
      const context = cleaned.slice(start, end).trim();
      claims.push({ type, direction: "positive", match: match[0], context });
    }
  }

  for (const { pattern, type } of CONFIG_CLAIM_PATTERNS) {
    const match = cleaned.match(pattern);
    if (match) {
      // Only flag credential claims, not general hex (commit hashes etc.)
      if (type === "config_claim") {
        // Skip if it looks like a git commit hash in context
        const surrounding = cleaned.slice(
          Math.max(0, cleaned.indexOf(match[0]) - 30),
          cleaned.indexOf(match[0]) + match[0].length + 30
        );
        if (/(?:commit|sha|hash|ref|merge|cherry)/i.test(surrounding)) continue;
      }
      claims.push({ type, direction: "config", match: match[0].slice(0, 40), context: match[0].slice(0, 80) });
    }
  }

  return claims;
}

// ── Main Hook ──

const claimScannerHook = async (event) => {
  if (event.type !== "message" || event.action !== "sent") return;

  const rawContent = String(event?.context?.content || "");
  const content = stripChannelEnvelope(rawContent).trim();

  if (!content || content.length < 30) return;

  const claims = detectClaims(content);
  if (claims.length === 0) return;

  const sessionKey = event?.sessionKey || "";
  const workspaceDir = resolveWorkspaceDir(event);
  const agentId = resolveAgentId(event, workspaceDir);

  // Check if there were recent tool calls that could serve as evidence
  const hasEvidence = hasRecentVerificationTool(sessionKey);

  if (hasEvidence) {
    console.warn(`[claim-scanner] ${claims.length} claims found but tool evidence exists (agent=${agentId})`);
    return; // Claims are backed by tool calls — OK
  }

  console.warn(`[claim-scanner] ${claims.length} UNVERIFIED claims detected (agent=${agentId})`);

  // Store each unverified claim
  for (const claim of claims.slice(0, 3)) { // Max 3 per response
    // Store to NoldoMem
    if (_memoryApiKey) {
      try {
        await fetch(`${MEMORY_API}/store`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
          body: JSON.stringify({
            text: `[Unverified Claim] ${claim.direction}: "${claim.context}" — No tool verification found in session. Agent should verify with grep/cat/curl before making such claims.`,
            category: "unverified_claim",
            importance: 0.80,
            agent: agentId,
            source: "claim-scanner-hook",
            memory_type: "lesson",
          }),
          signal: AbortSignal.timeout(SCAN_TIMEOUT_MS),
        });
      } catch (e) {
        console.warn("[claim-scanner] store failed:", e.message);
      }
    }

    // Log to fabrication-log.json
    appendFabricationIncident(workspaceDir, {
      date: new Date().toISOString(),
      type: claim.type,
      claim: claim.context.slice(0, 200),
      direction: claim.direction,
      rootCause: "no_tool_call",
      agent: agentId,
      sessionKey: sessionKey.slice(0, 100),
    });
  }

  // Optionally inject warning (configurable via env)
  if (process.env.NOLDO_CLAIM_WARNING === "1" && event.messages && Array.isArray(event.messages)) {
    event.messages.push(
      `⚠️ [claim-scanner] ${claims.length} unverified claim(s) detected in your response. ` +
      `No tool call (grep/cat/curl) found for verification. Consider re-checking.`
    );
  }
};

export default claimScannerHook;
