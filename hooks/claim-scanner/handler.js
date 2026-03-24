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
  resolveSessionKey,
  stripChannelEnvelope,
} from "../lib/runtime.js";
import {
  hasRecentVerificationTool,
  appendFabricationIncident,
  incrementFabricationScore,
} from "../lib/shared-state.js";
import { increment as metricsIncrement, recordEvent as metricsEvent } from "../lib/metrics.js";

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

// Kill switch: set MAST_CLAIM_SCANNER_ENABLED=0 to disable entire claim-scanner
const CLAIM_SCANNER_ENABLED = process.env.MAST_CLAIM_SCANNER_ENABLED !== "0";

// ── Claim Detection Patterns ──
// NOTE: Turkish patterns tightened to reduce false positives on casual speech.
// Only technical/operational claims are flagged — not general conversation.

const NEGATIVE_CLAIM_PATTERNS = [
  // Turkish — require technical subject before the claim verb
  { pattern: /(?:destekle(?:miyor|nmez|nmedi)|mevcut\s+değil|bulunmuyor)/i, type: "feature_claim" },
  { pattern: /(?:özellik|feature|endpoint|config|service|hook|plugin|api|port|komut)\s+(?:\w+\s+)?yok\b/i, type: "feature_claim" },
  // Removed overly broad: "çalışmıyor", "mümkün değil", "yapamıyor" — too many FP in bug discussions
  { pattern: /(?:aktif\s+değil|devre\s+dışı|kullanılamaz)\b/i, type: "feature_claim" },
  // English
  { pattern: /(?:not\s+support(?:ed)?|doesn'?t\s+(?:have|support|exist|include))/i, type: "feature_claim" },
  { pattern: /(?:not\s+available|not\s+(?:possible|implemented|enabled))/i, type: "feature_claim" },
  { pattern: /(?:there\s+is\s+no|there\s+are\s+no|no\s+(?:support|way)\s+for)/i, type: "feature_claim" },
  { pattern: /(?:cannot|can'?t\s+(?:do|handle|use|run))/i, type: "feature_claim" },
];

const POSITIVE_CLAIM_PATTERNS = [
  // Turkish — removed "yaptım" (too casual), kept specific completion verbs
  { pattern: /(?:düzelttim|güncelledim|tamamlandı|hallettim|ekledim)/i, type: "completion_claim" },
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
  if (!CLAIM_SCANNER_ENABLED) return;
  if (event.type !== "message" || event.action !== "sent") return;

  const rawContent = String(event?.context?.content || "");
  const content = stripChannelEnvelope(rawContent).trim();

  if (!content || content.length < 30) return;

  const claims = detectClaims(content);
  if (claims.length === 0) return;

  const sessionKey = resolveSessionKey(event);
  const workspaceDir = resolveWorkspaceDir(event);
  const agentId = resolveAgentId(event, workspaceDir);

  // Extract keywords from claim contexts for targeted evidence matching
  const claimKeywords = claims
    .flatMap((c) => (c.context || "").split(/\s+/))
    .filter((w) => w.length > 3)
    .map((w) => w.toLowerCase().replace(/[^a-z0-9çğıöşü]/gi, ""))
    .filter(Boolean);
  const uniqueKeywords = [...new Set(claimKeywords)].slice(0, 10);

  // Check if there were recent tool calls relevant to these specific claims
  // Use typed proof matching: each claim type requires specific evidence
  const unverifiedClaims = claims.filter((claim) => {
    const hasEvidence = hasRecentVerificationTool(sessionKey, uniqueKeywords, claim.type);
    return !hasEvidence;
  });

  // Metrics: track total scanned claims
  try { metricsIncrement("claim_scanner.total", claims.length); } catch (e) { console.warn("[claim-scanner] metrics error:", e.message); }

  if (unverifiedClaims.length === 0) {
    console.warn(`[claim-scanner] ${claims.length} claims found but all have matching proof evidence (agent=${agentId})`);
    try { metricsIncrement("claim_scanner.verified", claims.length); } catch (e) { console.warn("[claim-scanner] metrics error:", e.message); }
    return; // All claims are backed by appropriate tool calls — OK
  }

  // Metrics: track unverified claims
  try {
    metricsIncrement("claim_scanner.unverified", unverifiedClaims.length);
    metricsEvent("unverified_claims", { agent: agentId, count: unverifiedClaims.length, types: unverifiedClaims.map(c => c.type) });
  } catch (e) { console.warn("[claim-scanner] metrics error:", e.message); }

  console.warn(`[claim-scanner] ${unverifiedClaims.length} UNVERIFIED claims detected (agent=${agentId})`);

  const claimsToProcess = unverifiedClaims.slice(0, 3);

  // Fire-and-forget: store all claims in parallel with shared deadline
  const deadline = AbortSignal.timeout(SCAN_TIMEOUT_MS);

  // NoldoMem stores — parallel, not sequential
  if (_memoryApiKey) {
    const storePromises = claimsToProcess.map((claim) =>
      fetch(`${MEMORY_API}/store`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
        body: JSON.stringify({
          text: `[Unverified Claim] ${claim.direction}: "${claim.context}" — No tool verification found in session. Agent should verify with grep/cat/curl before making such claims.`,
          category: "unverified_claim",
          importance: 0.80,
          agent: agentId,
          source: "claim-scanner-hook",
          memory_type: "fabrication_incident",
        }),
        signal: deadline,
      }).catch((e) => console.warn("[claim-scanner] store failed:", e.message))
    );
    // Don't await individually — allSettled with shared deadline
    await Promise.allSettled(storePromises);
  }

  // Log to fabrication-log.json (sync file I/O, fast)
  for (const claim of claimsToProcess) {
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

  // ── Fabrication score tracking (enforcement moved to bootstrap-context — C-2 fix) ──
  // claim-scanner fires on message:sent (post-delivery), so event.messages.push()
  // was dead code — warnings never reached the agent. Now enforcement lives in
  // bootstrap-context which runs pre-response.
  incrementFabricationScore(sessionKey);
};

export default claimScannerHook;
