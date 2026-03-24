/**
 * Shared state between hooks — tool call tracking + fabrication log.
 * Uses temp file for cross-hook communication since hooks may not share module cache.
 *
 * Created: 2026-03-22 [Fabrication Prevention System]
 */

import fs from "node:fs";
import path from "node:path";
import { atomicWrite, withFileLock } from "./util.js";

const TOOL_STATE_FILE = "/tmp/noldo-recent-tools.json";
const TOOL_CALL_TTL_MS = 300_000; // 5 minutes

/**
 * Record a tool call for later evidence checking.
 * @param {string} sessionKey
 * @param {string} toolName
 * @param {string} command - The command or file path
 */
export function recordToolCall(sessionKey, toolName, command) {
  try {
    withFileLock(TOOL_STATE_FILE, () => {
      let state = {};
      try {
        state = JSON.parse(fs.readFileSync(TOOL_STATE_FILE, "utf-8"));
      } catch { /* empty or missing */ }

      if (!state.calls) state.calls = {};
      if (!state.calls[sessionKey]) state.calls[sessionKey] = [];

      state.calls[sessionKey].push({
        tool: toolName,
        cmd: (command || "").slice(0, 300),
        ts: Date.now(),
      });

      // Keep only last 20 calls per session, prune old sessions
      const now = Date.now();
      for (const key of Object.keys(state.calls)) {
        state.calls[key] = state.calls[key]
          .filter((c) => now - c.ts < TOOL_CALL_TTL_MS)
          .slice(-20);
        if (state.calls[key].length === 0) delete state.calls[key];
      }

      atomicWrite(TOOL_STATE_FILE, JSON.stringify(state));
    });
  } catch (err) {
    console.warn("[shared-state] recordToolCall error:", err.message);
  }
}

/**
 * Get recent tool calls for a session (within windowMs).
 * @param {string} sessionKey
 * @param {number} windowMs - How far back to look (default 2 min)
 * @returns {Array<{tool: string, cmd: string, ts: number}>}
 */
export function getRecentToolCalls(sessionKey, windowMs = 120_000) {
  try {
    const state = JSON.parse(fs.readFileSync(TOOL_STATE_FILE, "utf-8"));
    const calls = state.calls?.[sessionKey] || [];
    const cutoff = Date.now() - windowMs;
    return calls.filter((c) => c.ts > cutoff);
  } catch {
    return [];
  }
}

// ── Verification Gate: Proof Type Table (MAST P0-7) ──
// Maps claim categories to required proof tools/patterns.
// A claim is "verified" only if the matching proof type was used.

const PROOF_TYPE_TABLE = {
  completion_claim: {
    // Completion claims need actual work evidence (edit/write/exec)
    requiredTools: ["edit", "write", "exec"],
    requiredPatterns: [/\b(?:edit|write|sed|cp|mv|tee|echo\s+.*>)\b/i],
    description: "Completion claims require edit/write/exec proof",
  },
  feature_claim: {
    // Feature existence claims need read/search/grep evidence
    requiredTools: ["read", "exec", "web_search", "web_fetch"],
    requiredPatterns: [/\b(?:grep|find|cat|ls|which|dpkg|npm\s+list|pip\s+list)\b/i],
    description: "Feature claims require read/grep/search proof",
  },
  config_claim: {
    // Config claims need file read or status check
    requiredTools: ["read", "exec"],
    requiredPatterns: [/\b(?:cat|grep|head|tail|jq|yq)\b/i],
    description: "Config claims require file read proof",
  },
  credential_claim: {
    // Credential claims need careful handling
    requiredTools: ["read", "exec"],
    requiredPatterns: [/\b(?:cat|grep|curl|openssl|test\s+-f)\b/i],
    description: "Credential claims require read/test proof",
  },
};

/**
 * Check if any recent tool calls match verification patterns
 * AND are relevant to the given claim keywords.
 * @param {string} sessionKey
 * @param {string[]} claimKeywords - Keywords extracted from the claim context
 * @param {string} claimType - Optional: specific claim type for proof-type matching
 * @returns {boolean}
 */
export function hasRecentVerificationTool(sessionKey, claimKeywords = [], claimType = null) {
  // Use type-specific windows: completions take longer than lookups (review H-5)
  const windowMs = claimType === "completion_claim" ? 300_000 : 120_000;
  const calls = getRecentToolCalls(sessionKey, windowMs);
  if (calls.length === 0) return false;

  // General verification patterns (fallback)
  const verifyPatterns = [
    /\bgrep\b/i,
    /\bfind\b/i,
    /\bcat\b/i,
    /\bcurl\b/i,
    /\bhead\b/i,
    /\btail\b/i,
    /\btest\s+-/i,
    /\bstat\b/i,
    /\bwhich\b/i,
    /\bdpkg\b/i,
    /\bsystemctl\s+status\b/i,
  ];

  const isVerificationCall = (c) => {
    // read and web_search/web_fetch are always verification-shaped
    if (c.tool === "read" || c.tool === "web_search" || c.tool === "web_fetch") return true;
    // edit and write are proof of completion
    if (c.tool === "edit" || c.tool === "write") return true;
    // exec must match a verification pattern (not just any command)
    if (c.tool === "exec") return verifyPatterns.some((p) => p.test(c.cmd));
    return false;
  };

  // If a specific claim type is provided, use proof-type table for stricter matching
  if (claimType && PROOF_TYPE_TABLE[claimType]) {
    const proof = PROOF_TYPE_TABLE[claimType];
    const matchingCalls = calls.filter((c) => {
      // Check if tool is in required list
      if (proof.requiredTools.includes(c.tool)) {
        // For exec, also check command patterns
        if (c.tool === "exec") {
          return proof.requiredPatterns.some((p) => p.test(c.cmd));
        }
        return true;
      }
      return false;
    });

    if (matchingCalls.length === 0) return false;

    // If no keywords, just having the right tool type is enough
    if (!claimKeywords || claimKeywords.length === 0) return true;

    // Check keyword relevance
    const keywords = claimKeywords.map((k) => k.toLowerCase());
    return matchingCalls.some((c) => {
      const cmdLower = (c.cmd || "").toLowerCase();
      return keywords.some((kw) => cmdLower.includes(kw));
    });
  }

  // Fallback: general verification check (backwards compatible)
  const verifyCalls = calls.filter(isVerificationCall);
  if (verifyCalls.length === 0) return false;

  if (!claimKeywords || claimKeywords.length === 0) return true;

  const keywords = claimKeywords.map((k) => k.toLowerCase());
  return verifyCalls.some((c) => {
    const cmdLower = (c.cmd || "").toLowerCase();
    return keywords.some((kw) => cmdLower.includes(kw));
  });
}

/**
 * Get the proof type description for a claim type.
 */
export function getProofRequirement(claimType) {
  return PROOF_TYPE_TABLE[claimType] || null;
}

// ── Fabrication Log ──

/**
 * Read fabrication log from workspace.
 * @param {string} workspaceDir
 * @returns {object}
 */
export function readFabricationLog(workspaceDir) {
  const logPath = path.join(workspaceDir, "fabrication-log.json");
  try {
    return JSON.parse(fs.readFileSync(logPath, "utf-8"));
  } catch {
    return { incidents: [], stats: { total: 0, last7d: 0, byType: {}, byRootCause: {} } };
  }
}

/**
 * Append a fabrication incident and update stats.
 * @param {string} workspaceDir
 * @param {object} incident
 */
export function appendFabricationIncident(workspaceDir, incident) {
  try {
    const logPath = path.join(workspaceDir, "fabrication-log.json");
    const log = readFabricationLog(workspaceDir);

    log.incidents.push(incident);

    // Keep last 100 incidents
    if (log.incidents.length > 100) {
      log.incidents = log.incidents.slice(-100);
    }

    // Recompute stats
    const now = Date.now();
    const week = 7 * 24 * 60 * 60 * 1000;
    const byType = {};
    const byRootCause = {};
    let last7d = 0;

    for (const inc of log.incidents) {
      const incDate = new Date(inc.date || 0).getTime();
      if (now - incDate < week) last7d++;
      byType[inc.type] = (byType[inc.type] || 0) + 1;
      byRootCause[inc.rootCause] = (byRootCause[inc.rootCause] || 0) + 1;
    }

    log.stats = {
      total: log.incidents.length,
      last7d,
      byType,
      byRootCause,
    };

    atomicWrite(logPath, JSON.stringify(log, null, 2));
    console.warn(`[shared-state] fabrication incident logged: ${incident.type} (total=${log.stats.total})`);
  } catch (err) {
    console.warn("[shared-state] appendFabricationIncident error:", err.message);
  }
}

// ── Verified Facts ──

/**
 * Read verified facts from workspace.
 * @param {string} workspaceDir
 * @returns {object}
 */
export function readVerifiedFacts(workspaceDir) {
  const factsPath = path.join(workspaceDir, "verified-facts.json");
  try {
    return JSON.parse(fs.readFileSync(factsPath, "utf-8"));
  } catch {
    return { facts: {} };
  }
}

/**
 * Write a verified fact.
 * @param {string} workspaceDir
 * @param {string} key - Slugified fact key
 * @param {object} fact - { claim, verified, source, agent, ttl }
 */
export function writeVerifiedFact(workspaceDir, key, fact) {
  try {
    const factsPath = path.join(workspaceDir, "verified-facts.json");
    const data = readVerifiedFacts(workspaceDir);

    data.facts[key] = {
      ...fact,
      verifiedAt: new Date().toISOString(),
      ttl: fact.ttl || 604800, // 7 days default
    };

    // Prune expired facts: mark stale after TTL, delete after 30 days
    const now = Date.now();
    const HARD_DELETE_MS = 30 * 24 * 60 * 60 * 1000; // 30 days
    const MAX_FACTS = 200;
    for (const [k, v] of Object.entries(data.facts)) {
      const verifiedAt = new Date(v.verifiedAt || 0).getTime();
      const ttlMs = (v.ttl || 604800) * 1000;
      if (now - verifiedAt > HARD_DELETE_MS) {
        delete data.facts[k]; // Hard delete after 30 days
      } else if (now - verifiedAt > ttlMs) {
        v.stale = true;
      }
    }
    // Cap at MAX_FACTS — keep newest
    const factEntries = Object.entries(data.facts);
    if (factEntries.length > MAX_FACTS) {
      const sorted = factEntries.sort((a, b) =>
        new Date(b[1].verifiedAt || 0).getTime() - new Date(a[1].verifiedAt || 0).getTime()
      );
      data.facts = Object.fromEntries(sorted.slice(0, MAX_FACTS));
    }

    atomicWrite(factsPath, JSON.stringify(data, null, 2));
    console.warn(`[shared-state] verified fact stored: ${key}`);
  } catch (err) {
    console.warn("[shared-state] writeVerifiedFact error:", err.message);
  }
}

// ── Fabrication Score (per-session) ── [MAST P0]

const FAB_SCORE_FILE = "/tmp/noldo-fab-scores.json";

/**
 * Increment fabrication score for a session.
 * @param {string} sessionKey
 * @returns {number} current score after increment
 */
export function incrementFabricationScore(sessionKey) {
  if (!sessionKey) return 0; // Skip empty keys to avoid "" pollution
  try {
    return withFileLock(FAB_SCORE_FILE, () => {
      let scores = {};
      try { scores = JSON.parse(fs.readFileSync(FAB_SCORE_FILE, "utf-8")); } catch { /* empty */ }
      if (!scores[sessionKey]) scores[sessionKey] = { count: 0, first: Date.now() };
      scores[sessionKey].count++;
      scores[sessionKey].last = Date.now();

      // Prune sessions older than 24h
      const cutoff = Date.now() - 24 * 60 * 60 * 1000;
      for (const key of Object.keys(scores)) {
        if ((scores[key].last || 0) < cutoff) delete scores[key];
      }

      atomicWrite(FAB_SCORE_FILE, JSON.stringify(scores));
      return scores[sessionKey].count;
    });
  } catch (err) {
    console.warn("[shared-state] incrementFabricationScore error:", err.message);
    return 0;
  }
}

/**
 * Get current fabrication score for a session.
 * @param {string} sessionKey
 * @returns {number}
 */
export function getFabricationScore(sessionKey) {
  try {
    const scores = JSON.parse(fs.readFileSync(FAB_SCORE_FILE, "utf-8"));
    return scores[sessionKey]?.count || 0;
  } catch {
    return 0;
  }
}

// ── Verification Required State ── [MAST P0]

/**
 * Set verification-required flag for a session.
 */
export function setVerificationRequired(sessionKey, data) {
  try {
    let state = {};
    try { state = JSON.parse(fs.readFileSync(TOOL_STATE_FILE, "utf-8")); } catch { /* empty */ }
    if (!state.verificationRequired) state.verificationRequired = {};
    state.verificationRequired[sessionKey] = {
      pending: true,
      ...data,
      setAt: Date.now(),
    };
    atomicWrite(TOOL_STATE_FILE, JSON.stringify(state));
  } catch (err) {
    console.warn("[shared-state] setVerificationRequired error:", err.message);
  }
}

/**
 * Clear verification flag when proof is provided.
 */
export function clearVerificationIfProved(sessionKey) {
  try {
    let state = {};
    try { state = JSON.parse(fs.readFileSync(TOOL_STATE_FILE, "utf-8")); } catch { /* empty */ }
    if (state.verificationRequired?.[sessionKey]) {
      delete state.verificationRequired[sessionKey];
      atomicWrite(TOOL_STATE_FILE, JSON.stringify(state));
    }
  } catch (err) {
    console.warn("[shared-state] clearVerificationIfProved error:", err.message);
  }
}

/**
 * Get verification state for a session.
 */
export function getVerificationState(sessionKey) {
  try {
    const state = JSON.parse(fs.readFileSync(TOOL_STATE_FILE, "utf-8"));
    return state.verificationRequired?.[sessionKey] || null;
  } catch {
    return null;
  }
}
