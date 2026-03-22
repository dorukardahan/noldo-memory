/**
 * Shared state between hooks — tool call tracking + fabrication log.
 * Uses temp file for cross-hook communication since hooks may not share module cache.
 *
 * Created: 2026-03-22 [Fabrication Prevention System]
 */

import fs from "node:fs";
import path from "node:path";

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

    fs.writeFileSync(TOOL_STATE_FILE, JSON.stringify(state), "utf-8");
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

/**
 * Check if any recent tool calls match verification patterns
 * (grep, find, cat, curl, exec with specific patterns).
 * @param {string} sessionKey
 * @returns {boolean}
 */
export function hasRecentVerificationTool(sessionKey) {
  const calls = getRecentToolCalls(sessionKey, 120_000);
  if (calls.length === 0) return false;
  const verifyPatterns = [
    /\bgrep\b/i,
    /\bfind\b/i,
    /\bcat\b/i,
    /\bcurl\b/i,
    /\bls\b/i,
    /\bread\b/i,
    /\bhead\b/i,
    /\btail\b/i,
    /\bwc\b/i,
    /\bfile\b/i,
    /\btest\s+-/i,
    /\bstat\b/i,
  ];
  return calls.some((c) =>
    c.tool === "exec" || c.tool === "read" || c.tool === "web_search" || c.tool === "web_fetch" ||
    verifyPatterns.some((p) => p.test(c.cmd))
  );
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

    fs.writeFileSync(logPath, JSON.stringify(log, null, 2), "utf-8");
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

    // Prune expired facts (mark as stale, don't delete)
    const now = Date.now();
    for (const [k, v] of Object.entries(data.facts)) {
      const verifiedAt = new Date(v.verifiedAt || 0).getTime();
      const ttlMs = (v.ttl || 604800) * 1000;
      if (now - verifiedAt > ttlMs) {
        v.stale = true;
      }
    }

    fs.writeFileSync(factsPath, JSON.stringify(data, null, 2), "utf-8");
    console.warn(`[shared-state] verified fact stored: ${key}`);
  } catch (err) {
    console.warn("[shared-state] writeVerifiedFact error:", err.message);
  }
}
