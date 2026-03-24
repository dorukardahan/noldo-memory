/**
 * MAST Metrics — Lightweight observability for hook system.
 * Tracks claim-scanner FP rate, ATS/SPR usage, and hook performance.
 *
 * File-based: /tmp/noldo-mast-metrics.json
 * Created: 2026-03-24 [MAST P1]
 */

import fs from "node:fs";
import { atomicWrite } from "./util.js";

const METRICS_PATH = process.env.MAST_METRICS_PATH || "/tmp/noldo-mast-metrics.json";
const MAX_HISTORY = 1000; // Keep last N events

function readMetrics() {
  try {
    return JSON.parse(fs.readFileSync(METRICS_PATH, "utf-8"));
  } catch {
    return {
      version: 1,
      counters: {},
      events: [],
      lastReset: new Date().toISOString(),
    };
  }
}

function writeMetrics(m) {
  try {
    atomicWrite(METRICS_PATH, JSON.stringify(m, null, 2));
  } catch (err) {
    console.warn("[metrics] write error:", err.message);
  }
}

/**
 * Increment a named counter.
 * @param {string} name - Counter name (e.g., "claim_scanner.total", "claim_scanner.fp")
 * @param {number} delta - Amount to increment (default 1)
 */
export function increment(name, delta = 1) {
  const m = readMetrics();
  m.counters[name] = (m.counters[name] || 0) + delta;
  writeMetrics(m);
}

/**
 * Record a timestamped event.
 * @param {string} type - Event type
 * @param {object} data - Event data
 */
export function recordEvent(type, data = {}) {
  const m = readMetrics();
  m.events.push({
    type,
    ts: Date.now(),
    ...data,
  });
  // Cap events
  if (m.events.length > MAX_HISTORY) {
    m.events = m.events.slice(-MAX_HISTORY);
  }
  writeMetrics(m);
}

/**
 * Get current metrics snapshot.
 * @returns {object} { counters, rates, recentEvents }
 */
export function getSnapshot() {
  const m = readMetrics();

  // Calculate rates
  const rates = {};
  const total = m.counters["claim_scanner.total"] || 0;
  const unverified = m.counters["claim_scanner.unverified"] || 0;
  const fp = m.counters["claim_scanner.false_positive"] || 0;

  if (total > 0) {
    rates["claim_scanner.unverified_rate"] = Math.round((unverified / total) * 100) / 100;
    rates["claim_scanner.fp_rate"] = Math.round((fp / total) * 100) / 100;
  }

  const atsCreated = m.counters["ats.created"] || 0;
  const atsCompleted = m.counters["ats.completed"] || 0;
  if (atsCreated > 0) {
    rates["ats.completion_rate"] = Math.round((atsCompleted / atsCreated) * 100) / 100;
  }

  // Recent events (last 20)
  const recentEvents = m.events.slice(-20).reverse();

  return {
    counters: m.counters,
    rates,
    recentEvents,
    lastReset: m.lastReset,
    totalEvents: m.events.length,
  };
}

/**
 * Reset all metrics.
 */
export function reset() {
  writeMetrics({
    version: 1,
    counters: {},
    events: [],
    lastReset: new Date().toISOString(),
  });
}

/**
 * Format metrics as human-readable text.
 */
export function formatMetrics() {
  const snap = getSnapshot();
  const lines = ["# MAST Metrics", ""];

  lines.push("## Counters");
  for (const [k, v] of Object.entries(snap.counters).sort()) {
    lines.push(`  ${k}: ${v}`);
  }

  lines.push("");
  lines.push("## Rates");
  for (const [k, v] of Object.entries(snap.rates).sort()) {
    lines.push(`  ${k}: ${(v * 100).toFixed(1)}%`);
  }

  lines.push("");
  lines.push("## Recent Events");
  for (const e of snap.recentEvents.slice(0, 10)) {
    const ts = new Date(e.ts).toISOString().slice(11, 19);
    lines.push(`  ${ts} ${e.type} ${JSON.stringify(e).slice(0, 100)}`);
  }

  return lines.join("\n");
}

export default { increment, recordEvent, getSnapshot, reset, formatMetrics };
