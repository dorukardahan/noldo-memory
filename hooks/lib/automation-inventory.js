/**
 * Automation Inventory — 6-layer discovery for workspace automation.
 * Prevents "execution layer blindspot" (MAST FM-2.1).
 *
 * Layers:
 *   1. OpenClaw cron jobs
 *   2. System crontab
 *   3. PM2 processes
 *   4. Systemd services
 *   5. Docker containers
 *   6. Custom queue/scheduler (SQLite, scripts)
 *
 * Created: 2026-03-24 [MAST P1]
 */

import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const TIMEOUT = 5000;

function tryExec(cmd) {
  try {
    return execSync(cmd, { timeout: TIMEOUT, encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

/**
 * Layer 1: OpenClaw cron jobs
 * Reads from gateway cron API or falls back to file scan.
 */
function discoverOpenClawCrons(workspaceDir) {
  const results = [];
  // Try gateway API
  const port = process.env.OPENCLAW_GATEWAY_PORT || "18789";
  const raw = tryExec(`curl -s http://127.0.0.1:${port}/api/cron/list 2>/dev/null`);
  if (raw) {
    try {
      const data = JSON.parse(raw);
      const jobs = data.jobs || data || [];
      for (const job of Array.isArray(jobs) ? jobs : []) {
        results.push({
          layer: "openclaw-cron",
          id: job.id || job.jobId,
          name: job.name || "(unnamed)",
          schedule: job.schedule?.expr || job.schedule?.kind || "unknown",
          enabled: job.enabled !== false,
          agent: job.sessionTarget || "unknown",
        });
      }
    } catch { /* parse error, skip */ }
  }
  return results;
}

/**
 * Layer 2: System crontab entries
 */
function discoverCrontab() {
  const results = [];
  const crontab = tryExec("crontab -l 2>/dev/null");
  if (crontab) {
    for (const line of crontab.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      results.push({
        layer: "crontab",
        schedule: trimmed.split(/\s+/).slice(0, 5).join(" "),
        command: trimmed.split(/\s+/).slice(5).join(" "),
        enabled: true,
      });
    }
  }
  return results;
}

/**
 * Layer 3: PM2 processes
 */
function discoverPM2() {
  const results = [];
  const raw = tryExec("pm2 jlist 2>/dev/null");
  if (raw) {
    try {
      const procs = JSON.parse(raw);
      for (const p of procs) {
        results.push({
          layer: "pm2",
          id: p.pm_id,
          name: p.name,
          status: p.pm2_env?.status || "unknown",
          script: p.pm2_env?.pm_exec_path || "unknown",
          restarts: p.pm2_env?.restart_time || 0,
        });
      }
    } catch { /* parse error */ }
  }
  return results;
}

/**
 * Layer 4: Systemd services (user + relevant system)
 */
function discoverSystemd(keywords = []) {
  const results = [];
  // User services
  const userUnits = tryExec("systemctl --user list-units --type=service --no-pager --no-legend 2>/dev/null");
  if (userUnits) {
    for (const line of userUnits.split("\n")) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 4) {
        results.push({
          layer: "systemd-user",
          unit: parts[0],
          status: parts[2], // active/inactive
          sub: parts[3],    // running/dead
        });
      }
    }
  }
  // System services matching keywords
  if (keywords.length > 0) {
    const pattern = keywords.join("\\|");
    const sysUnits = tryExec(`systemctl list-units --type=service --no-pager --no-legend 2>/dev/null | grep -i "${pattern}"`);
    if (sysUnits) {
      for (const line of sysUnits.split("\n")) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 4) {
          results.push({
            layer: "systemd-system",
            unit: parts[0],
            status: parts[2],
            sub: parts[3],
          });
        }
      }
    }
  }
  return results;
}

/**
 * Layer 5: Docker containers
 */
function discoverDocker() {
  const results = [];
  const raw = tryExec('docker ps --format "{{.ID}}|{{.Names}}|{{.Status}}|{{.Image}}" 2>/dev/null');
  if (raw) {
    for (const line of raw.split("\n")) {
      const [id, name, status, image] = line.split("|");
      if (id) {
        results.push({
          layer: "docker",
          id: id.slice(0, 12),
          name,
          status,
          image,
        });
      }
    }
  }
  return results;
}

/**
 * Layer 6: Custom queue/scheduler detection
 * Scans workspace for SQLite DBs, scheduler scripts, queue files.
 */
function discoverCustomQueues(workspaceDir) {
  const results = [];
  if (!workspaceDir || !fs.existsSync(workspaceDir)) return results;

  // Look for SQLite databases
  const sqliteFiles = tryExec(`find "${workspaceDir}" -maxdepth 3 -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" 2>/dev/null`);
  if (sqliteFiles) {
    for (const f of sqliteFiles.split("\n").filter(Boolean)) {
      // Check if it has job/queue tables
      const tables = tryExec(`sqlite3 "${f}" ".tables" 2>/dev/null`);
      const hasQueue = tables && /job|queue|task|schedule/i.test(tables);
      results.push({
        layer: "custom-queue",
        type: "sqlite",
        path: f,
        tables: tables || "(unreadable)",
        isQueue: hasQueue,
      });
    }
  }

  // Look for scheduler/coordinator scripts
  const schedulerFiles = tryExec(`find "${workspaceDir}" -maxdepth 3 -name "*scheduler*" -o -name "*coordinator*" -o -name "*orchestrator*" 2>/dev/null`);
  if (schedulerFiles) {
    for (const f of schedulerFiles.split("\n").filter(Boolean)) {
      results.push({
        layer: "custom-queue",
        type: "script",
        path: f,
      });
    }
  }

  return results;
}

/**
 * Run full 6-layer inventory.
 * @param {object} opts
 * @param {string} opts.workspaceDir - Workspace to scan for custom queues
 * @param {string[]} opts.systemdKeywords - Keywords to filter systemd services
 * @returns {object} Full inventory report
 */
export function runInventory(opts = {}) {
  const { workspaceDir, systemdKeywords = [] } = opts;

  const inventory = {
    timestamp: new Date().toISOString(),
    layers: {
      openclawCron: discoverOpenClawCrons(workspaceDir),
      crontab: discoverCrontab(),
      pm2: discoverPM2(),
      systemd: discoverSystemd(systemdKeywords),
      docker: discoverDocker(),
      customQueue: discoverCustomQueues(workspaceDir),
    },
  };

  // Summary
  const summary = {};
  for (const [layer, items] of Object.entries(inventory.layers)) {
    summary[layer] = items.length;
  }
  inventory.summary = summary;
  inventory.totalItems = Object.values(summary).reduce((a, b) => a + b, 0);

  return inventory;
}

/**
 * Format inventory as human-readable text.
 */
export function formatInventory(inventory) {
  const lines = [`# Automation Inventory (${inventory.timestamp})`, ""];

  for (const [layer, items] of Object.entries(inventory.layers)) {
    lines.push(`## ${layer} (${items.length})`);
    if (items.length === 0) {
      lines.push("  (none)");
    } else {
      for (const item of items) {
        const details = Object.entries(item)
          .filter(([k]) => k !== "layer")
          .map(([k, v]) => `${k}=${v}`)
          .join(", ");
        lines.push(`  - ${details}`);
      }
    }
    lines.push("");
  }

  lines.push(`**Total: ${inventory.totalItems} items**`);
  return lines.join("\n");
}

export default { runInventory, formatInventory };
