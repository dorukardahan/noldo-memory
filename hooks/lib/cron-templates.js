/**
 * Standardized Cron Templates — 3-tier job templates for OpenClaw cron.
 * Prevents scheduling inconsistency (MAST FM-2.3).
 *
 * Tiers:
 *   1. Health Check — lightweight, frequent (1-4h)
 *   2. Maintenance — medium, daily/weekly
 *   3. Intelligence — heavy, custom schedule
 *
 * Created: 2026-03-24 [MAST P2]
 */

/** Sanitize a string for safe embedding in agent instructions. */
function sanitize(str, maxLen = 200) {
  return String(str || "").replace(/[`$(){}\\]/g, "").slice(0, maxLen);
}

/** Clamp interval to safe range (minimum 0.5h to prevent busy-loop). */
function safeInterval(hours) {
  const h = Number(hours);
  if (!h || h < 0.5) return 0.5;
  return h;
}

/**
 * Tier 1: Health Check templates.
 * Quick checks that should run frequently and fail gracefully.
 */
export const HEALTH_CHECK = {
  /** Service health endpoint check */
  serviceHealth: (name, url, intervalHours = 2) => ({
    name: `${sanitize(name)} Health Check`,
    schedule: { kind: "every", everyMs: safeInterval(intervalHours) * 60 * 60 * 1000 },
    payload: {
      kind: "agentTurn",
      message: [
        `Health check for ${sanitize(name)}.`,
        `1. Check if ${sanitize(url, 500)} is reachable (curl with 10s timeout)`,
        `2. Verify response status is 200`,
        `3. If unhealthy: report the error clearly`,
        `4. If healthy: reply with a single line "✅ ${sanitize(name)} healthy"`,
        `Do NOT write files or make changes. Read-only check.`,
      ].join("\n"),
      timeoutSeconds: 120,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "health",
  }),

  /** OAuth token expiry check */
  tokenExpiry: (provider, intervalHours = 4) => ({
    name: `${sanitize(provider)} Token Expiry Check`,
    schedule: { kind: "every", everyMs: safeInterval(intervalHours) * 60 * 60 * 1000 },
    payload: {
      kind: "agentTurn",
      message: [
        `Check ${sanitize(provider)} OAuth token expiry.`,
        `1. Read the relevant auth-profiles.json / credentials file`,
        `2. Parse token expiry timestamps`,
        `3. If any token expires within 24h: WARN with details`,
        `4. If all tokens valid: reply "✅ ${sanitize(provider)} tokens valid"`,
        `Do NOT refresh tokens — only check and report.`,
      ].join("\n"),
      timeoutSeconds: 120,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "health",
  }),

  /** Docker container status */
  dockerHealth: (intervalHours = 2) => ({
    name: "Docker Health Check",
    schedule: { kind: "every", everyMs: safeInterval(intervalHours) * 60 * 60 * 1000 },
    payload: {
      kind: "agentTurn",
      message: [
        "Check Docker container health.",
        "1. Run `docker ps --format '{{.Names}} {{.Status}}'`",
        "2. Flag any containers not in 'Up' state",
        "3. Check for containers restarting frequently",
        "4. If all healthy: reply '✅ All containers healthy'",
      ].join("\n"),
      timeoutSeconds: 120,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "health",
  }),
};

/**
 * Tier 2: Maintenance templates.
 * Medium-weight tasks that run daily or weekly.
 */
export const MAINTENANCE = {
  /** Memory decay and consolidation */
  memoryDecay: (cronExpr = "0 4 * * 0") => ({
    name: "Memory Decay & Consolidation",
    schedule: { kind: "cron", expr: cronExpr, tz: "UTC" },
    payload: {
      kind: "agentTurn",
      message: [
        "Run memory maintenance.",
        "1. Check memory/solved-problems.json — archive entries older than 30 days",
        "2. Review memory/*.md daily files older than 14 days — summarize if needed",
        "3. Run bulletin archiveExpired if bulletin board is in use",
        "4. Report: items archived, items kept, any issues found",
      ].join("\n"),
      timeoutSeconds: 300,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "maintenance",
  }),

  /** Self-improvement audit */
  selfAudit: (cronExpr = "30 7 * * *") => ({
    name: "Daily Self-Improvement Audit",
    schedule: { kind: "cron", expr: cronExpr, tz: "Europe/Istanbul" },
    payload: {
      kind: "agentTurn",
      message: [
        "Daily self-improvement audit.",
        "1. Read fabrication-log.json — count incidents in last 24h",
        "2. Read MAST metrics (/tmp/noldo-mast-metrics.json) — check claim scanner rates",
        "3. Identify top 3 recurring issues",
        "4. Suggest 1 concrete improvement",
        "Keep report under 200 words.",
      ].join("\n"),
      timeoutSeconds: 300,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "maintenance",
  }),

  /** Update watch */
  updateWatch: (name, checkCommand, intervalHours = 4) => ({
    name: `${sanitize(name)} Update Watch`,
    schedule: { kind: "every", everyMs: safeInterval(intervalHours) * 60 * 60 * 1000 },
    payload: {
      kind: "agentTurn",
      message: [
        `Check for ${sanitize(name)} updates.`,
        `Run: ${sanitize(checkCommand, 500)}`,
        `Compare with currently installed version.`,
        `If update available: report version diff and changelog link.`,
        `If current: reply "✅ ${sanitize(name)} up to date"`,
      ].join("\n"),
      timeoutSeconds: 180,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "maintenance",
  }),
};

/**
 * Tier 3: Intelligence templates.
 * Heavy tasks — research, analysis, engagement.
 */
export const INTELLIGENCE = {
  /** Engagement task (social, community) */
  engagement: (name, instructions, cronExpr) => ({
    name: sanitize(name),
    schedule: { kind: "cron", expr: String(cronExpr || "0 0 * * *"), tz: "UTC" },
    payload: {
      kind: "agentTurn",
      message: String(instructions || "").slice(0, 2000),
      timeoutSeconds: 600,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "intelligence",
  }),

  /** Research task */
  research: (name, topic, cronExpr) => ({
    name: sanitize(name),
    schedule: { kind: "cron", expr: String(cronExpr || "0 0 * * *"), tz: "UTC" },
    payload: {
      kind: "agentTurn",
      message: [
        `Research task: ${sanitize(topic, 500)}`,
        "1. Search for recent developments (last 7 days)",
        "2. Summarize key findings",
        "3. Note any actionable items",
        "4. Write findings to /tmp/ with timestamped filename",
        "Keep report under 500 words.",
      ].join("\n"),
      timeoutSeconds: 600,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "intelligence",
  }),
};

/**
 * Validate a cron job config against template standards.
 * @param {object} job - Cron job config
 * @returns {object} { valid, warnings, tier }
 */
export function validateCronJob(job) {
  const warnings = [];
  const errors = [];

  if (!job.name || typeof job.name !== "string" || !job.name.trim()) {
    errors.push("Missing or empty job name");
  }
  if (!job.schedule) {
    errors.push("Missing schedule");
  } else {
    const { kind, expr, everyMs } = job.schedule;
    if (kind === "cron" && (!expr || typeof expr !== "string")) {
      errors.push("Cron schedule missing expr");
    }
    if (kind === "every" && (!everyMs || everyMs < 1800000)) {
      warnings.push("Interval < 30min — may cause excessive runs");
    }
  }
  if (!job.payload) {
    errors.push("Missing payload");
  }
  if (!job.sessionTarget) warnings.push("Missing sessionTarget");

  // Check timeout
  const timeout = job.payload?.timeoutSeconds;
  if (timeout === undefined || timeout === null) {
    warnings.push("No timeout set — job could run indefinitely");
  } else if (timeout === 0) {
    warnings.push("Timeout is 0 — job will be killed immediately");
  } else if (timeout > 1800) {
    warnings.push("Timeout > 30min — consider breaking into smaller tasks");
  }

  // Check delivery
  const validModes = ["announce", "channel", "dm", "none"];
  if (!job.delivery || !job.delivery.mode) {
    warnings.push("No delivery configured — results may be lost");
  } else if (!validModes.includes(job.delivery.mode)) {
    warnings.push(`Unknown delivery mode: ${job.delivery.mode}`);
  }

  // Determine tier (use explicit tier if set, otherwise infer from timeout)
  let tier = "unknown";
  if (job.tier && ["health", "maintenance", "intelligence"].includes(job.tier)) {
    tier = job.tier;
  } else if (typeof timeout === "number") {
    if (timeout < 180) {
      tier = "health";
    } else if (timeout <= 300) {
      tier = "maintenance";
    } else {
      tier = "intelligence";
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    tier,
  };
}

/**
 * Format a cron template as human-readable text.
 */
export function formatTemplate(template) {
  const lines = [];
  lines.push(`**${template.name}**`);
  lines.push(`  Schedule: ${JSON.stringify(template.schedule)}`);
  lines.push(`  Tier: ${template.tier || "unknown"}`);
  lines.push(`  Timeout: ${template.payload?.timeoutSeconds || 0}s`);
  lines.push(`  Delivery: ${template.delivery?.mode || "none"}`);
  return lines.join("\n");
}

export default {
  HEALTH_CHECK,
  MAINTENANCE,
  INTELLIGENCE,
  validateCronJob,
  formatTemplate,
};
