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

/**
 * Tier 1: Health Check templates.
 * Quick checks that should run frequently and fail gracefully.
 */
export const HEALTH_CHECK = {
  /** Service health endpoint check */
  serviceHealth: (name, url, intervalHours = 2) => ({
    name: `${name} Health Check`,
    schedule: { kind: "every", everyMs: intervalHours * 60 * 60 * 1000 },
    payload: {
      kind: "agentTurn",
      message: [
        `Health check for ${name}.`,
        `1. Check if ${url} is reachable (curl with 10s timeout)`,
        `2. Verify response status is 200`,
        `3. If unhealthy: report the error clearly`,
        `4. If healthy: reply with a single line "✅ ${name} healthy"`,
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
    name: `${provider} Token Expiry Check`,
    schedule: { kind: "every", everyMs: intervalHours * 60 * 60 * 1000 },
    payload: {
      kind: "agentTurn",
      message: [
        `Check ${provider} OAuth token expiry.`,
        `1. Read the relevant auth-profiles.json / credentials file`,
        `2. Parse token expiry timestamps`,
        `3. If any token expires within 24h: WARN with details`,
        `4. If all tokens valid: reply "✅ ${provider} tokens valid"`,
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
    schedule: { kind: "every", everyMs: intervalHours * 60 * 60 * 1000 },
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
    name: `${name} Update Watch`,
    schedule: { kind: "every", everyMs: intervalHours * 60 * 60 * 1000 },
    payload: {
      kind: "agentTurn",
      message: [
        `Check for ${name} updates.`,
        `Run: ${checkCommand}`,
        `Compare with currently installed version.`,
        `If update available: report version diff and changelog link.`,
        `If current: reply "✅ ${name} up to date"`,
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
    name,
    schedule: { kind: "cron", expr: cronExpr, tz: "UTC" },
    payload: {
      kind: "agentTurn",
      message: instructions,
      timeoutSeconds: 600,
    },
    sessionTarget: "isolated",
    delivery: { mode: "announce" },
    tier: "intelligence",
  }),

  /** Research task */
  research: (name, topic, cronExpr) => ({
    name,
    schedule: { kind: "cron", expr: cronExpr, tz: "UTC" },
    payload: {
      kind: "agentTurn",
      message: [
        `Research task: ${topic}`,
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

  if (!job.name) warnings.push("Missing job name");
  if (!job.schedule) warnings.push("Missing schedule");
  if (!job.payload) warnings.push("Missing payload");
  if (!job.sessionTarget) warnings.push("Missing sessionTarget");

  // Check timeout
  const timeout = job.payload?.timeoutSeconds || 0;
  if (timeout === 0) warnings.push("No timeout set — job could run indefinitely");
  if (timeout > 1800) warnings.push("Timeout > 30min — consider breaking into smaller tasks");

  // Check delivery
  if (!job.delivery || job.delivery.mode === "none") {
    warnings.push("No delivery configured — results may be lost");
  }

  // Determine tier
  let tier = "unknown";
  if (job.tier) {
    tier = job.tier;
  } else if (timeout <= 180) {
    tier = "health";
  } else if (timeout <= 300) {
    tier = "maintenance";
  } else {
    tier = "intelligence";
  }

  return {
    valid: warnings.length === 0,
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
