/**
 * Bulletin Board — Cross-workspace shared notices.
 * Any agent can post notices; all agents see them at bootstrap.
 * Prevents siloed knowledge (MAST FM-3.1).
 *
 * Directories:
 *   high/   — urgent notices (shown first)
 *   normal/ — informational notices
 *   archive/ — expired notices (auto-moved)
 *
 * Created: 2026-03-24 [MAST P1]
 */

import fs from "node:fs";
import path from "node:path";
import { atomicWrite } from "./util.js";

const BULLETIN_ROOT = process.env.MAST_BULLETIN_DIR || "/opt/openclaw/shared/bulletin";
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

/**
 * Post a notice to the bulletin board.
 * @param {object} opts
 * @param {string} opts.title - Short title
 * @param {string} opts.body - Notice content
 * @param {"high"|"normal"} opts.priority - Priority tier
 * @param {string} opts.agent - Posting agent
 * @param {number} opts.ttlHours - Time to live in hours (default 168 = 7 days)
 * @returns {string} Notice filename
 */
export function postNotice({ title, body, priority = "normal", agent = "unknown", ttlHours = 168 }) {
  const dir = path.join(BULLETIN_ROOT, priority === "high" ? "high" : "normal");
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const safeAgent = (agent || "unknown").replace(/[^a-zA-Z0-9_.-]/g, "_").slice(0, 30);
  const safeTitle = (title || "notice").replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 40);
  const filename = `${ts}_${safeAgent}_${safeTitle}.md`;
  const filepath = path.join(dir, filename);

  const expiresAt = new Date(Date.now() + ttlHours * 60 * 60 * 1000).toISOString();

  const content = [
    `<!-- bulletin: version=1 agent=${safeAgent} expires=${expiresAt} -->`,
    `**${title}** (${safeAgent}, ${new Date().toISOString().slice(0, 16)})`,
    "",
    body.slice(0, 1000),
  ].join("\n");

  atomicWrite(filepath, content);
  console.warn(`[bulletin] posted: ${filename} (priority=${priority})`);
  return filename;
}

/**
 * List active notices.
 * @param {"high"|"normal"|"all"} priority
 * @returns {Array<{file, priority, content, agent, expiresAt}>}
 */
export function listNotices(priority = "all") {
  const results = [];
  const dirs = priority === "all"
    ? ["high", "normal"]
    : [priority];

  for (const tier of dirs) {
    const dir = path.join(BULLETIN_ROOT, tier);
    if (!fs.existsSync(dir)) continue;

    for (const file of fs.readdirSync(dir).sort().reverse()) {
      if (!file.endsWith(".md")) continue;
      try {
        const content = fs.readFileSync(path.join(dir, file), "utf-8");
        // Parse metadata from HTML comment
        const meta = content.match(/<!-- bulletin:.*?agent=(\S+) expires=(\S+) -->/);
        results.push({
          file,
          priority: tier,
          content: content.replace(/<!-- bulletin:.*?-->/, "").trim(),
          agent: meta?.[1] || "unknown",
          expiresAt: meta?.[2] || null,
        });
      } catch { /* skip unreadable */ }
    }
  }

  return results;
}

/**
 * Archive expired notices.
 * Called periodically (e.g., from session-end or cron).
 * @returns {number} Number of archived notices
 */
export function archiveExpired() {
  let archived = 0;
  const archiveDir = path.join(BULLETIN_ROOT, "archive");
  if (!fs.existsSync(archiveDir)) fs.mkdirSync(archiveDir, { recursive: true });

  for (const tier of ["high", "normal"]) {
    const dir = path.join(BULLETIN_ROOT, tier);
    if (!fs.existsSync(dir)) continue;

    for (const file of fs.readdirSync(dir)) {
      if (!file.endsWith(".md")) continue;
      const filepath = path.join(dir, file);
      try {
        const content = fs.readFileSync(filepath, "utf-8");
        const meta = content.match(/expires=(\S+)/);
        if (meta?.[1]) {
          const expires = new Date(meta[1]);
          if (isNaN(expires.getTime())) {
            // Malformed date — fall back to file age check
            const stat = fs.statSync(filepath);
            if (Date.now() - stat.mtimeMs > MAX_AGE_MS) {
              fs.renameSync(filepath, path.join(archiveDir, file));
              archived++;
            }
          } else if (expires < new Date()) {
            fs.renameSync(filepath, path.join(archiveDir, file));
            archived++;
          }
        } else {
          // No expiry — check file age
          const stat = fs.statSync(filepath);
          if (Date.now() - stat.mtimeMs > MAX_AGE_MS) {
            fs.renameSync(filepath, path.join(archiveDir, file));
            archived++;
          }
        }
      } catch { /* skip */ }
    }
  }

  if (archived > 0) {
    console.warn(`[bulletin] archived ${archived} expired notices`);
  }
  return archived;
}

export default { postNotice, listNotices, archiveExpired };
