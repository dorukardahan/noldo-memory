/**
 * Shared utility functions for noldo-memory hooks.
 * Created: 2026-03-24 [Dedup atomicWrite from ats.js, spr.js, shared-state.js]
 */

import fs from "node:fs";
import crypto from "node:crypto";

/**
 * Atomic write — write to temp file then rename to avoid partial writes.
 * @param {string} filePath
 * @param {string} content
 */
export function atomicWrite(filePath, content) {
  const tmpPath = `${filePath}.${crypto.randomBytes(4).toString("hex")}.tmp`;
  fs.writeFileSync(tmpPath, content, { mode: 0o600 });
  fs.renameSync(tmpPath, filePath);
}

/**
 * Simple file-based mutex using O_EXCL lockfile.
 * Prevents TOCTOU race conditions on shared state files (review fix H-2).
 * Graceful degradation: stale lock cleanup (5s), max 3 retries.
 * @param {string} filePath - The data file to lock
 * @param {function} fn - Critical section
 * @returns {*} Return value of fn
 */
export function withFileLock(filePath, fn) {
  const lockPath = `${filePath}.lock`;
  const maxRetries = 3;
  const baseDelay = 5; // ms

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      // O_EXCL fails if file exists — atomic create-or-fail
      const fd = fs.openSync(lockPath, "wx");
      fs.closeSync(fd);
      try {
        return fn();
      } finally {
        try { fs.unlinkSync(lockPath); } catch { /* best effort */ }
      }
    } catch (e) {
      if (e.code === "EEXIST") {
        // Lock held — check if stale (>5s = likely crashed holder)
        try {
          const stat = fs.statSync(lockPath);
          if (Date.now() - stat.mtimeMs > 5000) {
            try { fs.unlinkSync(lockPath); } catch { /* race ok */ }
            continue;
          }
        } catch { /* stat failed, retry */ }
        // Exponential backoff
        const delay = baseDelay * Math.pow(2, attempt);
        const end = Date.now() + delay;
        while (Date.now() < end) { /* busy wait — hooks are short-lived */ }
        continue;
      }
      // Non-lock error — run without lock (graceful degradation)
      return fn();
    }
  }
  // All retries exhausted — run without lock
  return fn();
}
