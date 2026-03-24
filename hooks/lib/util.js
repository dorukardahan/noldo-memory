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
