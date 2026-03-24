/**
 * Active Task State (ATS) — Per-workspace structured task tracking.
 * Prevents "ne yapmaya çalışıyordun?" amnesia (MAST FM-1.4).
 *
 * Created: 2026-03-23 [MAST Improvement — P0]
 */

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { atomicWrite } from "./util.js";

const MAX_ACTIVE = 5;
const EXPIRE_MS = 6 * 60 * 60 * 1000; // 6 hours (reduced from 48h per review H-1)

function atsPath(workspaceDir) {
  return path.join(workspaceDir, "memory", "active-tasks.json");
}

/**
 * Read active tasks from workspace.
 * @param {string} workspaceDir
 * @returns {{ version: number, tasks: Array, max_active: number, last_cleanup: string }}
 */
export function readATS(workspaceDir) {
  try {
    return JSON.parse(fs.readFileSync(atsPath(workspaceDir), "utf-8"));
  } catch {
    return { version: 1, tasks: [], max_active: MAX_ACTIVE, last_cleanup: new Date().toISOString() };
  }
}

/**
 * Write ATS atomically.
 */
export function writeATS(workspaceDir, ats) {
  try {
    const dir = path.join(workspaceDir, "memory");
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    atomicWrite(atsPath(workspaceDir), JSON.stringify(ats, null, 2));
  } catch (err) {
    console.warn("[ats] write error:", err.message);
  }
}

/**
 * Generate a task ID.
 */
function generateId() {
  const date = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  const hex = crypto.randomBytes(3).toString("hex");
  return `ats-${date}-${hex}`;
}

/**
 * Add a new task. Auto-expires old tasks, caps at max_active.
 * @param {string} workspaceDir
 * @param {{ title: string, agent: string, goal: string, session_key?: string, priority?: string }} task
 * @returns {string|null} task id or null if capped
 */
export function addTask(workspaceDir, task) {
  try {
    const ats = readATS(workspaceDir);
    cleanupExpired(ats);

    // Don't add duplicates (same title within 2h)
    const twoHoursAgo = Date.now() - 2 * 60 * 60 * 1000;
    const isDupe = ats.tasks.some(
      (t) =>
        t.status === "in_progress" &&
        t.title === task.title &&
        new Date(t.created_at).getTime() > twoHoursAgo
    );
    if (isDupe) return null;

    const activeTasks = ats.tasks.filter((t) => t.status === "in_progress");
    if (activeTasks.length >= MAX_ACTIVE) {
      // Expire oldest
      const oldest = activeTasks.sort((a, b) => new Date(a.updated_at) - new Date(b.updated_at))[0];
      if (oldest) oldest.status = "expired";
    }

    const id = generateId();
    const now = new Date().toISOString();
    ats.tasks.push({
      id,
      title: (task.title || "Untitled").slice(0, 120),
      status: "in_progress",
      created_at: now,
      updated_at: now,
      agent: task.agent || "main",
      context: {
        goal: (task.goal || "").slice(0, 500),
        current_step: "",
        files_touched: [],
        blockers: [],
        findings_so_far: "",
      },
      session_key: (task.session_key || "").slice(0, 100),
      priority: task.priority || "medium",
    });

    writeATS(workspaceDir, ats);
    console.warn(`[ats] task added: ${id} "${task.title?.slice(0, 60)}"`);
    return id;
  } catch (err) {
    console.warn("[ats] addTask error:", err.message);
    return null;
  }
}

/**
 * Update a task by ID.
 */
export function updateTask(workspaceDir, taskId, updates) {
  try {
    const ats = readATS(workspaceDir);
    const task = ats.tasks.find((t) => t.id === taskId);
    if (!task) return false;

    if (updates.status) task.status = updates.status;
    if (updates.context) {
      task.context = { ...task.context, ...updates.context };
      // Cap context size to prevent unbounded growth
      for (const [key, val] of Object.entries(task.context)) {
        if (typeof val === "string" && val.length > 600) {
          task.context[key] = val.slice(0, 600);
        }
      }
    }
    if (updates.title) task.title = updates.title.slice(0, 120);
    if (updates.priority) task.priority = updates.priority;
    task.updated_at = new Date().toISOString();

    writeATS(workspaceDir, ats);
    return true;
  } catch (err) {
    console.warn("[ats] updateTask error:", err.message);
    return false;
  }
}

/**
 * Find first in_progress task.
 */
export function findActiveTask(workspaceDir) {
  const ats = readATS(workspaceDir);
  return ats.tasks.find((t) => t.status === "in_progress") || null;
}

/**
 * Get all active (in_progress) tasks.
 */
export function getActiveTasks(workspaceDir) {
  const ats = readATS(workspaceDir);
  return ats.tasks.filter((t) => t.status === "in_progress");
}

/**
 * Cleanup expired tasks (not updated for 48h).
 */
function cleanupExpired(ats) {
  const now = Date.now();
  for (const task of ats.tasks) {
    if (task.status === "in_progress") {
      const updated = new Date(task.updated_at).getTime();
      if (now - updated > EXPIRE_MS) {
        task.status = "expired";
      }
    }
  }
  // Keep last 20 tasks total
  if (ats.tasks.length > 20) {
    ats.tasks = ats.tasks.slice(-20);
  }
  ats.last_cleanup = new Date().toISOString();
}

/**
 * Run cleanup and persist.
 */
export function cleanupExpiredTasks(workspaceDir) {
  try {
    const ats = readATS(workspaceDir);
    cleanupExpired(ats);
    writeATS(workspaceDir, ats);
  } catch (err) {
    console.warn("[ats] cleanup error:", err.message);
  }
}
