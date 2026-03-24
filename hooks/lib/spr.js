/**
 * Solved Problem Registry (SPR) — Prevents step repetition (MAST FM-1.3).
 * Tracks problem→root_cause→fix→verification records per workspace.
 *
 * Created: 2026-03-23 [MAST Improvement — P0]
 */

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { atomicWrite } from "./util.js";

const MAX_PROBLEMS = 50;

function sprPath(workspaceDir) {
  return path.join(workspaceDir, "memory", "solved-problems.json");
}

/**
 * Read solved problems from workspace.
 */
export function readSPR(workspaceDir) {
  try {
    return JSON.parse(fs.readFileSync(sprPath(workspaceDir), "utf-8"));
  } catch {
    return { version: 1, problems: [], index: {} };
  }
}

/**
 * Write SPR atomically.
 */
export function writeSPR(workspaceDir, spr) {
  try {
    const dir = path.join(workspaceDir, "memory");
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    atomicWrite(sprPath(workspaceDir), JSON.stringify(spr, null, 2));
  } catch (err) {
    console.warn("[spr] write error:", err.message);
  }
}

/**
 * Add a solved problem with keyword index.
 * @param {string} workspaceDir
 * @param {{ title: string, root_cause: string, fix: string, files_changed?: string[], tags?: string[], verification?: object, agent?: string }} problem
 * @returns {string} problem id
 */
export function addSolvedProblem(workspaceDir, problem) {
  try {
    const spr = readSPR(workspaceDir);

    const date = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    const hex = crypto.randomBytes(3).toString("hex");
    const id = `spr-${date}-${hex}`;

    const entry = {
      id,
      title: (problem.title || "").slice(0, 200),
      root_cause: (problem.root_cause || "").slice(0, 500),
      fix: (problem.fix || "").slice(0, 500),
      files_changed: (problem.files_changed || []).slice(0, 10),
      verification: problem.verification || { method: "manual", verified_at: new Date().toISOString(), verified_by_tool: false },
      tags: (problem.tags || []).slice(0, 10),
      created_at: new Date().toISOString(),
      agent: problem.agent || "main",
    };

    spr.problems.push(entry);

    // Cap at MAX_PROBLEMS
    if (spr.problems.length > MAX_PROBLEMS) {
      spr.problems = spr.problems.slice(-MAX_PROBLEMS);
    }

    // Rebuild index
    rebuildIndex(spr);

    writeSPR(workspaceDir, spr);
    console.warn(`[spr] problem added: ${id} "${entry.title.slice(0, 60)}"`);
    return id;
  } catch (err) {
    console.warn("[spr] addSolvedProblem error:", err.message);
    return null;
  }
}

/**
 * Search SPR by keywords. Returns matching problems.
 * @param {string} workspaceDir
 * @param {string[]} keywords
 * @returns {Array}
 */
export function searchSPR(workspaceDir, keywords) {
  try {
    const spr = readSPR(workspaceDir);
    if (!keywords || keywords.length === 0) return [];

    const lowerKw = keywords.map((k) => k.toLowerCase());

    // First try index
    const matchedIds = new Set();
    for (const kw of lowerKw) {
      for (const [indexKey, ids] of Object.entries(spr.index || {})) {
        if (indexKey.includes(kw) || kw.includes(indexKey)) {
          for (const id of ids) matchedIds.add(id);
        }
      }
    }

    // Also do full-text search on title, root_cause, fix, tags
    for (const problem of spr.problems) {
      const haystack = [
        problem.title,
        problem.root_cause,
        problem.fix,
        ...(problem.tags || []),
      ]
        .join(" ")
        .toLowerCase();

      if (lowerKw.some((kw) => haystack.includes(kw))) {
        matchedIds.add(problem.id);
      }
    }

    return spr.problems.filter((p) => matchedIds.has(p.id));
  } catch (err) {
    console.warn("[spr] search error:", err.message);
    return [];
  }
}

/**
 * Get most recent N problems.
 */
export function getRecentProblems(workspaceDir, limit = 5) {
  const spr = readSPR(workspaceDir);
  return spr.problems.slice(-limit);
}

function rebuildIndex(spr) {
  spr.index = {};
  for (const problem of spr.problems) {
    const keywords = [
      ...(problem.tags || []),
      ...problem.title.toLowerCase().split(/\s+/).filter((w) => w.length > 1),
    ];
    for (const kw of keywords) {
      const key = kw.toLowerCase().replace(/[^a-z0-9çğıöşü-]/gi, "");
      if (!key) continue;
      if (!spr.index[key]) spr.index[key] = [];
      if (!spr.index[key].includes(problem.id)) {
        spr.index[key].push(problem.id);
      }
    }
  }
}
