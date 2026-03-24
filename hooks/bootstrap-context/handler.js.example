import fs from "node:fs/promises";
import path from "node:path";
import { readFileSync } from "node:fs";
import {
  deriveSessionNamespace,
  readWorkspacePolicy,
  resolveAgentId,
  resolveWorkspaceDir,
  resolveSessionKey,
} from "../lib/runtime.js";
import {
  readFabricationLog,
  readVerifiedFacts,
  getFabricationScore,
  getProofRequirement,
} from "../lib/shared-state.js";
import { readATS } from "../lib/ats.js";
import { getRecentProblems } from "../lib/spr.js";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH =
  process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try {
  _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim();
} catch (e) {
  console.warn("[bootstrap-context] error:", e.message || e);
}

const MAX_MEMORY_CHARS = 8000;
const MAX_DAILY_CHARS = 6000;
const MAX_CROSSAGENT_CHARS = 2000;
const MAX_LAST_SESSION_CHARS = 2000;
const MAX_LESSON_CHARS = 3000;
const ENABLE_CROSS_AGENT_RECALL = process.env.OPENCLAW_ENABLE_CROSS_AGENT_RECALL === "1";

const CRON_NOISE_PATTERNS = [/^\[cron:/i, /steward-engage/i, /steward-post/i, /\/steward-/i];

const LOW_SIGNAL_MEMORY_PATTERNS = [
  /A new session was started via \/new or \/reset/i,
  /\[Subagent Context\]/i,
  /Conversation info \(untrusted metadata\)/i,
  /^User:\s*Conversation info/i,
  /\[System Message\]/i,
  /\[Queued announce messages while agent was busy\]/i,
  /^✅\s*Subagent\s+.+\s+finished/i,
  /\bA cron job\b/i,
  /\bA subagent task\b/i,
  /^System:\s*\[System Message\]/i,
  /^System:\s*\[(?:Queued|Internal|Subagent)\b/i,
];

function isCronNoise(text = "") {
  return CRON_NOISE_PATTERNS.some((p) => p.test(text));
}

function isLowSignalMemory(text = "") {
  if (!text) return true;
  if (isCronNoise(text)) return true;
  return LOW_SIGNAL_MEMORY_PATTERNS.some((p) => p.test(text));
}

function truncateSection(text, maxChars) {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars) + `\n... [truncated to ${maxChars} chars]`;
}

function stripLowSignalText(text = "") {
  let raw = String(text || "");
  // Strip OpenClaw Slack envelope prefix from recalled memories
  raw = raw.replace(
    /^System:\s*\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s+GMT[+-]\d+\]\s*Slack message(?:\s+edited)?\s+in\s+#\S+(?:\s+from\s+[^:]+)?[.:]\s*/gim,
    ""
  );
  const withoutBlocks = raw.replace(
    /Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```/gi,
    ""
  );
  const cleanedLines = withoutBlocks
    .split("\n")
    .filter((line) => !LOW_SIGNAL_MEMORY_PATTERNS.some((p) => p.test(line)));
  return cleanedLines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

async function recall(agentId, query, options = {}) {
  if (!_memoryApiKey) return [];
  try {
    const payload = {
      query,
      limit: options.limit ?? 20,
      min_score: options.minScore ?? 0.01,
      agent: agentId,
    };
    if (options.namespace) payload.namespace = options.namespace;
    if (options.memoryType) payload.memory_type = options.memoryType;

    const res = await fetch(`${MEMORY_API}/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(options.timeoutMs ?? 5000),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.results || data.memories || [];
  } catch (e) {
    console.warn("[bootstrap-context] error:", e.message || e);
    return [];
  }
}

function dedupeMemories(items) {
  const deduped = [];
  const seen = new Set();
  for (const item of items || []) {
    const text = item?.text || item?.content || "";
    if (isLowSignalMemory(text)) continue;
    const key = text.slice(0, 120);
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(item);
  }
  return deduped;
}

async function fetchRecentMemories(agentId, sessionNamespace) {
  if (!_memoryApiKey) return [];

  const queries = [
    "recent decisions and configuration changes",
    "last conversation summary and active work",
    "credential updates, config file changes, environment variable modifications",
  ];

  const sessionScoped = [];
  if (sessionNamespace && sessionNamespace !== "default") {
    for (const q of queries) {
      const scoped = await recall(agentId, q, {
        namespace: sessionNamespace,
        limit: 20,
        minScore: 0.0,
        timeoutMs: 4000,
      });
      sessionScoped.push(...scoped);
    }
  }

  const dedupedSession = dedupeMemories(sessionScoped);
  if (dedupedSession.length >= 6) {
    return dedupedSession;
  }

  const global = [];
  for (const q of queries) {
    const query = agentId !== "main" ? `${agentId}: ${q}` : q;
    const results = await recall(agentId, query, { limit: 20, minScore: 0.01, timeoutMs: 5000 });
    global.push(...results);
  }

  return dedupeMemories([...dedupedSession, ...global]);
}

async function fetchSharedNamespaceMemories(agentId, sharedNamespaces = []) {
  const items = [];
  for (const namespace of sharedNamespaces) {
    const results = await recall(agentId, "shared operational context and reusable decisions", {
      namespace,
      limit: 6,
      minScore: 0.0,
      timeoutMs: 4000,
    });
    items.push(...results);
  }
  return dedupeMemories(items);
}

async function fetchDecisionMemories(agentId) {
  if (!_memoryApiKey) return null;
  return recall(agentId, "recent decisions, architectural choices, configuration changes", {
    limit: 8,
    minScore: 0.01,
    timeoutMs: 4000,
  });
}

async function fetchLessons(agentId) {
  if (!_memoryApiKey) return [];
  const results = await recall(agentId, "behavioral lessons", {
    limit: 20,
    minScore: 0.0,
    memoryType: "lesson",
    timeoutMs: 4000,
  });

  const sorted = results.sort((a, b) => {
    const imp = (Number(b.importance) || 0) - (Number(a.importance) || 0);
    if (imp !== 0) return imp;
    return (Number(b.created_at) || 0) - (Number(a.created_at) || 0);
  });

  const deduped = [];
  for (const item of sorted) {
    const text = stripLowSignalText(item.text || item.content || "");
    if (!text) continue;
    if (isLowSignalMemory(text)) continue;
    const isDuplicate = deduped.some((existing) => {
      const prev = stripLowSignalText(existing.text || existing.content || "");
      return prev.includes(text) || text.includes(prev);
    });
    if (!isDuplicate) deduped.push(item);
    if (deduped.length >= 5) break;
  }

  return deduped;
}

async function readLastSessionSummary(workspaceDir) {
  const filePath = path.join(workspaceDir, "memory", "last-session.md");
  try {
    const content = await fs.readFile(filePath, "utf-8");
    const cleaned = stripLowSignalText(content);
    if (cleaned) {
      return truncateSection(cleaned, MAX_LAST_SESSION_CHARS);
    }
  } catch (e) {
    if (e?.code !== "ENOENT") {
      console.warn("[bootstrap-context] error:", e.message || e);
    }
  }
  return null;
}

async function readDailyNotes(memoryDir) {
  const now = new Date();
  const today = now.toISOString().split("T")[0];
  const yesterday = new Date(now - 86400000).toISOString().split("T")[0];
  const parts = [];

  for (const date of [today, yesterday]) {
    try {
      const files = await fs.readdir(memoryDir);
      const matches = files.filter((f) => f.startsWith(date) && f.endsWith(".md"));
      for (const file of matches) {
        const content = await fs.readFile(path.join(memoryDir, file), "utf-8");
        const cleaned = stripLowSignalText(content);
        if (cleaned) parts.push(`## ${file}\n${cleaned}`);
      }
    } catch (e) {
      if (e?.code !== "ENOENT") {
        console.warn("[bootstrap-context] error:", e.message || e);
      }
    }
  }
  return parts.join("\n\n");
}

const bootstrapContextHook = async (event) => {
  if (event.type !== "agent" || event.action !== "bootstrap") return;

  const context = event.context;
  if (!context || !context.bootstrapFiles) return;

  const workspaceDir = resolveWorkspaceDir(event);
  const agentId = resolveAgentId(event, workspaceDir);
  const sessionNamespace = deriveSessionNamespace(event?.sessionKey);
  const memoryDir = path.join(workspaceDir, "memory");
  const policy = await readWorkspacePolicy(workspaceDir);

  console.warn(
    `[bootstrap-context] hook fired (agent=${agentId} namespace=${sessionNamespace})`
  );

  const sections = [];

  const lessons = await fetchLessons(agentId);
  if (lessons.length > 0) {
    console.warn(`[bootstrap-context] fetched ${lessons.length} lessons for ${agentId}`);
    const lessonLines = ["# Critical Lessons (Behavioral Memory)\n"];
    let lessonChars = 0;
    for (const l of lessons) {
      const text = stripLowSignalText(l.text || l.content || "");
      if (!text || isLowSignalMemory(text)) continue;
      const line = `- ${text.slice(0, 400)}`;
      lessonChars += line.length;
      if (lessonChars > MAX_LESSON_CHARS) break;
      lessonLines.push(line);
    }
    lessonLines.push("");
    sections.push(...lessonLines);
  }

  const memories = await fetchRecentMemories(agentId, sessionNamespace);
  if (memories && memories.length > 0) {
    console.warn(
      `[bootstrap-context] fetched ${memories.length} memories for ${agentId} namespace=${sessionNamespace}`
    );
    const memLines = ["# Recent Memory Recall\n"];
    let memChars = 0;
    for (const mem of memories) {
      const text = mem.text || mem.content || JSON.stringify(mem);
      if (isLowSignalMemory(text)) continue;
      const cat = mem.category ? ` [${mem.category}]` : "";
      const strength = mem.strength ? ` (strength: ${mem.strength})` : "";
      const line = `- ${text.slice(0, 300)}${cat}${strength}`;
      memChars += line.length;
      if (memChars > MAX_MEMORY_CHARS) break;
      memLines.push(line);
    }
    sections.push(...memLines);
  } else {
    console.warn(`[bootstrap-context] no memories for ${agentId}`);
  }

  const sharedNamespaceMemories = await fetchSharedNamespaceMemories(
    agentId,
    policy.sharedNamespaces || []
  );
  if (sharedNamespaceMemories.length > 0) {
    const sharedLines = ["\n# Shared Namespace Recall\n"];
    let sharedChars = 0;
    for (const mem of sharedNamespaceMemories.slice(0, 5)) {
      const text = mem.text || mem.content || "";
      if (isLowSignalMemory(text)) continue;
      const namespace = mem.namespace ? ` [${mem.namespace}]` : "";
      const line = `- ${text.slice(0, 220)}${namespace}`;
      sharedChars += line.length;
      if (sharedChars > MAX_CROSSAGENT_CHARS) break;
      sharedLines.push(line);
    }
    if (sharedLines.length > 1) {
      sections.push(...sharedLines);
    }
  }

  const allowCrossWorkspaceRecall =
    ENABLE_CROSS_AGENT_RECALL || policy.crossWorkspaceRecall === true;
  if (
    allowCrossWorkspaceRecall &&
    _memoryApiKey &&
    agentId !== "main" &&
    (!memories || memories.length < 5)
  ) {
    try {
      const mainRes = await fetch(`${MEMORY_API}/recall`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
        body: JSON.stringify({
          query: `recent decisions and context for ${agentId} tasks`,
          limit: 5,
          agent: "main",
        }),
        signal: AbortSignal.timeout(3000),
      });
      if (mainRes.ok) {
        const mainData = await mainRes.json();
        const mainMemories = mainData.results || [];
        if (mainMemories.length > 0) {
          console.warn(`[bootstrap-context] cross-agent: ${mainMemories.length} from main for ${agentId}`);
          const crossLines = ["\n# Main Agent Context\n"];
          let crossChars = 0;
          for (const mem of mainMemories.slice(0, 3)) {
            const t = mem.text || "";
            if (isLowSignalMemory(t)) continue;
            const line = `- ${t.slice(0, 200)} [from main]`;
            crossChars += line.length;
            if (crossChars > MAX_CROSSAGENT_CHARS) break;
            crossLines.push(line);
          }
          sections.push(...crossLines);
        }
      }
    } catch (e) {
      console.warn("[bootstrap-context] error:", e.message || e);
    }
  }

  const decisions = await fetchDecisionMemories(agentId);
  if (decisions && decisions.length > 0) {
    const existingTexts = new Set((memories || []).map((m) => (m.text || "").slice(0, 100)));
    const newDecisions = decisions.filter((d) => !existingTexts.has((d.text || "").slice(0, 100)));
    if (newDecisions.length > 0) {
      console.warn(`[bootstrap-context] fetched ${newDecisions.length} unique decisions for ${agentId}`);
      const decLines = ["\n# Recent Decisions & Changes\n"];
      for (const dec of newDecisions.slice(0, 5)) {
        const text = dec.text || dec.content || "";
        if (isLowSignalMemory(text)) continue;
        decLines.push(`- ${text.slice(0, 250)}`);
      }
      sections.push(...decLines);
    }
  }

  // Recent operational events — typed recall first, text recall fallback
  if (_memoryApiKey) {
    const opsTypedQueries = [
      { memoryType: "config_change", query: "config credential update", title: "Config/Credential Changes", limit: 5 },
      { memoryType: "incident", query: "incident failure crash", title: "Recent Incidents", limit: 3 },
      { memoryType: "deployment", query: "deploy release", title: "Recent Deployments", limit: 3 },
    ];
    for (const oq of opsTypedQueries) {
      try {
        // Try typed recall first (precise)
        let opsItems = await recall(agentId, oq.query, {
          limit: oq.limit,
          minScore: 0.1,
          timeoutMs: 3000,
          memoryType: oq.memoryType,
        });
        // Fallback to text recall if typed returns nothing
        if (!opsItems || opsItems.length === 0) {
          opsItems = await recall(agentId, oq.query, {
            limit: oq.limit,
            minScore: 0.3,
            timeoutMs: 3000,
          });
        }
        if (opsItems && opsItems.length > 0) {
          const opsLines = [`\n# ${oq.title} (CHECK BEFORE ASKING USER)\n`];
          opsLines.push("> ⚠️ Check these memories FIRST before asking the user about related issues.\n");
          for (const item of opsItems.slice(0, oq.limit)) {
            const text = (item.text || "").slice(0, 300);
            opsLines.push(`- ${text}`);
          }
          sections.push(...opsLines);
          console.warn(`[bootstrap-context] injected ${opsItems.length} ${oq.title} for ${agentId}`);
        }
      } catch (e) {
        console.warn(`[bootstrap-context] ops recall error (${oq.title}):`, e.message || e);
      }
    }
  }

  const lastSession = await readLastSessionSummary(workspaceDir);
  if (lastSession) {
    console.warn(`[bootstrap-context] read last-session.md (${lastSession.length} chars)`);
    sections.push("\n# Last Session Summary\n");
    sections.push(lastSession);
  }

  // ── MAST P0 Features (each with independent kill switch) ──
  const MAST_ATS_ENABLED = process.env.MAST_ATS_ENABLED !== "0";
  const MAST_SPR_ENABLED = process.env.MAST_SPR_ENABLED !== "0";
  const MAST_HANDOFF_ENABLED = process.env.MAST_HANDOFF_ENABLED !== "0";
  const MAST_BULLETIN_ENABLED = process.env.MAST_BULLETIN_ENABLED !== "0";
  const MAST_GOVERNANCE_ENABLED = process.env.MAST_GOVERNANCE_ENABLED !== "0";

  // ── MAST P0-8: Clarification Protocol (Safe/Ask/Block tiers) ──
  if (MAST_GOVERNANCE_ENABLED) {
    sections.push(`
# Governance Protocol (MAST)

## Clarification Tiers — Before Acting, Check the Tier

**SAFE (do freely):** Read files, search web, check status, explore workspace, organize memory, git status/log/diff
**ASK (confirm first):** Change passwords/credentials, modify production config, send external messages (email/tweet/DM), delete files, restart services, modify cron jobs, push to git, deploy
**BLOCK (never without explicit request):** Change user passwords, revoke tokens, send messages as the user, modify auth profiles, delete workspaces, run destructive commands on production data

## Verification Protocol

Before claiming completion/existence/absence:
- **completion_claim** → Must have edit/write/exec evidence
- **feature_claim** → Must have read/grep/search evidence
- **config_claim** → Must have file read evidence
- If unsure: "Emin değilim, kontrol ediyorum" → then verify with tool

## Self-Check Before Responding

1. Am I claiming something I haven't verified with a tool? → STOP, verify first
2. Am I about to do an ASK/BLOCK action without user request? → STOP, ask first
3. Did I already solve this problem before? → Check SPR/memory before re-solving
`);
  }

  // ── MAST P0: Active Task State ──
  if (MAST_ATS_ENABLED) try {
    const ats = readATS(workspaceDir);
    const activeTasks = (ats.tasks || []).filter((t) => t.status === "in_progress");
    if (activeTasks.length > 0) {
      const atsLines = ["\n# Active Tasks (resume these)\n"];
      let atsChars = 0;
      for (const task of activeTasks.slice(0, 5)) {
        const line = `- **${task.title}** [${task.priority}] — ${task.context?.goal?.slice(0, 200) || "no goal set"}` +
          (task.context?.current_step ? `\n  Current step: ${task.context.current_step.slice(0, 150)}` : "") +
          (task.context?.findings_so_far ? `\n  Findings: ${task.context.findings_so_far.slice(0, 150)}` : "");
        atsChars += line.length;
        if (atsChars > 2000) break;
        atsLines.push(line);
      }
      sections.push(...atsLines);
      console.warn(`[bootstrap-context] injected ${activeTasks.length} active tasks`);
    }
  } catch (e) {
    if (e?.code !== "ENOENT") console.warn("[bootstrap-context] ATS read error:", e.message);
  }

  // ── MAST P0: Session Handoff ──
  if (MAST_HANDOFF_ENABLED) try {
    const handoffPath = path.join(workspaceDir, "memory", "session-handoff.json");
    const handoff = JSON.parse(await fs.readFile(handoffPath, "utf-8"));
    if (handoff && handoff.summary) {
      const handoffLines = ["\n# Session Handoff (from last session)\n"];
      handoffLines.push(`> ${handoff.summary.slice(0, 300)}`);
      if (handoff.unfinished_work?.length > 0) {
        handoffLines.push("\n**Unfinished:**");
        for (const item of handoff.unfinished_work.slice(0, 5)) {
          handoffLines.push(`- ${item.slice(0, 150)}`);
        }
      }
      if (handoff.next_steps?.length > 0) {
        handoffLines.push("\n**Next steps:**");
        for (const item of handoff.next_steps.slice(0, 5)) {
          handoffLines.push(`- ${item.slice(0, 150)}`);
        }
      }
      if (handoff.user_mood && handoff.user_mood !== "unknown") {
        handoffLines.push(`\n**User mood:** ${handoff.user_mood}`);
      }
      const handoffText = handoffLines.join("\n");
      if (handoffText.length <= 2000) {
        sections.push(handoffText);
      } else {
        sections.push(handoffText.slice(0, 2000) + "\n... [truncated]");
      }
      console.warn(`[bootstrap-context] injected session handoff (${handoff.summary.length} chars)`);
    }
  } catch (e) {
    if (e?.code !== "ENOENT") console.warn("[bootstrap-context] handoff read error:", e.message);
  }

  // ── MAST P1: Recently Solved Problems ──
  if (MAST_SPR_ENABLED) try {
    const recentProblems = getRecentProblems(workspaceDir, 5);
    if (recentProblems.length > 0) {
      const sprLines = ["\n# Recently Solved Problems (don't re-solve these)\n"];
      let sprChars = 0;
      for (const p of recentProblems) {
        const line = `- **${p.title}** → Root cause: ${(p.root_cause || "").slice(0, 100)}. Fix: ${(p.fix || "").slice(0, 100)}` +
          (p.tags?.length > 0 ? ` [${p.tags.join(", ")}]` : "");
        sprChars += line.length;
        if (sprChars > 1500) break;
        sprLines.push(line);
      }
      sections.push(...sprLines);
      console.warn(`[bootstrap-context] injected ${recentProblems.length} solved problems`);
    }
  } catch (e) {
    if (e?.code !== "ENOENT") console.warn("[bootstrap-context] SPR read error:", e.message);
  }

  // ── MAST P1: Cross-Agent Bulletin Board ──
  if (MAST_BULLETIN_ENABLED) try {
    const bulletinDirs = ["/opt/openclaw/shared/bulletin/high", "/opt/openclaw/shared/bulletin/normal"];
    const bulletinLines = [];
    for (const dir of bulletinDirs) {
      try {
        const files = await fs.readdir(dir);
        for (const file of files.filter((f) => f.endsWith(".yaml") || f.endsWith(".yml") || f.endsWith(".md"))) {
          const content = await fs.readFile(path.join(dir, file), "utf-8");
          if (content.length > 0 && content.length < 500) {
            bulletinLines.push(`- [${path.basename(dir)}] ${content.trim().slice(0, 300)}`);
          }
        }
      } catch { /* dir doesn't exist yet, skip */ }
    }
    if (bulletinLines.length > 0) {
      sections.push("\n# Cross-Agent Bulletins\n");
      sections.push(...bulletinLines.slice(0, 10));
      console.warn(`[bootstrap-context] injected ${bulletinLines.length} bulletins`);
    }
  } catch (e) {
    console.warn("[bootstrap-context] bulletin board error:", e.message);
  }

  // ── Fabrication Stats Injection ──
  try {
    const fabLog = readFabricationLog(workspaceDir);
    if (fabLog.stats && fabLog.stats.total > 0) {
      const s = fabLog.stats;
      const fabLines = ["\n# Fabrication Alert\n"];
      if (s.last7d >= 3) {
        fabLines.push(`> ⚠️ **Son 7 günde ${s.last7d} fabrication incident.** Tool ile doğrulama zorunlu.\n`);
      }
      fabLines.push(`Toplam: ${s.total} | Son 7 gün: ${s.last7d}`);
      if (s.byType && Object.keys(s.byType).length > 0) {
        fabLines.push(`Türe göre: ${Object.entries(s.byType).map(([k, v]) => `${k}=${v}`).join(", ")}`);
      }
      if (s.byRootCause && Object.keys(s.byRootCause).length > 0) {
        fabLines.push(`Kök neden: ${Object.entries(s.byRootCause).map(([k, v]) => `${k}=${v}`).join(", ")}`);
      }
      sections.push(...fabLines);
      console.warn(`[bootstrap-context] fabrication stats injected: total=${s.total} last7d=${s.last7d}`);
    }
  } catch (e) {
    console.warn("[bootstrap-context] fabrication log read error:", e.message);
  }

  // ── Fabrication Enforcement Tier (moved from claim-scanner C-2 fix) ──
  if (process.env.MAST_CLAIM_ENFORCE !== "0") {
    try {
      const sessionKey = resolveSessionKey(event);
      if (!sessionKey) throw new Error("no sessionKey");
      const fabScore = getFabricationScore(sessionKey);
      if (fabScore >= 5) {
        sections.push(
          "\n# 🚫 MANDATORY VERIFICATION MODE",
          `Score: ${fabScore} unverified claims this session.`,
          "You MUST use a verification tool BEFORE making any claim.",
          "DO NOT respond with claims until you have tool output as evidence.\n"
        );
      } else if (fabScore >= 3) {
        sections.push(
          "\n# 🚨 Fabrication Pattern Warning",
          `Score: ${fabScore} unverified claims this session.`,
          "Tool verification is MANDATORY. Do NOT say \"done\" without tool proof.\n"
        );
      }
    } catch (e) {
      console.warn("[bootstrap-context] enforcement tier error:", e.message);
    }
  }

  // ── Verified Facts Injection ──
  try {
    const vf = readVerifiedFacts(workspaceDir);
    const facts = Object.entries(vf.facts || {}).filter(([, f]) => !f.stale);
    if (facts.length > 0) {
      const vfLines = ["\n# Verified Facts (auto-captured)\n"];
      const MAX_VF = 10;
      for (const [key, fact] of facts.slice(0, MAX_VF)) {
        const status = fact.verified ? "✅" : "❌";
        vfLines.push(`- ${status} ${fact.claim || key} (${new Date(fact.verifiedAt).toISOString().split("T")[0]})`);
      }
      if (facts.length > MAX_VF) {
        vfLines.push(`... ve ${facts.length - MAX_VF} tane daha (verified-facts.json'da)`);
      }
      sections.push(...vfLines);
      console.warn(`[bootstrap-context] verified facts injected: ${facts.length}`);
    }
  } catch (e) {
    console.warn("[bootstrap-context] verified facts read error:", e.message);
  }

  if (policy.dailyNotesEnabled !== false) {
    const dailyNotes = await readDailyNotes(memoryDir);
    if (dailyNotes) {
      const trimmedNotes = truncateSection(dailyNotes, MAX_DAILY_CHARS);
      console.warn(
        `[bootstrap-context] read daily notes (${dailyNotes.length} -> ${trimmedNotes.length} chars)`
      );
      sections.push("\n# Recent Daily Notes\n");
      sections.push(trimmedNotes);
    }
  }

  if (sections.length === 0) {
    console.warn("[bootstrap-context] no content to inject");
    return;
  }

  // Total size cap to prevent context budget blowout (target: ~20K chars max)
  const MAX_TOTAL_CHARS = 20000;
  let content = sections.join("\n");
  if (content.length > MAX_TOTAL_CHARS) {
    console.warn(`[bootstrap-context] total context ${content.length} chars exceeds ${MAX_TOTAL_CHARS}, truncating`);
    content = content.slice(0, MAX_TOTAL_CHARS) + "\n... [truncated to fit context budget]";
  }

  context.bootstrapFiles.push({
    name: "SESSION_CONTEXT",
    path: "SESSION_CONTEXT",
    content,
  });
  console.warn(`[bootstrap-context] injected SESSION_CONTEXT (${content.length} chars, agent=${agentId})`);
};

import { resilientHandler } from "../lib/resilient-import.js";
export default resilientHandler(bootstrapContextHook, "bootstrap-context");
