import path from "node:path";
import crypto from "node:crypto";
import { readFileSync } from "node:fs";
import { resolveAgentId as _resolveAgentId } from "../lib/runtime.js";

const MEMORY_API = "http://localhost:8787/v1";
const API_KEY_PATH =
  process.env.AGENT_MEMORY_API_KEY_FILE || `${process.env.HOME}/.noldomem/memory-api-key`;
let _memoryApiKey = "";
try {
  _memoryApiKey = readFileSync(API_KEY_PATH, "utf-8").trim();
} catch (e) {
  console.warn("[realtime-capture] error:", e.message || e);
}

const DECISION_MARKERS = [
  /\bkarar\b/i,
  /yapalım/i,
  /yapacağız/i,
  /\btamam\b/i,
  /anlaştık/i,
  /\bdecided\b/i,
  /\blet'?s do\b/i,
];

const CRON_PATTERNS = [
  /^\[cron:/i,
  /steward-engage/i,
  /steward-post/i,
  /\/steward-/i,
  /HEARTBEAT_OK/i,
];

const LOW_SIGNAL_PATTERNS = [
  /A new session was started via \/new or \/reset/i,
  /Conversation info \(untrusted metadata\)/i,
  /\[Subagent Context\]/i,
  /^User:\s*Conversation info/i,
  /\[System Message\]/i,
  /\[Queued announce messages while agent was busy\]/i,
  /\bA cron job\b/i,
  /\bA subagent task\b/i,
  /^System:\s*\[System Message\]/i,
  /^System:\s*\[(?:Queued|Internal|Subagent)\b/i,
  // Repetitive health check alerts (devops cron fires every 2h, creates duplicates)
  /(?:system\s+(?:health|watchdog)\s+alert|DOCKER\s+HEALTHY\s+COUNT)/i,
];

const FEEDBACK_TAG_PATTERNS = {
  verification: [
    /neden\s*(?:doğrulamadın|kontrol\s*etmedin|bakmadın)/i,
    /\bverify\b/i,
    /doğrula/i,
    /kontrol\s*et/i,
    /bak\s*(?:kaynak|koda|dosyaya)/i,
    /tekrar\s*(?:oku|bak|kontrol)/i,
    /check\s*again/i,
    /look\s*at\s*the\s*source/i,
  ],
  fabrication: [
    /\bfabrication\b/i,
    /yanlış\s*(?:bilgi|söyledin|anladın)/i,
    /uydur(?:ma|dun)/i,
    /hayır\s*öyle\s*değil/i,
    /that'?s\s*(?:wrong|incorrect|not\s*(?:right|true))/i,
    /actually,?\s*(?:that'?s|it'?s|you'?re)\s*(?:wrong|incorrect|not)/i,
    /no\s*that'?s\s*not/i,
  ],
  premature_suggestion: [/öner(?:i|im)/i, /premature\s*suggestion/i, /too\s*early\s*to\s*suggest/i],
  did_not_read_code: [
    /kodu\s*(?:okumadın|incelemedin)/i,
    /did\s*not\s*read\s*code/i,
    /read\s*the\s*code\s*first/i,
  ],
  context_loss: [
    /ne\s*yapmaya\s*çalış(?:ıyordun|tın)/i,
    /context\s*(?:loss|kaybı)/i,
    /az\s*önce\s*(?:söyledim|anlattım|konuştuk)/i,
    /daha\s*(?:yeni|demin)\s*(?:söyledim|konuştuk)/i,
    /we\s*just\s*(?:talked|discussed)/i,
  ],
};

const FEEDBACK_MARKERS = [
  /kaç\s*kere\s*(?:dedim|söyledim)/i,
  /neden\s*(?:doğrulamadın|kontrol\s*etmedin|bakmadın)/i,
  /bir\s*daha\s*yapma/i,
  /hata\s*(?:yaptın|yapıyorsun|tekrar)/i,
  /yanlış\s*(?:yaptın|bilgi|söyledin|anladın)/i,
  /\bfabrication\b/i,
  /güven(?:i|ini)?\s*(?:kaybett|sarsar|kırıl)/i,
  /sürekli\s*(?:aynı|böyle|yanlış)/i,
  /aa\s*(?:zaten\s*)?varmış/i,
  /don'?t\s*(?:repeat|do\s*that|make\s*that)/i,
  /never\s*again/i,
  /wrong\s*(?:again|info)/i,
  /how\s*many\s*times/i,
  // Broader feedback patterns (TR + EN) — require error/correction context
  /hayır\s*öyle\s*değil/i,
  /tekrar\s*oku/i,
  /bak\s*(?:kaynak|koda|dosyaya)/i,
  /(?:neden|niye)\s*kontrol\s*etmedin/i,  // "kontrol et" alone too broad, require "neden/niye"
  /that'?s\s*(?:wrong|incorrect|not\s*right)/i,
  /check\s*(?:again|the\s*source|the\s*code)/i,
  /no\s*that'?s\s*not/i,
  /look\s*at\s*the\s*(?:source|code|file)/i,
];

// Agent self-correction markers — capture when agent admits mistakes
// Patterns must be specific to avoid FP on casual speech
const SELF_CORRECTION_MARKERS = [
  /yanılmışım/i,
  /pardon,?\s*(?:yanlış|hata)/i,    // "pardon" only with error context, not as politeness
  /düzeltiyorum/i,
  /hatalıydım/i,
  /yanlış\s*söyledim/i,
  /I\s*was\s*wrong/i,
  /my\s*mistake/i,
  /I\s*(?:was\s*)?incorrect/i,
  /let\s*me\s*correct/i,
  /I\s*(?:should\s*have|need\s*to)\s*(?:check|verify|look)/i,
];

function cleanMessage(raw = "") {
  let text = String(raw || "").replace(/\r\n/g, "\n");
  // Strip OpenClaw Slack envelope prefix (RC-1 fix: this prefix caused isLowSignal to match ALL Slack messages)
  text = text.replace(
    /^System:\s*\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?\s+GMT[+-]\d+\]\s*Slack message(?:\s+edited)?\s+in\s+#\S+(?:\s+from\s+[^:]+)?[.:]\s*/i,
    ""
  );
  // Strip OpenClaw runtime context preamble
  text = text.replace(
    /^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+GMT[+-]\d+\]\s*OpenClaw runtime context \(internal\):[\s\S]*?(?=\n\n|$)/gi,
    ""
  );
  text = text.replace(/Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```/gi, "");
  text = text.replace(/\[Subagent Context\][\s\S]*?(?=\n\n|$)/gi, "");
  text = text.replace(/^\s*\[[^\]]+\]\s*\[System Message\].*$/gim, "");
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

// parseAgentIdFromSessionKey and parseAgentIdFromWorkspaceDir removed —
// consolidated into lib/runtime.js (review fix H-3)

function deriveSessionNamespace(sessionKey = "") {
  const raw = String(sessionKey || "").trim().toLowerCase();
  if (!raw) return null;
  const hash = crypto.createHash("sha1").update(raw).digest("hex").slice(0, 16);
  return `session-${hash}`;
}

function detectFeedbackTags(text = "") {
  const tags = [];
  for (const [tag, patterns] of Object.entries(FEEDBACK_TAG_PATTERNS)) {
    if (patterns.some((p) => p.test(text))) tags.push(tag);
  }
  return tags.length > 0 ? tags : ["verification"];
}

function isCronNoise(text = "") {
  return CRON_PATTERNS.some((p) => p.test(text));
}
function isDecision(text = "") {
  return DECISION_MARKERS.some((p) => p.test(text));
}
function isFeedback(text = "") {
  return FEEDBACK_MARKERS.some((p) => p.test(text));
}
function isLowSignal(text = "") {
  return LOW_SIGNAL_PATTERNS.some((p) => p.test(text));
}

function scoreImportance(text = "") {
  if (!text || text.length < 10 || isCronNoise(text) || isLowSignal(text)) return 0;
  let score = 0.45;
  if (isDecision(text)) score = Math.max(score, 0.9);
  if (isFeedback(text)) score = Math.max(score, 0.85);
  if (text.length > 80) score += 0.05;
  if (text.length > 200) score += 0.05;
  // Boost corrections, confirmations, results — only when message has substance (>30ch)
  // to avoid capturing bare acknowledgments like "tamam" or "evet"
  if (text.length > 30 && /\b(evet|hayır|tamam|onaylıyorum|hayır yapma|yes|no|confirmed|approved)\b/i.test(text)) score += 0.10;
  if (/\b(hata|bug|fix|düzelt|sorun|problem|error)\b/i.test(text)) score += 0.10;
  return Math.min(1.0, score);
}

function resolveAgentId(event) {
  // Delegate to shared runtime.js implementation (dedup H-3)
  return _resolveAgentId(event, event?.context?.workspaceDir || "");
}

function resolveMessage(event) {
  if (event?.type !== "message") return null;
  const content = cleanMessage(String(event?.context?.content || ""));
  if (!content) return null;
  if (event.action === "received") {
    return { role: "user", content };
  }
  if (event.action === "sent") {
    if (event?.context?.success === false) return null;
    return { role: "assistant", content };
  }
  return null;
}

function shouldStoreAssistant(content = "") {
  if (!content || content.length < 60) return false;
  if (isCronNoise(content) || isLowSignal(content)) return false;
  if (isDecision(content)) return true;
  // Lowered from 220 to 80 chars — let API-side classify importance
  return content.length >= 80 || /\b(plan|next steps?|todo|aksiyon|yapılacak|fix|result|sonuç|tamamlandı|done|verified)\b/i.test(content);
}

async function store(text, category, importance, agent = "main", namespace = "default") {
  if (!_memoryApiKey) return;
  try {
    const res = await fetch(`${MEMORY_API}/store`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": _memoryApiKey },
      body: JSON.stringify({
        text: text.slice(0, 3000),
        category,
        importance,
        agent,
        namespace,
        source: "hook",
      }),
      signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) {
      console.warn(
        `[realtime-capture] store failed: status=${res.status} category=${category} agent=${agent}`
      );
    }
  } catch (e) {
    console.warn("[realtime-capture] error:", e.message || e);
  }
}

const realtimeCaptureHook = async (event) => {
  const message = resolveMessage(event);
  if (!message) return;

  const content = message.content;
  if (!content || content.startsWith("/") || isCronNoise(content) || isLowSignal(content)) return;

  const agentId = resolveAgentId(event);
  const sessionNamespace = deriveSessionNamespace(event?.sessionKey) || "default";

  if (message.role === "user") {
    if (isFeedback(content)) {
      const tags = detectFeedbackTags(content);
      const tagged = tags.map((tag) => `[tag=${tag}]`).join("");
      await store(`[Feedback]${tagged}[lang=tr] ${content}`, "lesson", 0.95, agentId, "default");
      return;
    }

    if (isDecision(content)) {
      await store(`[Decision] ${content}`, "decision", 0.9, agentId, "default");
      return;
    }

    if (content.length > 15) {
      const imp = scoreImportance(content);
      if (imp >= 0.3) {
        await store(content, "user", imp, agentId, sessionNamespace);
        console.log(`[realtime-capture] stored user msg (${content.length}ch, imp=${imp.toFixed(2)}, agent=${agentId})`);
      } else if (content.length > 50) {
        console.log(`[realtime-capture] dropped user msg (${content.length}ch, imp=${imp.toFixed(2)}, agent=${agentId})`);
      }
    }
    return;
  }

  // Self-correction capture — agent admitting mistakes → store as lesson
  if (message.role === "assistant" && SELF_CORRECTION_MARKERS.some((p) => p.test(content))) {
    if (content.length > 30) {
      await store(`[Self-Correction] ${content}`, "lesson", 0.85, agentId, "default");
      console.log(`[realtime-capture] self-correction captured (${content.length}ch, agent=${agentId})`);
    }
  }

  // Propagation completeness check — detect "N files updated" claims
  if (message.role === "assistant") {
    const propagationMatch = content.match(
      /(\d+)\s*(?:dosya|file|location|yer)\s*(?:güncellen|update|değiştir|changed|modified)/i
    ) || content.match(
      /(?:update[ds]?|güncellen|değiştir|changed|modified)\s*(?:in\s+)?(\d+)\s*(?:dosya|file|location|yer)/i
    );
    if (propagationMatch) {
      const claimedCount = parseInt(propagationMatch[1], 10);
      // Count actual file references in the message (paths with extensions)
      // Match both normal files (name.ext) and bare dotfiles (.env, .env.local)
      const normalFiles = content.match(/[\w/.-]+\.\w{1,10}/g) || [];
      const dotFiles = content.match(/(?:^|\s|,|:)(\.env(?:\.\w+)?)\b/gi) || [];
      const cleanDotFiles = dotFiles.map(f => f.trim().replace(/^[,:\s]+/, ""));
      const allFileRefs = [...normalFiles, ...cleanDotFiles];
      const uniqueFiles = [...new Set(allFileRefs.filter(f =>
        /\.(env|conf|json|yaml|yml|toml|py|js|ts|sh|service|cfg|ini)$/i.test(f) ||
        /^\.env(?:\.\w+)?$/i.test(f)
      ))];
      if (claimedCount > 0 && uniqueFiles.length > 0 && uniqueFiles.length < claimedCount) {
        const warning = `[Propagation Warning] Agent claimed ${claimedCount} files updated but only ${uniqueFiles.length} identified: ${uniqueFiles.join(", ")}. Verify all target files were actually modified.`;
        await store(warning, "lesson", 0.90, agentId, "default");
        console.warn(`[realtime-capture] propagation warning: claimed=${claimedCount} found=${uniqueFiles.length}`);
      }
    }
  }

  if (!shouldStoreAssistant(content)) {
    if (content.length > 100) {
      console.log(`[realtime-capture] dropped assistant msg (${content.length}ch, agent=${agentId})`);
    }
    return;
  }
  const importance = isDecision(content) ? 0.85 : Math.max(0.45, scoreImportance(content));
  await store(content, "assistant", importance, agentId, sessionNamespace);
  console.log(`[realtime-capture] stored assistant msg (${content.length}ch, imp=${importance.toFixed(2)}, agent=${agentId})`);
};

export default realtimeCaptureHook;
