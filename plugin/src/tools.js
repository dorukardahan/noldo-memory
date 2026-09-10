/**
 * NoldoMem tool registration — recall, store, pin.
 *
 * These tools let the agent actively search, save, and protect memories
 * without relying on hook-based passive injection.
 */

const VALID_RECALL_MEMORY_TYPES = new Set([
  "fact",
  "preference",
  "rule",
  "conversation",
  "lesson",
  "other",
]);

import { resolveAgentId } from "./scope.js";
import { hasSafeRecallMetadata } from "./sanitize.js";

function formatRecallResults(data) {
  const results = (data.results || []).filter(hasSafeRecallMetadata);
  if (results.length === 0) return "No relevant memories found.";

  return results
    .map((r, i) => {
      const type = r.memory_type ? `[${r.memory_type}]` : "";
      const score = Number.isFinite(r.score) ? ` (rank score ${r.score.toFixed(4)})` : "";
      const text = (r.text || r.content || "").slice(0, 500);
      return `${i + 1}. [id=${r.id} valid_from=${r.valid_from ?? "unknown"} valid_to=${r.valid_to ?? "open"} evidence=${JSON.stringify(r.evidence || {})}] ${type} ${text}${score}`;
    })
    .join("\n");
}

function normalizeRequestedMemoryType(value) {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  return VALID_RECALL_MEMORY_TYPES.has(normalized) ? normalized : undefined;
}

function objectSchema(properties, required = []) {
  return {
    type: "object",
    additionalProperties: false,
    properties,
    ...(required.length ? { required } : {}),
  };
}

function stringSchema(description) {
  return { type: "string", description };
}

function numberSchema(description) {
  return { type: "number", description };
}

export function registerTools(api, client, cfg) {
  // ── noldomem_recall ──
  api.registerTool(
    (ctx) => {
      const agent = resolveAgentId(ctx);
      if (!agent) return null;
      return {
        name: "noldomem_recall",
        label: "NoldoMem Recall",
        description:
          "Search your own NoldoMem history when relevant context is missing for questions " +
          "about prior work, decisions, dates, people, preferences, todos, or anything discussed " +
          "in previous sessions. Returns top matching memories with relevance scores. " +
          "Use proactively when needed; do not repeat a search already supplied by automatic recall.",
        parameters: objectSchema(
          {
            query: stringSchema("Natural language search query"),
            limit: numberSchema("Max results (default: 5)"),
            include_history: { type: "boolean", description: "Include previous versions." },
            as_of: numberSchema("Validity time as Unix seconds."),
            memory_type: stringSchema(
              "Filter by type: fact, preference, rule, conversation, lesson, other"
            ),
            namespace: stringSchema("Memory namespace. Omit to search all namespaces."),
          },
          ["query"]
        ),
        async execute(_toolCallId, params) {
          try {
            const body = {
              query: params.query,
              limit: params.limit || cfg.recallLimit,
              agent,
              max_tokens: cfg.recallMaxTokens,
            };
            for (const key of ["include_history", "as_of"]) {
              if (params[key] !== undefined) body[key] = params[key];
            }
            const namespace =
              typeof params.namespace === "string" && params.namespace.trim()
                ? params.namespace.trim()
                : undefined;
            const memoryType = normalizeRequestedMemoryType(params.memory_type);
            if (namespace) body.namespace = namespace;
            if (memoryType) body.memory_type = memoryType;

            const data = await client.recall(body);
            const text = formatRecallResults(data);
            return {
              content: [{ type: "text", text }],
              details: {
                count: (data.results || []).length,
                search_mode: data.search_mode,
                agent,
                memories: (data.results || []).map((r) => ({
                  id: r.id,
                  text: (r.text || "").slice(0, 300),
                  memory_type: r.memory_type,
                  score: r.score,
                })),
              },
            };
          } catch (err) {
            return {
              content: [
                {
                  type: "text",
                  text: `Memory recall failed: ${err.message || err}`,
                },
              ],
              details: { error: String(err) },
            };
          }
        },
      };
    },
    { name: "noldomem_recall" }
  );

  // ── noldomem_store ──
  api.registerTool(
    (ctx) => {
      const agent = resolveAgentId(ctx);
      if (!agent) return null;
      return {
        name: "noldomem_store",
        label: "NoldoMem Store",
        description:
          "Store important information in long-term memory. Use for decisions, " +
          "preferences, lessons learned, configuration changes, or any fact that " +
          "should persist across sessions. For a confirmed user correction, set supersedes " +
          "to the prior recalled ID. Do not supersede facts using model inference.",
        parameters: objectSchema(
          {
            content: stringSchema("The information to remember (be specific and concise)"),
            namespace: stringSchema("Memory namespace (default: default)"),
            source: stringSchema("Source label (default: agent-tool)"),
            supersedes: stringSchema("ID of the prior assertion being explicitly corrected."),
            valid_from: numberSchema("Validity start as Unix seconds; omitted means now."),
          },
          ["content"]
        ),
        async execute(_toolCallId, params) {
          try {
            const data = await client.store({
              text: params.content,
              supersedes: params.supersedes,
              valid_from: params.valid_from,
              agent,
              session_id: ctx.sessionKey || ctx.sessionId,
              source: params.source || "agent-tool",
              namespace: params.namespace || cfg.defaultNamespace,
            });
            return {
              content: [
                {
                  type: "text",
                  text: `Memory stored: "${params.content.slice(0, 100)}${params.content.length > 100 ? "..." : ""}"`,
                },
              ],
              details: data,
            };
          } catch (err) {
            return {
              content: [
                {
                  type: "text",
                  text: `Memory store failed: ${err.message || err}`,
                },
              ],
              details: { error: String(err) },
            };
          }
        },
      };
    },
    { name: "noldomem_store" }
  );

  // ── noldomem_pin ──
  api.registerTool(
    (ctx) => {
      const agent = resolveAgentId(ctx);
      if (!agent) return null;
      return {
        name: "noldomem_pin",
        label: "NoldoMem Pin",
        description:
          "Pin a critical memory so it survives decay, garbage collection, and consolidation. " +
          "Use for non-negotiable rules, key credentials info, or architectural decisions.",
        parameters: objectSchema(
          {
            memory_id: stringSchema("The memory ID to pin"),
          },
          ["memory_id"]
        ),
        async execute(_toolCallId, params) {
          try {
            const data = await client.pin({
              id: params.memory_id,
              agent,
            });
            return {
              content: [
                { type: "text", text: `Pinned memory ${params.memory_id}.` },
              ],
              details: data,
            };
          } catch (err) {
            return {
              content: [
                {
                  type: "text",
                  text: `Pin failed: ${err.message || err}`,
                },
              ],
              details: { error: String(err) },
            };
          }
        },
      };
    },
    { name: "noldomem_pin" }
  );
  api.registerTool((ctx) => {
    const agent = resolveAgentId(ctx);
    if (!agent) return null;
    return {
      name: "noldomem_forget",
      label: "NoldoMem Forget",
      description: "On an explicit user forgetting request, delete the selected memory and its connected previous versions from NoldoMem. Further ingestion from identified source sessions is blocked until explicit relearning. Original transcripts and other stores are separate.",
      parameters: objectSchema({ memory_id: stringSchema("ID of the memory to forget, including its revision family.") }, ["memory_id"]),
      async execute(_toolCallId, params) {
        try {
          const data = await client.forget({ id: params.memory_id, agent });
          return { content: [{ type: "text", text: data.deleted ? `Memory and its revision history forgotten. Relearning source keys: ${JSON.stringify(data.source_keys || [])}. Records without replay protection: ${data.unidentified_records ?? "unknown"}.` : "Memory not found." }], details: data };
        } catch {
          return { content: [{ type: "text", text: "Memory forgetting failed." }], isError: true };
        }
      },
    };
  }, { name: "noldomem_forget" });

  api.registerTool((ctx) => {
    const agent = resolveAgentId(ctx);
    if (!agent) return null;
    return {
      name: "noldomem_relearn_source",
      label: "NoldoMem Relearn Source",
      description: "Only on an explicit user request to learn again from a forgotten source session, unblock that exact session. Never use automatically after a rejected capture. Does not restore deleted content.",
      parameters: objectSchema({ session_id: stringSchema("Exact original source session ID, or provide source_key."), source_key: stringSchema("Opaque source key from the forgetting receipt. Provide this OR session_id, not both.") }),
      async execute(_toolCallId, params) {
        try {
          const data = await client.relearnSource({ agent, session_id: params.session_id, source_key: params.source_key, confirm: true });
          return { content: [{ type: "text", text: data.cleared ? "Source unblocked for future ingestion. Deleted content was not restored." : "No source block found." }], details: data };
        } catch {
          return { content: [{ type: "text", text: "Source relearning authorization failed." }], isError: true };
        }
      },
    };
  }, { name: "noldomem_relearn_source" });

}
