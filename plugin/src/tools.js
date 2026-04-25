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

function resolveAgentId(ctx) {
  // ctx from OpenClaw tool execution includes sessionKey
  // Extract agent id from session key pattern "agent:<id>:..."
  const sk = ctx?.sessionKey || ctx?.agentSessionKey || "";
  const match = sk.match(/^agent:([^:]+)/);
  return match ? match[1] : "main";
}

function resolveRequestedAgent(value, ctx) {
  if (typeof value !== "string") return resolveAgentId(ctx);
  const normalized = value.trim();
  if (!normalized) return resolveAgentId(ctx);
  if (normalized === "all") return "all";
  return normalized.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 80) || resolveAgentId(ctx);
}

function formatRecallResults(data) {
  const results = data.results || [];
  if (results.length === 0) return "No relevant memories found.";

  return results
    .map((r, i) => {
      const type = r.memory_type ? `[${r.memory_type}]` : "";
      const score = r.score ? ` (${(r.score * 100).toFixed(0)}%)` : "";
      const text = (r.text || r.content || "").slice(0, 500);
      return `${i + 1}. ${type} ${text}${score}`;
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
    {
      name: "noldomem_recall",
      label: "NoldoMem Recall",
      description:
        "Mandatory recall step: search NoldoMem long-term memory before answering questions " +
        "about prior work, decisions, dates, people, preferences, todos, or anything discussed " +
        "in previous sessions. Returns top matching memories with relevance scores. " +
        "Use this proactively — do NOT wait for the user to say 'remember'.",
      parameters: objectSchema(
        {
          query: stringSchema("Natural language search query"),
          limit: numberSchema("Max results (default: 5)"),
          memory_type: stringSchema(
            "Filter by type: fact, preference, rule, conversation, lesson, other"
          ),
          namespace: stringSchema("Memory namespace. Omit to search all namespaces."),
          agent: stringSchema(
            "Agent scope to search. Defaults to current agent. Use all for cross-agent recall."
          ),
        },
        ["query"]
      ),
      async execute(_toolCallId, params, ctx) {
        const agent = resolveRequestedAgent(params.agent, ctx);
        try {
          const body = {
            query: params.query,
            limit: params.limit || cfg.recallLimit,
            agent,
            max_tokens: cfg.recallMaxTokens,
          };
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
    },
    { name: "noldomem_recall" }
  );

  // ── noldomem_store ──
  api.registerTool(
    {
      name: "noldomem_store",
      label: "NoldoMem Store",
      description:
        "Store important information in long-term memory. Use for decisions, " +
        "preferences, lessons learned, configuration changes, or any fact that " +
        "should persist across sessions. The system auto-classifies the memory type.",
      parameters: objectSchema(
        {
          content: stringSchema("The information to remember (be specific and concise)"),
          namespace: stringSchema("Memory namespace (default: default)"),
          source: stringSchema("Source label (default: agent-tool)"),
        },
        ["content"]
      ),
      async execute(_toolCallId, params, ctx) {
        const agent = resolveAgentId(ctx);
        try {
          const data = await client.store({
            text: params.content,
            agent,
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
    },
    { name: "noldomem_store" }
  );

  // ── noldomem_pin ──
  api.registerTool(
    {
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
      async execute(_toolCallId, params, ctx) {
        const agent = resolveAgentId(ctx);
        try {
          const data = await client.pin({
            memory_id: params.memory_id,
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
    },
    { name: "noldomem_pin" }
  );
}
