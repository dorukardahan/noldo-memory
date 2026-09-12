/** Scope comes from the trusted host context, never model-supplied parameters. */
export function resolveAgentId(ctx) {
  const explicit = typeof ctx?.agentId === "string" ? ctx.agentId.trim() : "";
  const key = ctx?.sessionKey || ctx?.agentSessionKey || "";
  const sessionAgent = typeof key === "string" ? key.match(/^agent:([^:]+):/)?.[1] : "";
  if (explicit && sessionAgent && explicit !== sessionAgent) return null;
  const agent = explicit || sessionAgent;
  return agent && agent !== "all" && /^[a-z0-9][a-z0-9_-]{0,63}$/.test(agent) ? agent : null;
}
