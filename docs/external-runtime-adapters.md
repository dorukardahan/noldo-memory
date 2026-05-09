# External Runtime Adapters

NoldoMem can be used by agent runtimes outside OpenClaw through a small adapter
that calls the public HTTP API.

The adapter should treat NoldoMem as an external service. NoldoMem remains
responsible for storage, embedding, hybrid search, decay, and reranking. The
agent runtime should not duplicate the vector database, embedding server, or
reranker logic.

## Required API

| Purpose | Method | Endpoint |
|---------|--------|----------|
| Recall memories | `POST` | `/v1/recall` |
| Store memories | `POST` | `/v1/store` |
| Pin critical memories | `POST` | `/v1/pin` |

Authentication uses the `X-API-Key` header.

Example recall request:

```json
{
  "query": "what should I remember about this project?",
  "agent": "external-agent",
  "namespace": "default",
  "limit": 5
}
```

Example store request:

```json
{
  "text": "The user prefers concise status updates.",
  "agent": "external-agent",
  "namespace": "default",
  "memory_type": "preference"
}
```

Example pin request:

```json
{
  "id": "memory-id-to-pin",
  "agent": "external-agent"
}
```

## Memory Types

External adapters should use only the public `memory_type` enum:

- `fact`
- `preference`
- `rule`
- `conversation`
- `lesson`
- `other`

Operational labels such as incidents, deployments, config changes, and
decisions should remain in the memory text, `category`, `source`, or
`namespace`. They should not expand the public `memory_type` enum.

## Runtime Guidance

Adapter implementations should:

- bound recall by result count and character budget
- prefetch recall in the background when possible
- keep completed-turn storage off the user response path
- use short HTTP timeouts
- degrade gracefully when NoldoMem is unavailable
- avoid logging raw headers, API keys, or full private payloads
- skip cron, subagent, or system-context writes unless explicitly enabled

## Optional Tool Names

If the external runtime exposes explicit memory tools, reuse the OpenClaw plugin
tool names when possible:

- `noldomem_recall`
- `noldomem_store`
- `noldomem_pin`

Shared names make NoldoMem recognizable across runtimes, but they are not a
protocol requirement.

## Recommended Adapter Shape

An external runtime adapter should provide these lifecycle operations where the
host runtime supports them:

- availability check for endpoint and credentials
- initialization for endpoint, API key source, agent scope, namespace, and limits
- cached recall context for the current turn
- background recall for the next turn
- queued completed-turn storage
- explicit memory tool schemas when enabled
- tool-call mapping to NoldoMem HTTP endpoints
- shutdown flushing for queued writes

If NoldoMem is slow or unavailable, the host runtime should continue the
conversation without memory context instead of blocking the reply.
