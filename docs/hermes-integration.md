# Hermes Integration

NoldoMem can be used by Hermes through a native Hermes `MemoryProvider`.

Hermes should call the public NoldoMem HTTP API. NoldoMem remains responsible
for storage, embedding, hybrid search, decay, and reranking. Hermes should not
duplicate the vector database, embedding server, or reranker logic.

For production setups that already use NoldoMem as the durable memory system,
prefer making NoldoMem the single long-term memory source:

```yaml
memory:
  provider: noldomem
  memory_enabled: false
  user_profile_enabled: false

agent:
  disabled_toolsets:
    - memory
```

This disables Hermes' native `MEMORY.md` / `USER.md` prompt injection and hides
the built-in `memory` tool, while keeping the NoldoMem provider and
`session_search` available. Running both long-term memory systems at once is
possible, but it can create duplicate or conflicting facts.

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
  "query": "what was the embedding server issue?",
  "agent": "hermes",
  "namespace": "default",
  "limit": 5
}
```

Example store request:

```json
{
  "text": "The user prefers concise Turkish status updates.",
  "agent": "hermes",
  "namespace": "default",
  "memory_type": "preference"
}
```

Example pin request:

```json
{
  "id": "memory-id-to-pin",
  "agent": "hermes"
}
```

## Memory Types

Hermes integrations should use only the public `memory_type` enum:

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

Hermes provider implementations should:

- bound recall by result count and character budget
- prefetch recall in the background when possible
- keep completed-turn storage off the user response path
- use short HTTP timeouts
- degrade gracefully when NoldoMem is unavailable
- avoid logging raw headers, API keys, or full private payloads
- skip cron, subagent, or system-context writes unless explicitly enabled

## Tool Names

Use the same tool names as the OpenClaw plugin when exposing explicit memory
tools:

- `noldomem_recall`
- `noldomem_store`
- `noldomem_pin`

Shared names make NoldoMem recognizable across runtimes.

## Recommended Hermes Provider Shape

The native Hermes provider should implement Hermes' `MemoryProvider` lifecycle:

- `is_available()` checks endpoint and credentials without network calls
- `initialize()` loads endpoint, API key source, agent scope, namespace, and limits
- `prefetch()` returns cached recall context quickly
- `queue_prefetch()` performs background recall for the next turn
- `sync_turn()` queues completed-turn storage
- `get_tool_schemas()` exposes explicit memory tools when enabled
- `handle_tool_call()` maps tool calls to NoldoMem HTTP endpoints
- `shutdown()` flushes queued writes

The provider should treat NoldoMem as an external service. If NoldoMem is slow
or unavailable, Hermes should continue the conversation without memory context
instead of blocking the reply.

The repository ships a ready adapter at
[`adapters/hermes/noldomem`](../adapters/hermes/noldomem).
