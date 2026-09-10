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
```

This disables Hermes' native `MEMORY.md` / `USER.md` prompt injection while
keeping the NoldoMem provider and `session_search` available. Two independent long-term writers are not a supported synchronized store:
native replacement/deletion does not identify external records. For small curated
sets, evaluate native-only first. See the [stable comparison](platform-memory-alignment-2026-09-09.md).

Hermes v2026.5.28 gates `MemoryProvider` tools behind the `memory` toolset when
an explicit toolset list is configured. If a platform/profile uses
`platform_toolsets` or another explicit `enabled_toolsets` path, make sure the
effective toolsets still include `memory`; otherwise `noldomem_recall`,
`noldomem_store`, `noldomem_pin`, `noldomem_forget`, and
`noldomem_relearn_source` will not be injected. If you need to hide
Hermes' built-in file-backed `memory` tool, verify the live tool surface after
changing toolsets instead of assuming the external provider remains visible.

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
  "session_id": "20260528_231900_ab12cd",
  "namespace": "default",
  "memory_type": "preference"
}
```

`session_id` is optional. When supplied, NoldoMem stores it as
`source_session` for provenance. Automatic recall retains supplied media
`modality`, `representation`, `observed_at` and `confidence` alongside assertion
and delivery labels, within the existing context budget. Labels are constrained
and numeric metadata must be finite and valid; absent legacy fields are omitted.
This does not infer media origin from ordinary quotations or restore metadata
that the host discarded before invoking the provider.

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
- use one bounded current-query prefetch; do not repeat completed-query searches without a reusable cache
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
- `noldomem_forget`
- `noldomem_relearn_source`

Shared names make NoldoMem recognizable across runtimes.

## Recommended Hermes Provider Shape

The native Hermes provider should implement Hermes' `MemoryProvider` lifecycle:

- `is_available()` checks endpoint and credentials without network calls
- `initialize()` loads endpoint, API key source, agent scope, namespace, and limits
- `prefetch()` returns cached recall context quickly
- `queue_prefetch()` performs background recall for the next turn
- `sync_turn()` queues completed-turn storage
- `on_session_switch()` updates cached session scope after `/resume`,
  `/branch`, `/reset`, `/new`, and context compression
- `get_tool_schemas()` exposes explicit memory tools when enabled
- `handle_tool_call()` maps tool calls to NoldoMem HTTP endpoints
- `shutdown()` flushes queued writes

The provider should treat NoldoMem as an external service. If NoldoMem is slow
or unavailable, Hermes should continue the conversation without memory context
instead of blocking the reply.

The repository ships a ready adapter at
[`adapters/hermes/noldomem`](../adapters/hermes/noldomem).

`noldomem_forget` accepts a recalled `memory_id` for an explicit user forgetting
request. It deletes that assertion and its revision family in the current agent
scope; original transcripts and other stores remain separate.

For explicit forgetting and relearning, see [source replay protection](forgetting-sources.md).
The source-session block is agent-local; unkeyed legacy data has no replay guarantee.
