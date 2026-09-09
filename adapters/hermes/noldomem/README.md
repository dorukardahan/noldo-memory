# NoldoMem Hermes MemoryProvider

This adapter lets Hermes Agent use NoldoMem through Hermes' native
`MemoryProvider` contract.

## Install

Copy this directory into a Hermes profile:

```bash
mkdir -p "$HERMES_HOME/plugins/noldomem"
rsync -a adapters/hermes/noldomem/ "$HERMES_HOME/plugins/noldomem/"
```

Set Hermes memory config:

```yaml
memory:
  provider: noldomem
  memory_enabled: false
  user_profile_enabled: false
```

This makes NoldoMem the long-term memory source and prevents Hermes'
`MEMORY.md` / `USER.md` prompt injection from drifting away from the semantic
store. Keep `session_search` enabled if you still want transcript search.

Hermes v0.19 injects external `MemoryProvider` tools only when the effective
toolsets are unset or include `memory`. If your platform/profile uses an
explicit toolset list, keep `memory` enabled and verify that `noldomem_recall`,
`noldomem_store`, and `noldomem_pin` appear in the live tool surface. Builds
that expose `memory.external_tools_enabled_when_memory_toolset_disabled` may
use that explicit compatibility option instead. Use a distinct `agent` value
for each Hermes identity rather than sharing the legacy `hermes` scope.

## Configure

Create `$HERMES_HOME/noldomem.json`:

```json
{
  "base_url": "http://127.0.0.1:8787",
  "api_key_file": "/path/to/noldomem-api-key",
  "agent": "hermes-primary",
  "namespace": "default",
  "recall_limit": 5,
  "recall_max_chars": 3500,
  "timeout_seconds": 8.0,
  "prefetch_enabled": true,
  "sync_prefetch_on_miss": true,
  "sync_turns_enabled": false,
  "tools_enabled": true,
  "recall_cache_ttl_seconds": 0.0,
  "recall_cache_max_entries": 128
}
```

Secrets can also be supplied through `NOLDOMEM_API_KEY` or
`NOLDOMEM_API_KEY_FILE`. Do not put secrets in committed config files.
Configuration is published to the provider during initialization; later
availability checks do not hot-reconfigure an active recall scope.

## Tools

The provider exposes:

- `noldomem_recall`
- `noldomem_store`
- `noldomem_pin`

The provider also tracks Hermes session rotations. Store calls include the
current Hermes `session_id`, and NoldoMem preserves that value as
`source_session`.

The provider degrades silently when NoldoMem is unavailable. User replies should
not block on memory backend outages.

## Lifecycle and cache behavior

Hermes v0.19 runs `queue_prefetch` and `sync_turn` in its host-owned background
executor. The adapter performs those operations synchronously inside that lane
and does not create a nested thread per turn. The explicit `noldomem_store` tool
also remains synchronous: one tool call performs one direct store and returns
the backend result.

Hermes invokes `on_memory_write` inline for successful built-in memory writes,
so that compatibility hook is synchronous too. The adapter owns no worker
threads or queues. It keeps only a bounded-lifetime registry of currently
executing backend calls so shutdown can fence recall, store, pin, turn sync,
queued prefetch, and built-in-memory hooks consistently.

Shutdown marks the provider as closing before it waits, rejects every late
backend call, and clears the client and cache when its bounded wait ends. Its
provider-local five-second deadline is anchored to the oldest operation already
admitted by the adapter. Hermes v0.19 does not pass its earlier host-executor
drain deadline into provider shutdown, so the adapter does not claim one shared
deadline across both lifecycle stages. Because Python cannot force-stop an
external caller thread, a call may finish after the provider-local deadline,
but the adapter discards any tool response that arrives after close and cannot
return or cache recalled context. A remote store or pin may therefore have an
indeterminate backend outcome if shutdown wins that race. The real HTTP request
remains bounded by `timeout_seconds`.

A provider instance can be initialized again after a clean shutdown. If the
previous shutdown deadline expired while an admitted operation was still
running, reinitialization fails closed until that operation has drained; this
prevents an old-lifecycle response from crossing into the new provider state.
Request construction also captures the active lifecycle generation, and
network admission rejects a body if shutdown, session switch, or
reinitialization happened before that body reached the client.
Host-lane recall and write callbacks register before snapshotting request state,
so shutdown cannot miss an old callback and let it adopt a later lifecycle.
Admission captures the current session generation atomically; a session switch
before request snapshotting therefore invalidates rather than retargets the callback.

Hermes v0.19 owns the executor submission queue used before the provider's
`queue_prefetch` method is invoked. The provider does not add another queue or
thread, but the provider contract cannot coalesce calls that are still waiting
inside that host-owned queue. Bounding that host backlog requires a Hermes-side
executor policy rather than an adapter change.

Recall can use an opt-in thread-safe LRU cache. The default TTL is zero so writes
from another session cannot leave a provider-local stale copy. The server owns
shared search caching. Completed-query background prefetch is skipped at TTL zero;
it does not predict the next user prompt. Own writes also invalidate local cache. `recall_cache_ttl_seconds` controls
expiry and `recall_cache_max_entries` controls the maximum entry count. The
cache is cleared on session changes, reset, Hermes `rewound=True` notifications,
legacy rewind reasons, compaction boundaries, and shutdown. Cache identity uses
the effective agent, namespace, recall limit, output limit, session ID, and
outbound query as structured fields. Query case and internal whitespace are
preserved exactly as sent while only leading/trailing whitespace is normalized.
This prevents cross-session, cross-namespace, delimiter, and case-only request
aliasing.

Recall payloads are sanitized with Hermes v0.19 fence semantics. A complete
`<memory-context>...</memory-context>` block is discarded rather than unwrapped,
so provider data cannot smuggle a nested trusted-memory fence into the prompt.

Both cache settings are validated. Supported ranges are:

- `recall_cache_ttl_seconds`: 0–3600 seconds (default 0; disabled)
- `recall_cache_max_entries`: 1–4096

Equivalent environment variables are available with the `NOLDOMEM_` prefix,
for example `NOLDOMEM_RECALL_CACHE_MAX_ENTRIES`.

## Diagnostics and readiness

Availability and tool discovery are configuration-only and never contact the
backend. The default doctor command follows the same rule:

```bash
python adapters/hermes/noldomem/doctor.py
```

It reports only safe booleans and endpoint classifications. It does not print
the endpoint host/path, API key, agent, namespace, response body, raw exception
text, or memory content.

Use `--live` only when an explicit bounded readiness read is wanted:

```bash
python adapters/hermes/noldomem/doctor.py --live --timeout 2
```

The timeout is capped at two seconds and enforced as one outer wall deadline on
POSIX main-thread calls, in addition to the transport timeout. If that deadline
cannot be installed safely (for example, from a non-main thread), the probe
fails closed without making a network call. The response body is size-bounded,
and only the health status, storage/embedding booleans, numeric uptime, and an
allowlisted error class are exposed. Exit codes are `0` for configured/default
or live-ready, `1` for unconfigured, and `2` for a completed or safely refused
live probe that is not ready.

## Stable-host alignment (2026-09-09)

The real Hermes `v2026.9.7` provider loader and MemoryManager were exercised with
an isolated HTTP API and temporary native MemoryStore. See the
[comparison](../../../docs/platform-memory-alignment-2026-09-09.md).

`sync_turn(..., messages=...)` preserves text blocks from the latest turn and
labels assistant/tool output generated, never delivery-confirmed. It does not
fetch attachments or transcribe media. `recall_min_semantic_score` is an optional
JSON-config admission floor for automatic prefetch; calibrate it for the actual
embedding model. Explicit recall tools do not inherit that floor.

Use `supersedes` on a store for a confirmed user correction. `valid_from` is Unix
seconds, defaulting to learning time when omitted. Recall supports `as_of` and
`include_history`. Ordinary queries select currently valid records.

Native write mirroring cannot propagate `replace` or `remove` from substrings to
external IDs and now refuses those operations instead of recapturing deleted text.
Do not operate it as two independent authorities. Keep native session search and
skills separately; a native-only curated memory can be preferable for small sets.

`noldomem_forget` accepts a recalled `memory_id` for an explicit user forgetting
request. It deletes that assertion and its revision family in the current agent
scope; original transcripts and other stores remain separate.
