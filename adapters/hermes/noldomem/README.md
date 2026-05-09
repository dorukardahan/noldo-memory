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

agent:
  disabled_toolsets:
    - memory
```

This makes NoldoMem the long-term memory source and prevents Hermes'
`MEMORY.md` / `USER.md` prompt injection from drifting away from the semantic
store. Keep `session_search` enabled if you still want transcript search.

## Configure

Create `$HERMES_HOME/noldomem.json`:

```json
{
  "base_url": "http://127.0.0.1:8787",
  "api_key_file": "/path/to/noldomem-api-key",
  "agent": "hermes",
  "namespace": "default",
  "recall_limit": 5,
  "recall_max_chars": 3500,
  "timeout_seconds": 2.0,
  "prefetch_enabled": true,
  "sync_prefetch_on_miss": true,
  "sync_turns_enabled": false,
  "tools_enabled": true
}
```

Secrets can also be supplied through `NOLDOMEM_API_KEY` or
`NOLDOMEM_API_KEY_FILE`. Do not put secrets in committed config files.

## Tools

The provider exposes:

- `noldomem_recall`
- `noldomem_store`
- `noldomem_pin`

The provider degrades silently when NoldoMem is unavailable. User replies should
not block on memory backend outages.
