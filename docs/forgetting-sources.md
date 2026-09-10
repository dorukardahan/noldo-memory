# Forgetting and source replay

Verified with synthetic data on 2026-09-10. This extends the earlier
[model acceptance evidence](model-acceptance-2026-09-10.md); it does not rerun or
upgrade that evidence into full native model/channel coverage.

## Behavior and source identity

`DELETE /v1/forget` still removes the selected memory's connected revision
family, including scheduled versions. It now blocks later ingestion from each
identified **source session** in that family, inside the same agent database.
Other retained memories in those sessions are not deleted. New writes from the
blocked sessions, including new turns, require explicit relearning. A new,
independent session can supply identical text without being blocked.

Session granularity is deliberate: neither stable host supplies a reliable
per-event identity on every supported provider/capture path. NoldoMem uses the
existing `source_session`, populated by API `session_id`, capture message
`session`, and session-import chunk identifiers. It does not infer an identity
from text, similarity, attachment URLs or media filenames. Callers must preserve the
exact session identifier when replaying a source. Changing or stripping that
identity is outside this protection. An `evidence.event_id` alone is not enough.

The additive `forgotten_sources` table has one column, `source_key`: SHA-256 of
the exact source-session identifier. Agent scope comes from the separate DB;
there is no shared block list. The marker contains no text, text hash, memory ID,
summary, vector, URL, timestamp or recoverable content copy. Legacy chunk IDs
can be content-derived, so they are intentionally not retained as tombstones.
A source digest is an identifier, not a claim of anonymous data.

Records without a nonempty source-session identifier can still be forgotten,
but their replay cannot be recognized safely. Their count appears as
`unidentified_records` in the forgetting receipt. Existing deleted records are
not retroactively assigned identities. The implementation does not label
unprotected legacy sources as protected.

## Explicit relearning

A forgetting response includes `source_keys`, opaque receipts for the blocked
sessions. Keep only a receipt if later relearning is desired; deleted text is
not needed. An explicit user request can authorize one source through either
host's `noldomem_relearn_source` tool or:

```http
POST /v1/relearn-source
Content-Type: application/json

{"agent":"alpha","session_id":"synthetic-session-a","confirm":true}
```

Supply **either** `session_id` **or** a `source_key` from the forgetting receipt.
The response is `{"cleared":true,"restored":false}` when a block was removed.
Relearning only allows future ingestion: it does not restore a deleted record
or revise another agent's block. `agent=all` is rejected. Host tools obtain
agent scope from trusted runtime context and instruct the model never to unblock
a rejected capture automatically. Tool execution is tested without a model;
model compliance with a natural relearning request was not tested.

## Writes, derived data and recovery

All storage admission paths, including batch import and explicit revision,
check source markers under a SQLite write transaction. Capture skips blocked
messages and reports `blocked`; single store/rule returns HTTP 409. An import
containing a blocked source fails atomically with 409, including when forgetting
happens during embedding. A batch storage failure creates no derived graph rows.
No new service or dependency is required.

New capture/session-import graph extraction links entities, relationships and temporal
facts to the stored memory. Extraction rechecks that the memory exists while
holding the write lock. Forgetting removes its support links and derived-only
orphans; shared entities remain supported by other memories. Shared relationship
context is cleared conservatively rather than retaining the forgotten text.
Independent manual graph support is preserved. Historical **unlinked** graph
rows cannot be attributed reliably and are not silently relabeled or purged.
This remains a legacy-data limitation, not a guarantee that old unlinked graph
content was erased.

Forgetting clears the agent DB's search/embedding cache and advances recall
invalidation. The API also clears the volatile embedding cache and fences its
in-flight requests. Replacement writes also remove their old vector/FTS entries, so later forgetting
does not miss those copies. A late background worker cannot attach a vector to a
deleted row. Previously orphaned, unlinked vectors cannot be attributed
retroactively. This is logical product deletion, not forensic erasure of SQLite free pages,
backups, already emitted context or host transcripts. No live database was opened
or migrated during this work.

Schema creation is additive at normal initialization. Only temporary databases
were migrated and tested. Existing memory rows are preserved; new graph source
links apply prospectively. An offline SQLite Backup API copy preserves the
markers and graph support table. Memory-only JSON export/import is not a full
backup and does not transfer deletion policy; do not use it as a replacement
for a complete DB recovery. Running an older writer against the upgraded DB
would not enforce the new admission contract and is not a supported rollback.

## Evidence and limits

`tests/test_forgetting_sources.py` exercises real API/storage operations with
synthetic text: replay, changed producer/namespace, agent isolation, independent
sources, explicit receipt-based relearning, missing identities, revision-family
purge, cache and graph removal, additive migration, SQLite backup recovery,
concurrent writers, and forgetting during paused embedding. Embeddings in these
tests are local stubs; no paid or network model call is made.

`scripts/check_hermes_stable.py` uses the pinned Hermes stable provider loader and
MemoryManager with a temporary HTTP API/DB. `scripts/check_openclaw_runtime.mjs`
uses the installed OpenClaw 2026.9.3 native loader, hook runner and tool factories
with the candidate plugin and a separate temporary API/DB. Both test blocked
recapture and explicit relearning through registered tools, along with cross-session
context and supplied media derivatives. OpenClaw uses recorded vectors; Hermes
uses degraded lexical retrieval. Neither test invokes a model or raw extractor.

The earlier eight model requests remain the total inference budget used by the
acceptance run. This correction provides no new generated-answer evidence and
no speedup claim. Raw extraction, complete outgoing attachment delivery,
Hermes audio origin already discarded by the host, and full native host/model
loops retain their separately documented limitations.

Local validation: 519 passed, 1 skipped; Ruff passed; sdist and wheel built.
The complete existing hash-pinned dependency set passed `pip-audit` without
exceptions. The new wheel matches candidate storage code and declares neither
NLTK nor Zeyrek. Dependency resolution and the Turkish quality comparison were
unchanged and were not repeated.

[Verification receipt and tested source digests](forgetting-source-results-2026-09-10.json)
separate the native-tested bytes from the final small API parity corrections.
