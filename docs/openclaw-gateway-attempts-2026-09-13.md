# OpenClaw Gateway acceptance attempts, 2026-09-13

**No new model acceptance passed.** Two authorized application attempts stopped
before an observed model-input event, using 9.82 seconds in total. A separate
model-free preflight exposed and verified a fix for short-event capture. The
[receipt](openclaw-gateway-attempts-2026-09-13.json) records failures as well as the
successful capture evidence; earlier Hermes results are unchanged.

The tested candidate was `da36ebdcf3f9600377b7aafb265596d8774ee0d1`, with matching
source digests in the receipt. The host was OpenClaw **2026.9.3**, commit
[`1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`](https://github.com/openclaw/openclaw/tree/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7),
Node **24.19.0**, selecting OpenAI `gpt-5.6-sol` through the Codex harness. This
identifies the intended runtime, not a successfully executed model. The existing
Codex package was reused read-only; a fresh isolated profile had native usable
access. No production memory, sessions or configuration were used.

## Product defect found before model execution

The original capture filter accepted preferences, explicit memory cues and longer
messages, but discarded these short synthetic declarations:

- “My Aurora observatory booking is Friday at 19:30.”
- “The Aurora observatory guide is Kestrel.”

Under the actual stable Gateway loader and public harness lifecycle SDK, three
synthetic `agent_end` events produced only **one** store request on `4bdebdc`: the
preference. The preflight assertion failed before any application attempt.

`plugin/src/hooks.js::shouldCapture` now also recognizes bounded English/Turkish
event-declaration forms through `isShortEventFact`. The original source strings
were not padded, reworded or supplied as hand-written model context. On the fixed
candidate, the same SDK events produced three successful automatic store requests:
two records for one synthetic agent and a separate guide record for the other.
The API export retained the exact booking/preference text, source sessions and
null validity boundaries. No calendar date or timezone was invented by capture.

The regression initially failed, then passed with the fix. The focused alignment
and plugin-package suites passed **28 tests**. Independent short appointment,
organizer and Turkish declarations are included, with negative question,
speculation, prompt-injection, assistant-role and missing-scope cases. Ruff and
diff checks passed. The bounded filename-only secret scan returned the same 36
heuristic filenames; it is not a secret-free certificate. No dependency changed.

This is a capture heuristic, not general semantic fact extraction. Unrecognized
short forms can still be missed. The native preflight manually supplied SDK
completion events: it proves loader/hook/API/storage behavior, not a model's
learning, automatic correction, reply grounding or model-triggered injection.

## The two application attempts

| Attempt | Intended observation | Actual result | Wall time |
| --- | --- | --- | --- |
| 1 | Undated natural correction using prior captured memory | Gateway admission failed: `published reply runtime missing`. No model-input hook observed. | 2.81 s |
| 2 | Remaining combined correction/current-versus-prior/booking/unknown-guide question | Normal startup settled, but Codex's lazy state access failed with `trusted-plugin-state-origin-path`. No memory HTTP calls or model-input hooks occurred in this attempt. | 7.00 s |

Both count against the two-application allowance. Neither timed out. No extra
application attempt was started. Physical model requests and token usage were not
measured; no successful model execution or answer was observed. Combining the
second prompt could not have established a subsequent session *after* correction,
even had it succeeded. The original two-session requirement remains open.

The causes are distinct and supported by the pinned host source:

1. **Startup rehearsal is insufficient for agent RPC.**
   [`startGatewayPostAttachRuntime`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/server-startup-post-attach.ts)
   returns early in `candidateCanary` mode, before normal sidecar model/reply
   runtime preparation. The
   [agent admission phase](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/agent-turn/agent-run-admission-phase.ts)
   requires that published reply runtime. The earlier model-free rehearsal
   remains valid for SDK hook dispatch, but using it for the first real attempt
   was a test-harness error. The second attempt used normal `startGatewayServer`
   and awaited its documented `startupSettled` promise, with channels and
   scheduled work disabled in the temporary profile.
2. **Loading, harness ownership and trusted state are different gates.**
   A `plugins.load.paths` entry alone has config origin. The
   [harness registrar](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/registry-registrars-providers.ts)
   rejected native compaction ownership in that initial setup. Official local
   `plugins install --link <existing-package>` registration, followed by removal
   of the config-path override, made the same package discoverable as global.
   This repaired registration without downloading or copying the package.
   It did **not** grant trusted state: `resolvePluginTrust` in
   [`manifest-registry.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/manifest-registry.ts)
   explicitly classifies local install records as `origin-path`.
   [Codex's lazy binding store](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/index.ts)
   then hits the
   [`openSyncKeyedStore` trust assertion](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/registry-runtime.ts).
   Usable OAuth and plugin registration did not prove this later gate would pass.

The host requests an official npm/ClawHub installation for trusted plugin state.
No install receipt was forged, no guard was relaxed, and no production plugin or
account was changed. This does not establish a defect in the production memory
path or a need for a new host bridge.

## Acceptance mapping and next boundary

- **Implicit/multimodal/temporal recall:** short-event capture improved; real
  natural correction, current/old answer grounding and automatic model-triggered
  capture/injection remain unverified by these attempts. Previous timestamp and
  unsupported date/timezone failures are not reclassified as passes.
- **Isolation/privacy:** the unchanged synthetic sources stayed in their own
  agent stores. The temporary Gateway/API, test root and runner were removed;
  selected production process IDs were unchanged. No logout or revoke was used,
  and the installed Codex artifact was preserved. Answer-level isolation still
  lacks a successful new model response.
- **Quality/performance/delivery:** the focused regression and code-commit CI
  passed. The timings above describe failed application startup, not retrieval
  latency or a speed improvement. Final-head CI is tracked on PR #34; independent
  automated review remains quota-unavailable, not clean.
- **External host work:** Hermes #107369 CI/audit and OpenClaw #109370's settled
  attachment design are separate outstanding dependencies. Raw paid media
  extraction, real outbound attachment delivery and irrecoverable legacy
  provenance remain unchanged limitations.

A further model run requires a new bounded authorization and new isolated access
after cleanup. Before requesting browser approval again, prepare the temporary
profile with an official pinned `@openclaw/codex@2026.9.3` installation, subject to
the existing package/supply-chain approval boundary, and verify trusted-state
access without inference. The official `openclaw codex sessions --agent synthetic-memory --json`
command on the new empty Gateway reaches the
[`managedThreads.snapshot()` path](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/session-catalog-listing.ts);
verify that actual catalog/store operation, not only auth status or plugin metadata.
No production hosts or sessions may be attached to that profile.
Local linkage is not a substitute. Only after that
preflight passes should the two-session correction/answer test be attempted.
No new package installation, OAuth login or model allowance is implied by this
proposal. Full original acceptance remains incomplete.
