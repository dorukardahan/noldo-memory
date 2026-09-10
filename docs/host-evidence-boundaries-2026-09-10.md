# Remaining host evidence boundaries, 2026-09-10

This is a narrow follow-up to the [platform comparison](platform-memory-alignment-2026-09-09.md)
and [eight-request acceptance run](model-acceptance-2026-09-10.md), not a new research baseline.
PR #34 started this follow-up at `1ab313184b7e1dbaf2d9debae5341bd5662978ef`, clean, with successful CI.
At that checkpoint no host source was changed. The later authorized candidate work is recorded in [the follow-up](host-candidate-follow-up-2026-09-10.md); production profiles, memory, credentials, embedding models and services remain unchanged.

## Versions and source verification

The official latest non-prerelease endpoints still selected Hermes **v2026.9.7**
([release](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.9.7), published
2026-09-07T22:17:01Z), commit `2237be355906fbe6065ce1815711eee52b2d646e`, and OpenClaw
**v2026.9.3** ([release](https://github.com/openclaw/openclaw/releases/tag/v2026.9.3), published
2026-09-08T14:15:53Z), commit `1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`.
All source references below use these commits, not moving main.

The installed OpenClaw package is 2026.9.3; its running executable is Node 24.19.0.
The temporary API used its existing Python 3.12.3 virtual environment under a separate
unprivileged temporary HOME/DB. Candidate source was loaded separately from the installed plugin.
The inspected Hermes installation is a downstream build, not a byte-identical upstream checkout.
Its `gateway/run_inbound.py`, `gateway/run_turn.py`, `gateway/run_turn_runner.py`,
`agent/memory_manager.py` and `plugins/memory/__init__.py` match the pinned stable bytes.
Other selected files differ; their row-builder/hook-dispatch functions were also read directly and the installed hook catalog lacks `post_transcription`. The local native test uses the verified upstream stable checkout,
not a claim that every installed downstream code path ran. Installation identities and paths
are deliberately excluded from this public report. Two other installation versions were not
resolvable through the selected service metadata; no private runtime files were searched.

Main was sampled once: Hermes `16a408534c4c2364e8e8f14ea7cef82aa80b4505`, OpenClaw
`fdc92faed369bb9fca64e17ed7a3354088d11839`. Hermes' hook catalog and OpenClaw's sent-message emitter
were identical to stable. Hermes' inbound file differs; a further function comparison hit an
HTTP 429 and was stopped. This does not establish a main-only fix or make one available in stable.

## Classification and user effect

| Gap | Classification | What the user can and cannot rely on |
| --- | --- | --- |
| OpenClaw incoming derivative origin/identity/time | 2 → 1, implemented through the official preprocessing hook | Existing transcript/derived text can be recalled with received/derived evidence. A typed caption next to a file remains typed text; a path alone is not image understanding. |
| Hermes event ID and timestamp | 2 → 1, implemented in the provider | Native row identity/time now survive capture. Old rows without those fields are not repaired retrospectively. |
| Hermes successful audio origin | 2 before STT; 3 for the checked post-STT provider row | The booking time spoken in a successfully transcribed clip remains usable text. Exact clip origin and the distinction from typed quotation are absent from the provider row. This does not mean audio content cannot be remembered. |
| OpenClaw complete outgoing attachment delivery | 2 for sending intent, transcript updates and plugin-owned sends; 3 for a universal per-attachment settled observation | Confirmed outgoing text, including an available spoken-text projection, can be remembered. A prepared media URL is not proof that that file reached the user; partial fan-out needs its own receipt. |
| Raw image/audio/document/link extraction | Existing host capability plus 4 for unexecuted byte-to-answer tests | Reuse host extraction where it runs. Native vision can interpret pixels without producing a reusable textual derivative. No NoldoMem decoder or new paid extraction pipeline was added. |
| Full native host/model/channel loop | 4 | Native loader/hook/MemoryManager and previous external inference evidence remain distinct. No new model answer or channel-delivery success is claimed. |
| Unidentified old sources / unlinked old graph data | Missing historical data | New metadata cannot reconstruct an unknown origin or establish ownership of an old graph row. See the migration boundary below. |

## OpenClaw: supported incoming path now used

The stable [`getReplyFromConfig`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/auto-reply/reply/get-reply.ts)
executes configured media/file and link understanding before `emitPreAgentMessageHooks`.
Native harness ownership can skip image/video/file extraction; explicit audio understanding
can still run. This is a configuration/runtime distinction, not a missing memory feature.

[`emitPreAgentMessageHooks`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/auto-reply/reply/message-preprocess-hooks.ts)
calls `deriveInboundMessageHookContext` and `toInternalMessagePreprocessedContext` in
[`message-hook-mappers.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/hooks/message-hook-mappers.ts).
The actual runtime projection carries `bodyForAgent`, `transcript`, `messageId`, `timestamp`,
staged `media`, and pending `originalMedia`. This is stronger evidence than a type declaration.

NoldoMem's new `autoCaptureSource: "preprocessed"` uses `api.registerHook("message:preprocessed", ...)`
in place of inbound `agent_end` capture. It retains the scoped session for forgetting, source event ID,
existing timestamp, derivative trust and a safely projected single-media reference when attributable.
It neither fetches pending URLs nor guesses which of several clips produced a combined transcript.
Text remains usable after an attachment expires. Successful native file-extraction wrappers are reduced to stable content without random envelope IDs; path-only/failure/rendered-image notices are omitted. The extracted content is still screened, labelled derived and injected inside the untrusted memory envelope. Prepared text without a modality label also stays derived because link processors can append unlabelled output. The existing `message_sent` text observer remains.

The mode requires both enabled internal hooks and explicit
`plugins.entries.noldomem.hooks.allowConversationAccess: true`. The plugin checks the grant itself:
legacy internal hooks do not inherit the typed-hook conversation gate. Tools still load if capture
is unavailable; the plugin warns, without silently installing a different writer.
The compatibility default remains `agent_end`. `preprocessed` is for Gateway ingress and does not
promise CLI-only capture. This explicit choice avoids a race-prone cross-hook join or duplicate writers.
No production selection was changed.

Evidence: `tests/test_preprocessed_capture.py` executes the callback, checking grants, scope,
single writer, expired/pending sources, ambiguous multi-media references, typed captions and
prompt-injection rejection. `scripts/check_openclaw_evidence.mjs` loads the candidate through the
installed native loader and dispatches through the real internal-hook dispatcher against temporary
HTTP/storage. Three concurrent copies produce one row; cross-session injection contains the event;
other-agent injection does not. Forgetting blocks replay and cached context, receipt-based relearning
works, and an independent source remains independent. These are synthetic hook events, not a raw
extractor or channel transport. External/model calls: zero. A further check executes the installed UTF-8 file decoder on synthetic bytes, then supplies the pinned file-envelope format to the native hook. Content reaches storage and another session's context; random envelope IDs do not create duplicates. The installed private file-outcome renderer was not callable, so that envelope is explicitly a fixture, not a claim of running the full file-preprocessing stack. This proves plain-text document decoding only, not OCR, ASR, PDF interpretation or channel receipt.

## Hermes: precise loss and proposed smallest host seam

In stable [`GatewayInboundMixin._transcribe_one_clip`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/gateway/run_inbound.py),
a successful backend result becomes `(transcript, '"transcript"')`. `_enrich_inbound_voice` receives
the successful-transcript list, optionally echoes it, and returns only enriched text.
`GatewayTurnMixin._hmwa_prepare_turn` / `_handle_message_with_agent` in
[`run_turn.py`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/gateway/run_turn.py)
pass the event ID/time but not successful clip provenance into the agent's durable user row.
The message type separately controls voice response behavior; that is not provider evidence.

[`_stage_turn_user_message`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/agent/turn_context.py)
retains `platform_message_id` and `timestamp`; `AIAgent._sync_external_memory_for_turn` and
[`MemoryManager.sync_all`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/agent/memory_manager.py)
pass those rows to NoldoMem. The adapter previously discarded both fields; it now preserves valid
supplied values. A host timestamp can be its row-creation time when the source supplied no event time;
it is not independently proven event validity time.

The alternate official surfaces were considered:

- `pre_gateway_dispatch` receives the real `MessageEvent` with media, but runs **before authorization
  and STT**. It proves arrival, not successful transcription. Capturing there would bypass the accepted-turn
  boundary. Rewriting its text would alter the host's deliberate plain-quotation behavior. A profile/session
  keyed cache or task-local context can retain arrival metadata, but cannot itself establish per-clip success
  across pending-event consumption, local fallback and mixed typed text. No such workaround is presented as
  complete provenance.
- `pre_transcription` precedes backend execution. Replacing the transcription provider merely to observe
  output would take over STT selection. The existing
  [post_transcription PR #100991](https://github.com/NousResearch/hermes-agent/pull/100991)
  (head `0e305cb3929824d7e02411852ab079823dfc7ddb`) is **open**, not stable; its proposed contract also excludes
  the local fallback path. It is relevant prior work, not a shipped solution or full event-correlation fix.
- `agent:start` exposes a 500-character message and session identity, not clip results; `pre_llm_call`
  receives already flattened text/history. Native session rows do not recover information never recorded.

`check_hermes_stable.py --evidence-only` executes the actual successful-STT formatter with a synthetic
backend result, the actual row builder, native provider loader/MemoryManager and temporary API. It
proves the event/time fix and cross-session text injection, while asserting the unresolved audio origin.
It does not decode audio. Unit checks reject absent/invalid metadata and do not classify every quotation
as speech.

For complete successful-clip provenance, the smallest proposed host change is **additive structured
origin on the normalized user row**, assembled after enrichment in `gateway/run_inbound.py` and
forwarded by `gateway/run_turn.py` into the existing `persist_user_display_metadata`/row path.
Include clip identity, derivative kind and success/uncertainty, through normal and pending/fallback paths;
do not change prompt text. NoldoMem can consume that extra field without a new service. Compatibility
risk: providers/exporters must tolerate optional metadata and keep it out of authoritative instructions.
At this checkpoint no Hermes files or upstream discussions were changed. The later [candidate follow-up](host-candidate-follow-up-2026-09-10.md) supersedes that implementation status, not the stable-version boundary.

## OpenClaw: outgoing intent is not a settled attachment receipt

The stable `reply_payload_sending` hook exposes media before transport. In
[`deliver-core.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/infra/outbound/deliver-core.ts),
rendering/normalization and adapter capability decisions happen afterward. The no-media-adapter branch
can send the caption and drop the URLs. Media fan-out can send one file and fail on the next.
`recordMessageSentEvent` retains success/content/one message ID; the failure path can retain only the
last successful part's ID. `hookContent` can preserve TTS spoken text, which is useful content already
available to the existing text observer.

[`createMessageSentEmitter`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/infra/outbound/message-sent-hook.ts)
constructs the canonical event explicitly and omits media/source-index/outcomes. This is an actual
loss point, not just an absent type field. The installed-emitter behavior test passes two different media lists and observes the same public event; the actual transcript formatter also maps two distinct resources with the same basename to the same mirror text. Joining an earlier intent by equal caption/session is
ambiguous across retries, concurrent messages, rendering and partial delivery.

Alternatives are useful but narrower:

- `api.runtime.events.onSessionTranscriptUpdate` exposes normalized committed messages and trusted
  agent/session identity without reading runtime files. `openclawDelivery.mediaUrls` can describe
  prepared display ownership before a send, so its presence alone is not a receipt.
- [`mirrorDeliveredPayloads`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/infra/outbound/deliver-transcript.ts)
  can mirror successful text/media into a session when configured. But
  [`resolveMirroredTranscriptText`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/config/sessions/transcript-mirror.ts)
  reduces URLs to basenames, not extracted content or unique per-attachment identities. Mirroring is
  optional/best effort and does not supply a universal complete delivery observer.
- `api.session.workflow.sendSessionAttachment` can return outcomes for a **plugin's own** send. It
  does not observe unrelated native/channel sends; taking ownership of all delivery would be a larger,
  unjustified change. The native run audit projection reports lifecycle outcomes, not file contents.

The narrow host proposal is an additive settled-media receipt in `deliver-core.ts` →
`message-sent-hook.ts` → `message-hook-mappers.ts`/`hook-message.types.ts`: actual per-item outcome,
source reference and stable message/run/session correlation, including partial and replayed delivery.
Do not attach the pre-send list wholesale or make signed URLs public. Existing text fields must remain
compatible. This is needed for **complete native outgoing attachment attribution**, not for remembering
already supplied outgoing text. The later [candidate follow-up](host-candidate-follow-up-2026-09-10.md) records the existing upstream design gate; OpenClaw source remains unchanged.

## Migration and remaining acceptance

The existing [source-replay migration](forgetting-sources.md) is additive and prospective. Full SQLite
backup/recovery preserves deletion markers; memory-only export/import does not carry deletion policy.
Source-less old records and unlinked graph rows cannot be safely assigned invented provenance. A supported
future ingestion can supply genuine source IDs; it does not prove or repair historical ownership. Any
legacy-data reconstruction would need a separately authorized, attributable source and review before
migration, never automatic deletion/reindexing of production data.

The new path reuses text the host already produced. Raw extraction quality, native vision without a
text derivative, complete outbound file delivery and a full model loop remain separate. The previous
eight actual model requests are still the entire spent budget; timeouts are not new success evidence.

A single bounded follow-up model plan, **subsequently authorized but not yet executed**, is at most **10 actual
requests, five per host**, using temporary profiles and existing access: (1) learn a synthetic event
and preference; (2) choose revision from a natural correction; (3) acknowledge after the tool result;
(4) in a fresh session choose a history lookup from a combined current/past/event question;
(5) answer from the real tool/injected evidence while abstaining on an unknown detail. Count every
request, disable automatic retry/fallback, and stop at the cap even if the host chooses more tool turns.
This closes the native inference-loop verification gap only if the expected steps occur. Raw OCR/ASR
and transport tests need their own explicit synthetic input/route scope; these ten requests do not
silently authorize extra extraction calls or prove the proposed host metadata changes.

## Validation receipt

[Machine-readable results and tested source digests](host-evidence-results-2026-09-10.json): 527 local tests passed, one skipped; Ruff and distribution builds passed. The existing full hash-pinned dependency set passed `pip-audit` without exceptions. Wheel metadata still declares neither NLTK nor Zeyrek. One syntax error and one private-renderer availability failure were diagnosed and corrected in the harness; neither is counted as a successful test. Independent automated review remains unavailable due the previously established quota, not clean.
