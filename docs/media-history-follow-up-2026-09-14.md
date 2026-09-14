# Recovering media provenance from the admitted source

The bounded follow-up exposed a real gap in candidate `0e7c5845`: the model
read the synthetic PNG correctly, and native hooks fired, but the completion
snapshot lacked the media fields required by automatic response capture.
The same source row retained those fields in the official `chat.history` API.
The scoped fix uses that existing surface; it does not require another host PR.

## Actual applications, before this fix

OpenClaw **2026.9.3**, Node **24.19.0**, Codex plugin **2026.9.3**,
`openai/gpt-5.6-sol`, native Codex harness, normal temporary Gateway lifecycle.
The primary synthetic memory was empty; only the other synthetic agent had a
guide record. The existing synthetic entrance PNG was attached as binary input.
Neither the expected answer nor image contents were supplied as text.

Two application launches consumed **10.900 seconds** in total:

1. The first stopped in **1.208 seconds** before opening a WebSocket: the test
   server used loopback auth-none, but the CLI configuration still expected
   credentials. It consumed an application slot. Aligning only the temporary
   configuration through the official CLI repaired this harness error; a native
   health RPC then passed before the remaining application. No OAuth retry,
   credential copy or production setting change was needed.
2. The remaining application was used for image learning because the first had
   not reached a model. Native execution succeeded in **8.505 seconds** and
   answered “Use the **blue door** as the entrance.” Automatic recall and
   `agent_end` were observed, but **zero memories were stored**. No second-session
   model answer was attempted after the allowance ended.

Native usage reported **10,942 total tokens**: 6,576 input, 14 output and 4,352
cache-read tokens. Physical provider requests/retries are unknown. The first
launch failed before provider submission; two application launches are not two
API calls. The installed Codex package was already present in the new profile;
an attempted duplicate install was refused and was not forced or deleted.

## Source and product boundary

Pinned host commit:
[`1391f7cd`](https://github.com/openclaw/openclaw/tree/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7).

- [`prepareChatSendUserTurn`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/server-methods/chat-send-user-turn.ts)
  persists admitted media and ordered image slots.
- [`resolveFinalCodexMirrorMessages`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/user-prompt-message.ts)
  returns enriched rows to
  [`transcript-mirror.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/transcript-mirror.ts).
  However,
  [`run-attempt-finalize.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/run-attempt-finalize.ts)
  sends the earlier `result.messagesSnapshot` to `agent_end`. The observed input
  had its run-bound idempotency key and submitted context, but no media fields.
- [`buildLlmInputEvent`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/run-attempt-turn-request.ts)
  includes both context and current images in `imagesCount`. That count alone
  cannot establish fresh source ownership.
- The public
  [`gateway-runtime` SDK](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugin-sdk/gateway-runtime.ts)
  exposes `callGatewayFromCli`. NoldoMem uses its native authentication internally
  with read-only client state and the existing `chat.history` method. No raw
  transcript/credential files or private host imports are used by the plugin.

## Scoped correction and risks

`registerAutoCapture` keeps at most 128 pending image-run identities, removing
each on completion and evicting the oldest if a completion never arrives.
It retains no prompt/image data in that set. A positive input count only permits
one bounded history lookup when completion media is absent. Existing complete
metadata needs no lookup; ordinary text turns do not use this path.

The lookup requires explicit conversation access, configured local/loopback
Gateway mode/binding and numeric port,
trusted agent/session/run identity, and reads at most 20 recent messages. Both
session identifiers, the unique source idempotency key, timestamp, content and
visible external-user provenance must match before media fields are accepted.
All existing capture admission, memory-echo, uncertainty and suppression guards
still apply. Unavailable SDK/auth/history, evicted hints, old sources outside
the bounded result and ambiguous/mismatched rows fail closed. This can miss
capture; it must not assign another source's media to the current answer.

Opaque `media://inbound/<filename>` references are preserved without fetching
them. Query strings, fragments, credentials, traversal and other authorities
are rejected. A reference is not OCR, content, durable file retention or proof
of delivery. The resulting episode remains `assistant` / `inferred` /
`generated`, with the host observation time rather than a fabricated event date.
The existing source-session tombstone still blocks replay after forgetting.

## Evidence after the correction

The new positive scenario fails on `0e7c5845` and passes on the candidate.
**34 focused tests** and **205 API/alignment/ingestion neighbors** passed, as did
Ruff. Tests cover stale/context-only images, identity mismatch, cross-agent
hints, bounded eviction, repeated capture, failed reads, unsafe media references,
and API persistence/forgetting/relearning.
An offline wheel build preserved the changed evidence module byte-for-byte;
its `Requires-Dist` entries contain neither NLTK nor Zeyrek. The tracked-only
filename-only secret scan flagged 37 files, including one new synthetic negative
test containing deliberately invalid token-bearing references. That finding was
reviewed as test data without suppressing the scanner or weakening assertions;
the scan is a bounded heuristic, not proof that all secrets are absent.

The installed Gateway loaded the changed candidate. Its public harness SDK
replayed the recorded synthetic `llm_input` and completion events; the candidate
itself queried real `chat.history`, captured the source-qualified response and
injected it into a different synthetic session. Another agent did not receive
the image episode. Repetition deduplicated; forgetting cleared injection and
blocked history reprocessing; explicit API relearning permitted it again.
**No model was called during this replay.** It is native lifecycle/API regression
evidence, not proof of automatic capture during a new model turn or a grounded
second-session model answer.

Four capture observations were 216.8, 41.7, 51.2 and 63.4 ms; the first includes
lazy SDK loading. These do not establish p95 or a general speed improvement.
The [receipt](media-history-results-2026-09-14.json) ties the real run, source,
candidate hashes, usage and subsequent model-free evidence together.

Temporary processes were stopped and the Gateway port closed. Selected
production service PIDs/start times were unchanged. The isolated profile and
OAuth entry are deliberately retained at the user's request until related
verification finishes; no logout/revoke was performed.

## Follow-up status

The subsequently authorized [two real applications](media-native-proof-2026-09-14.md)
completed the image capture/cross-session answer check described below. This
section records the plan at this earlier checkpoint, not another permission
request. The earlier failed allowance remains consumed.

### Original follow-up plan

No fresh login is needed while this isolated access remains usable. The smallest
remaining complete model check is **two additional applications**, at most
120 seconds each / 240 seconds total, using the same profile and synthetic PNG.
Before that check, clear only the synthetic acceptance memory through the API
and verify an empty primary memory; use fresh sessions so the replayed result
cannot seed the image answer. First require model-triggered capture from raw
input, then an indirect second-session entrance/unknown-guide question with
automatic injection. This proposal is not additional model-call authorization.

Hermes's selected STT prerequisite, the prior OpenClaw audio/PDF limits, real
outbound attachment delivery, pending host contributions and irrecoverable
legacy provenance remain separate. This correction does not close them.
