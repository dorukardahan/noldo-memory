# Native OpenClaw Gateway acceptance, 2026-09-13

Two newly authorized native application turns passed the bounded correction and
cross-session answer scenario. They used **24.23 seconds** total, without timeout.
The [synthetic receipt](openclaw-native-gateway-2026-09-13.json) links native run
IDs, retrieved/injected record IDs, model-selected revision, stored versions and
answers. This supersedes the setup blockers in the
[earlier attempts](openclaw-gateway-attempts-2026-09-13.md), not their failed results.

## Runtime and predetermined expectations

The tested NoldoMem head was `8d9c8afe590b2c2e9ce358d6e403cda0fc992057`.
OpenClaw **2026.9.3**, commit
[`1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`](https://github.com/openclaw/openclaw/tree/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7),
ran on Node **24.19.0**, with the official pinned Codex plugin of the same version
and native `trusted-official` installation provenance. Native terminal receipts
identified **OpenAI / gpt-5.6-sol / Codex harness**, without provider rerouting.

A fresh temporary profile used its own device-code login, normal Gateway startup
with `startupSettled` awaited, candidate plugin, observer hooks and temporary
NoldoMem API/DB. Channels, scheduled work and native memory flushing were off.
No production conversations/memory were read. Retrieval used the temporary
FTS/degraded path with no embedding or reranker service calls; this is not an
embedding comparison. The observer recorded hook evidence without supplying
model context or changing tool results.

Before either model turn, three **model-free** public SDK completion events
captured a booking, a preference and a separate agent's guide name. That seed
is not evidence of model learning. The expected behavior was written before
execution: an undated natural correction must target the existing preference
record and leave validity to the server; the next session must answer the
booking/current/previous question without inventing a guide, calendar date or
timezone. Each application had a 120-second cap, 240 seconds combined.

## Observed behavior

| Application | Native result | Evidence and qualification |
| --- | --- | --- |
| 1, correction | 16.47 s, exit 0 | The model called `noldomem_store` with `supersedes=355af4d82322437d`, the injected old preference ID. The HTTP request omitted `valid_from`; no script issued the revision. The server closed the old interval at exactly the new interval's start, within the observed write interval. |
| 2, distinct session | 7.76 s, exit 0 | Automatic recall/injection supplied the booking, revised preference and original correction text. The answer gave Friday 19:30, guided group now versus quiet/no-group before, and said no guide name was recorded. It added no calendar date or timezone. |

Both real applications emitted `before_prompt_build`, followed by a successful
automatic `/v1/recall`; its result IDs matched the native `llm_input` memory IDs.
The first input contained the prior preference and booking. The second contained
the new preference, booking and captured correction. Neither input or answer
contained the other agent's synthetic guide name. The second prompt had no
explicit recall command and ran with a different native session ID/key.

The second answer can derive the old preference from the captured correction
sentence. It did **not** invoke historical recall or inject the closed revision
itself. The old version and validity chain were independently inspected through
the temporary API export. This proves the observed current/previous answer and
stored chain, not every history-only retrieval scenario.

Both successful native `agent_end` events triggered `/v1/store` with
`source=plugin-auto-capture`, separately from the three seed events. The first
retained the user's actual correction text; the model-selected revision remained
a separate source-backed record. These are real model-triggered capture and
injection observations, not manually dispatched hooks labeled as model turns.

Native usage metadata reported a combined **25,111 total tokens**: 5,195 input,
332 output and 19,584 cache-read tokens, with 108 reasoning tokens reported
separately. These fields are reproduced as reported, not summed twice or treated
as billing proof. Physical transport request/retry counts were not independently
measured. Two application turns are not two guaranteed API requests. No extra
model turn, external retry loop, media call or channel delivery was started.
The small sample supports no general success rate, p95 or speedup claim.

## Product defect found and fixed afterward

The second question was also captured as a reported fact: the old `shouldCapture`
length/keyword fallback admitted it despite its interrogative form. That row is
retained in the receipt as a defect, not concealed or counted as useful learning.

`plugin/src/hooks.js::isQuestionOnly` now excludes recognizable English/Turkish
question-only turns before length/keyword admission. Mixed declarative turns
and explicit memory requests retain existing rules. This is bounded syntax
recognition, not general intent detection or a new language-processing service.

The new regression failed before the fix. Afterward **29 focused alignment and
plugin-package tests** passed, including independent questions and preserved
mixed factual turns. Ruff and direct diff checks passed. The installed stable
Gateway/public-SDK/API/DB regression also passed: the original question caused
no write; a preference caused one write; subsequent same-agent injection succeeded
and the other agent received none. Its three HTTP calls were one store and two
recalls. That follow-up used update-rehearsal startup and synthetic completion
events with **zero model calls**. It validates the capture change independently;
the earlier native model run was not repeated on the final question-filter code.
Both source versions are identified by digests in the receipt.

## Cleanup and original acceptance boundaries

The observation SSH exited 255 during shutdown. A new read-only check confirmed
both application receipts, final HTTP/hook traces, no remaining owned process and
a closed temporary Gateway port. No application was restarted. The temporary
profile, its new OAuth record, runner and later model-free fixture were removed
without logout/revoke. Selected production service PIDs remained unchanged.

- **Implicit/temporal/grounded recall:** the specified two-session scenario now
  has real native model/tool/hook evidence. Previous failed attempts remain
  historical failures; this is not general model compliance or complete recall.
- **Agent isolation:** the conflicting guide was stored only for the other
  agent, absent from primary injection and answers; the model abstained. Broader
  isolation/forgetting regressions remain the separate existing evidence.
- **Quality/performance:** one newly observed capture defect was fixed without
  another model call. Existing baselines and dependency comparisons were not
  rerun or reinterpreted as answer-accuracy benchmarks.
- **Other host/media work:** earlier Hermes native acceptance remains valid and
  was not repeated. Hermes #107369 CI/audit and OpenClaw #109370's settled-media
  design remain separate external gates at the last verified checkpoint. This
  run did not test raw image/audio/PDF extraction or real outbound attachment
  delivery, and cannot restore missing legacy provenance. Prior free local
  extraction evidence is unchanged. No new OCR/ASR service or host bridge was
  added.

The supported recommendation remains one durable memory authority per agent,
with compatible native context/session helpers. This small successful scenario
does not establish that NoldoMem universally beats native memory. Final-head
CI/review is recorded on PR #34; independent automated review remains
quota-unavailable, not clean. Full original acceptance, including the outstanding
host/media boundaries, is not marked complete.
