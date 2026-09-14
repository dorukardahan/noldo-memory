# Bounded raw-media acceptance: observations on 2026-09-13

Recorded on 2026-09-14. The [authorized plan](raw-media-acceptance-plan.md) was
**partially executed, not passed**. OpenClaw completed two native application
turns. It interpreted the PNG in the first answer but did not retain that detail
for the next session. Audio processing failed; the initial PDF configuration was
incomplete. Hermes stopped before model calls because its selected local STT
backend was unavailable in the isolated runtime.

The [synthetic receipt](raw-media-results-2026-09-13.json) identifies the actual
candidate, host, run/session IDs, inputs, answer text, usage, memory exports and
separate follow-up tests. Earlier successful text/correction acceptance was not
repeated or relabeled as raw-media evidence.

## Execution and predetermined expectations

NoldoMem `8331aebdff1d27f7ddf4d77d2efaaf7a7342ec68` ran with OpenClaw **2026.9.3**,
commit [`1391f7cd`](https://github.com/openclaw/openclaw/tree/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7),
Node **24.19.0**, official pinned Codex plugin **2026.9.3**, and the native
**OpenAI / gpt-5.6-sol / Codex harness**. A fresh device-code login was completed
in its own profile. Native status reported usable access. The normal Gateway
startup awaited `startupSettled`; this was not update rehearsal or a detached
model supplied with hand-written memory context.

The temporary profile had its own workspace, sessions, API and SQLite databases.
Automatic NoldoMem recall/capture and the necessary hook permissions were on;
channels, cron, heartbeat, native memory flushing and external send tools were
off. Retrieval used the existing lexical/degraded API path without embedding or
reranker requests. An independent agent had a synthetic guide record; the tested
agent's memory was initially empty. The three expected media facts were never
seeded into that agent's memory or prompt.

The first `chat.send` attached the exact prepared PNG, WAV and PDF as native
base64 attachment inputs and asked only for practical details. A different
session then asked about entrance, packing, meeting point and guide without
naming media types or asking to remember. Expected facts were blue door, red
notebook and east arch; guide identity was unknown. Input hashes are in the
receipt. The expected-answer manifest stayed outside the host workspace.

## Observed chain

| Input / behavior | Actual observation | Implication |
| --- | --- | --- |
| PNG | Native `llm_input` reported one image. The answer correctly gave the blue door; that fact was absent from the input text. | Raw image interpretation affected this answer. No separately reusable image description was observed. |
| WAV | The configured native OpenAI audio path reported `ProviderHttpError`; model input contained `[Audio attachment could not be analyzed]`. The answer did not invent the packing detail. | ASR did not succeed. The bounded safe trace did not retain an HTTP status or provider rejection reason; this is not evidence of quota exhaustion, broken login or an invalid WAV. |
| PDF, first turn | Model input contained `[Attachment could not be read]`. The temporary plugin allowlist omitted `document-extract`. | This was a test configuration defect. The first model turn did not learn the meeting point. |
| Capture, first turn | Real `agent_end` ran. Its clean user row contained the request to summarize; no store followed, and the primary export stayed empty. | The model's image interpretation did not become persistent memory. Tool visibility and successful inference did not close this gap. |
| Second session | Automatic recall and five model-selected `noldomem_recall` calls found no relevant primary-agent record. The answer said it could not find the details. | Cross-session media recall failed, including the previously interpreted blue door. Absence of memory was not hidden. |
| Isolation / abstention | Neither the other agent's guide nor the later independent PDF-probe record entered the primary prompt or answer. | The observed negative/isolation case passed; it does not establish retrieval quality with populated primary memory. |
| Capture, second turn | The topic-prefixed question itself was stored as a reported fact. | A NoldoMem defect, fixed below; the original erroneous row remains in the receipt. |

The host's native
[`agent_end`](https://github.com/openclaw/openclaw/tree/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server)
projection supplied clean user text alongside `__openclaw.upstreamUserText`.
That latter field included bootstrap/system context and failed-media notices,
not a reusable image description. NoldoMem does not mine that composite prompt
or automatically promote arbitrary assistant answers into user assertions. Its
current default completion capture consumes user text and recognized derivatives;
it cannot infer pixel contents from attachment presence. Retaining source-linked
model interpretations is still an implementation/design gap, distinct from the
question-filter bug and host delivery metadata work.

A focused source check on 2026-09-14 ruled out a count-only capture shortcut.
In stable
[`buildLlmInputEvent`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/run-attempt-turn-request.ts),
`imagesCount` combines current input images with `prompt.contextImages`.
[`run-attempt-prompt.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/run-attempt-prompt.ts)
can prepare those images from restored/assembled history. A positive count alone
therefore cannot identify a fresh source or distinguish it from a replayed one.
The host's existing failed-turn regression also shows that `llm_output` can fire
on failure; that notification alone is not successful capture or delivery.
Saving every answer after a positive image count could misattribute old content
to a new source and undermine source-replay protection. No such writer was added.
This does **not** establish that a host change is required: source correlation
through supported message/session surfaces still needs a bounded design and
behavior check before adding persistence.

The existing `registerAutoCapture` `message_sent` handler can retain successful
outgoing text as derived, delivered assistant evidence; its bounded-content,
failure and scope regressions remain applicable. This run used `deliver:false`
and only NoldoMem's advertised memory tools, so it did not exercise that channel
receipt or native session-search fallback. Its failure must not be generalized
to every production channel or the recommended combination of native helpers.
The unmet target is still automatic, source-grounded cross-session use of media
content, not simply writing more assistant text to make this fixture pass.

The native audio implementation is
[`transcribeOpenAiAudioWithContext`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/openai/audio-transcription.ts).
It supports subscription selection through native auth, but that source capability
is not proof that this actual audio request succeeded. No other account/provider
or external retry loop was substituted after failure.

## Small corrections and independent verification

The temporary harness initially used an incorrect Python argv name and placed
media models under an obsolete per-capability field. Both failures happened
before any model turn. The interpreter was bound to its actual installed path,
and the official CLI accepted `tools.media.models` with an audio capability tag.
The trace observer's prompt-hook grant was corrected for the follow-up; the
first turn's actual automatic recall is separately evidenced by the HTTP request.
A teardown thread-affinity error was also fixed in the temporary controller;
it did not turn the completed first model run into a success for media recall.

For PDF, only the temporary configuration was amended to enable the already
installed `document-extract` plugin. Native
[`extractPdfContent`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/media/pdf-extract.ts)
requires that plugin; an explicit allowlist can exclude it. The same raw PDF then
passed native `renderInboundDocumentContext`, yielding actual text and one page
image. A **model-free public SDK completion**, inside the running Gateway,
captured the text under a separate synthetic agent. A new-session SDK prompt
contained that record with `assertion=derived`, `modality=document` and
`representation=extracted_text`. No model, OCR service or new dependency was
used. This verifies the PDF → extraction → capture → injection chain, not PDF
learning/answer behavior in either real model turn.

In NoldoMem, `plugin/src/hooks.js::isQuestionOnly` now recognizes short English
topic prefixes such as “For the visit, which...?”. Recognizable clauses and mixed
factual turns retain existing admission rules. It is still a bounded syntax
heuristic. The expanded regression failed before the fix; afterward **32 focused
alignment, preprocessing and plugin-package tests** passed, together with Ruff
and direct diff checks. The installed Gateway/public-SDK/API/DB regression then
confirmed that both old and new question forms cause no write, while a real
fact still captures and injects across sessions without reaching another agent.
That regression used update rehearsal and **zero model calls**. Neither real
application was repeated on the corrected code. Source digests distinguish them.

## Completion source follow-up on 2026-09-14

The pinned stable host already retains more source information than NoldoMem's
default completion capture consumed. The source chain is
[`prepareChatSendUserTurn`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/server-methods/chat-send-user-turn.ts)
→ [`buildPersistedUserTurnMessage`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/sessions/user-turn-transcript.message.ts)
→ [`buildFromPrepared` / `resolveFinalCodexMirrorMessages`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/user-prompt-message.ts)
→ `agent_end`. The prepared row contains a source idempotency key, millisecond
observation time and ordered `__openclaw.media` facts. These do not identify
which sentence in an assistant answer came from which image.

NoldoMem now preserves the supplied key/time, and a single matching local media
reference when the captured text is already a recognized derivative. It does
not retrieve sessions, fetch files, consume the composite upstream prompt,
increase capture admission or save arbitrary model interpretations. Suppressed,
multiple, mismatched and remote-only media receive no inferred reference.
The agent/session binding and existing session-granularity replay tombstone
remain unchanged; a media path is not a new file-level forgetting guarantee.

The new synthetic adapter regression failed before the fix and passed afterward;
**33 focused tests** passed with Ruff. It covers identity/time retention,
attribution refusal, legacy absence and agent scope. This is source-contract and
adapter evidence, not a new native/model run. A model-free attempt to use the
host's source-tree `plugin-test-runtime` helper stopped with
`ERR_PACKAGE_PATH_NOT_EXPORTED` on the installed 2026.9.3 package. No private
import workaround or package installation followed. All earlier media outcomes
and model budgets above remain unchanged. Image interpretation persistence still
needs a design that preserves source and inference uncertainty.

The subsequent [source-linked response implementation](media-response-follow-up-2026-09-14.md)
adds a qualified generated-episode path and separate model-free native evidence.
It does not retroactively change either real model result above.

## Hermes preflight and bounded usage

The installed Hermes **0.21.1 / v2026.9.7** representative used Python **3.11.15**.
Its release directory had no Git metadata; the relevant transcription and profile
resolution files were byte-identical to pinned stable
[`2237be35`](https://github.com/NousResearch/hermes-agent/tree/2237be355906fbe6065ce1815711eee52b2d646e).
The receipt records those file digests without inventing a whole-install commit.
A fresh named profile used the supported global saved-auth fallback internally:
`openai-codex` / `codex_responses` was available without credential copying.
An earlier local preflight had no saved Codex access; that did not imply failure
of the installed runtime.

Native metadata confirmed an explicit **local** STT selection. In the isolated
installed runtime, neither `faster-whisper` nor a local Whisper command was
available, and native OpenAI audio access was unavailable. Another saved native
provider was detectable, but its STT route was not the selected path and was not
substituted. No lazy install, provider change or Hermes media/model request was
made. This is an isolated-path prerequisite failure, not a claim that production
text memory or all production voice handling is broken.

OpenClaw used **2/2 applications**, **33.71 + 31.54 = 65.25 seconds** wall time,
within 120 seconds per turn and 240 total. Its native driver receipts report
32.729 and 30.536 seconds separately. Native usage summed to **104,444 total
tokens**: 23,028 input, 1,160 output and 80,256 cache-read, with 570 reasoning
tokens reported separately. These are native counters, not billing proof.
Physical model requests/retries and audio billing were not independently measured.
One raw media batch was submitted; there was no external retry or restarted
application. Hermes used **0/2 applications and zero model seconds**. Unused
allowance is not a successful test or a newly reset budget.

## Remaining acceptance and cleanup

- **Independently fixed:** topic-prefixed question capture, temporary PDF plugin
  setup, and the associated focused/native regressions. Successful pixel
  interpretation still needs source-linked durable representation before the
  cross-session image goal is met. No blanket assistant-answer capture was added.
- **Access/verification limits:** OpenClaw audio failed at the provider boundary;
  the safe HTTP reason remains unknown. Hermes needs an available selected STT
  path before its approved three-file scenario can run. More model turns alone
  do not repair either prerequisite or the image representation gap.
- **Separate host dependencies:** Hermes #107369's clip-provenance CI/audit and
  OpenClaw #109370's settled per-attachment delivery design remain at their prior
  maintainer checkpoints. This run did not poll or duplicate those contributions.
- **Unproven/irrecoverable:** no real outbound file delivery, scanned-PDF OCR,
  universal channel coverage, or restoration of never-recorded legacy provenance.
  Incoming PDF text extraction does not close those requirements.

Owned temporary Gateway/control processes were absent and its port was closed
before cleanup. Both temporary remote profiles, the unused local profile and
owned helpers were removed without logout/revoke. Selected production service
PIDs and start times were unchanged. Public evidence contains only synthetic
content; no raw configuration, credentials or private operational paths were
exported. Full original acceptance remains incomplete. Final-head CI is recorded
on PR #34; quota-unavailable automatic review must not be called clean.
