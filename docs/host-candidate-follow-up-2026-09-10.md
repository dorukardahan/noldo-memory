# Host candidate follow-up, 2026-09-10

Later evidence: [native acceptance on September 11](native-acceptance-2026-09-11.md).
The preflight/request-limit observations below are historical; the old unused
0/10 allowance was superseded by bounded application turns.

This continues the existing comparison and acceptance record. It does not replace the
September 9 research cutoff or treat a draft host change as a stable feature.
NoldoMem started clean at `c2a643df05479bb7636c6dd9e94f7bb7947d43a2` with green PR #34 CI.

## Host changes and upstream decisions

The official stable releases remain Hermes v2026.9.7
(`2237be355906fbe6065ce1815711eee52b2d646e`) and OpenClaw v2026.9.3
(`1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`). The installed OpenClaw package and
running Node executable were verified as 2026.9.3 and 24.19.0. The inspected
installed Hermes downstream identity was unchanged; the earlier selected-file
stable equivalence still applies. No production runtime was replaced.

Hermes [draft PR #107369](https://github.com/NousResearch/hermes-agent/pull/107369),
head `35bbc894ee6ab5378ee01b822844c613f4ecfd54`, is based on upstream main
`67764dc0863349a384c16425e73ee8571f3a94b7`. It preserves successful audio text,
clip/message origin, derivative kind, completion status and configured/fallback
route in structured message metadata. Prompt and echo text are unchanged;
confidence is unknown. Pending merges keep each clip's original message ID.
This follows the structured-metadata direction in #17762 by @river-morgan.
Open #100991 instead proposes a text-transforming `post_transcription` hook;
it does not replace pending/fallback/message-persistence evidence.

NoldoMem now consumes these optional records as separate derived audio memories,
while preserving typed captions and rejecting failed, empty or detached records.
It removes signed URL query/fragment components and does not fetch references.
On old stable, successful quoted transcript text remains recallable, but its
missing clip origin cannot be reconstructed. Neither change backfills old data.

OpenClaw main was pinned at `c55297caa246aedabd1fe349cc59ee3558f5e07e`. Its
`deliver-core.ts`, `message-sent-hook.ts` and `hook-message.types.ts` were identical
to the checked stable version. Existing issue [#109370](https://github.com/openclaw/openclaw/issues/109370)
and related [#125376](https://github.com/openclaw/openclaw/issues/125376) already
require a maintainer decision about delivery/attempt identity and SDK scope.
The [narrow proposal](https://github.com/openclaw/openclaw/issues/109370#issuecomment-5618872251)
requests settled per-part outcomes and platform IDs without raw URLs, paths or
receipt tokens, distinguishing failed/omitted/unattempted parts and queue-owned
replay identity. No duplicate issue, OpenClaw patch or draft PR was created past
that gate. Existing text receipts remain useful; universal per-attachment settled
observation is still missing.

## What ran

| Evidence | Result and boundary |
| --- | --- |
| Hermes candidate gateway preparation, pending merge, row persistence and neighboring queue tests | 31 passed through the canonical runner, file retries disabled; nine new cases fail on unchanged main with missing metadata. STT backend results are synthetic, not decoded audio. |
| Candidate Hermes loader/MemoryManager → temporary HTTP API/SQLite → fresh-session injection | Passed with derived audio label and original identity; other-agent export remained empty. Forgetting invalidated context and blocked replay of the same source. [Receipt](host-candidate-audio-results-2026-09-10.json) identifies the clean committed host and exact adapter bytes. |
| NoldoMem regression suite | 532 passed, one skipped; changed adapter tests include two distinct clips, typed caption, signed-reference stripping and invalid metadata. |
| NoldoMem lint/build/dependency audit | Ruff and sdist/wheel build passed; existing hash-locked package resolution reports no known vulnerabilities. No NLTK/Zeyrek dependency or optional extra was restored. |
| Hermes dependency audit | Not clean: unchanged dev pins httpcore2 2.7.0 and httpx2 2.7.0 report six PYSEC findings, 2026-3844 through 2026-3849. No exception or dependency-policy relaxation was added. |
| Hermes local PDF bytes | A newly generated text-layer PDF was extracted by real `tools.read_extract.extract_document_bytes` with firecrawl-anydoc 0.2.4, without model/network access. This is decoder evidence, not capture or answer evidence; image-only PDF OCR and raw audio were not tested. |
| OpenClaw local PDF bytes | Installed 2026.9.3 under its verified Node 24.19.0 extracted the synthetic booking sentence through native `extractFileContentFromBuffer` and the bundled document extractor. A separate textless vector page returned one image and no text derivative. No OCR/model/network fetch ran; the isolated temporary profile was removed. [Receipt](host-pdf-results-2026-09-10.json). |
| OpenClaw integration | Earlier real installed-stable loader/hook/capture/recall/injection evidence reused. No changed OpenClaw candidate or new full inference loop was tested. |

Hermes exact-head CI runs are `action_required` pending maintainer approval of
fork workflows ([CI run](https://github.com/NousResearch/hermes-agent/actions/runs/34480901728));
no green host-CI claim is made. The full host secret scan could not inspect all
tracked paths inside the credential protection profile. The changed-file,
tracked-only, filename-only heuristic scan completed without matches; that is
not a complete security audit. NoldoMem's previously established automated
review quota remains unavailable, not clean.

## Native inference preflight and remaining acceptance

The [September 11 native-runtime preflight](native-runtime-preflight-2026-09-11.md)
supersedes the access/backend interpretation below. The app-server observation is
specific to that optional backend; it does not describe the verified primary
Hermes routes or prove that deployed memory is broken. No third bridge is planned.

The new allowance is at most ten actual requests, at most five per host, including
all tool continuations, retries and ambiguous timeouts. **Zero new model requests
have been sent.** The earlier eight remain observations through the explicitly
bounded action adapter, not full native host inference proof.

The predeclared allocation is: (1) learn the synthetic booking and preference;
(2) naturally correct that preference and let the model select its predecessor;
(3) continue after its revision tool; (4) ask a combined current/past/booking
question in a fresh session; (5) continue after a history tool and abstain on an
unknown guide name. Any extra continuation consumes the same five-request host
cap, potentially leaving later observations unexecuted. No hidden title,
background-review, fallback or transport retry request may run outside the cap.

A concrete Hermes backend boundary prevents treating the available native Codex
CLI session as a ready replacement for the earlier action adapter:
[`build_turn_context`](https://github.com/NousResearch/hermes-agent/blob/67764dc0863349a384c16425e73ee8571f3a94b7/agent/turn_context.py)
collects provider prefetch but skips the API-content sidecar for `codex_app_server`.
[`run_codex_app_server_turn`](https://github.com/NousResearch/hermes-agent/blob/67764dc0863349a384c16425e73ee8571f3a94b7/agent/codex_runtime.py)
passes only `user_message` to `CodexAppServerSession.run_turn`, without the Hermes
memory tool schemas. A synthetic transport recorder invoked that real function:
a supplied message sidecar did not reach the transport. No subprocess or model
ran in this boundary test. The runtime's `api_calls=1` accounting is per native
turn, so it cannot itself enforce a five-wire-request budget across Codex tools.
This observation prevents substituting that test backend as native memory proof;
it does not make a bridge change necessary for other Hermes transports.

Normal native Codex CLI 0.153.4 reported an existing ChatGPT login. This does not
prove that a separately isolated Hermes/OpenClaw profile has usable credentials.
Hermes' OAuth status path can load/heal credential pools and resolve/refresh
legacy credentials; it was not used to mutate an existing store during this task.
OpenClaw has an official native-user-home Codex route, but no local OpenClaw
runtime was installed or existing package protection bypassed to combine it with
that login. A command-resolution probe on the installed host did not find Codex;
that limited probe is not proof that every configured binary path is absent.
Native access selection and a per-wire-request hard cap therefore remain
unverified, and the ten-request model test has not started.

The remaining boundaries are separate:

- NoldoMem-side available audio evidence loss is fixed for the host candidate;
  old stable still lacks that evidence, not the transcribed words themselves.
- Complete outgoing attachment evidence awaits the existing OpenClaw maintainer
  decision and eventual implementation; text-only delivery remains supported.
- Raw image/audio recognition and actual external attachment delivery have not
  been validated. Text-layer PDF decoding in both hosts and OpenClaw visual-page rendering do not close those gates.
- Full native learning/correction/current-history/answer acceptance remains open;
  existing retrieval/injection observations cannot substitute for it.
- Source-less legacy memories and unlinked old graph records cannot gain invented
  provenance. The documented prospective ingestion and deletion-marker migration
  remain the supported path; no production migration/reindex was performed.

A bounded next inference run still fits the unspent five-plus-five allowance only
after a supported native transport carries the memory context/tools and exposes
an enforceable actual-request cap. Raw media/channel acceptance would separately
need one synthetic image, audio clip and document plus one controlled attachment
send/partial-failure route; it is not authorized by that model allowance.

## Reproducing the installed OpenClaw PDF boundary

Run `scripts/check_openclaw_pdf.mjs HOST_PACKAGE_DIRECTORY` with the Node runtime
belonging to that installed 2026.9.3 package, an empty temporary `HOME`, and
`OPENCLAW_STATE_DIR` beneath that HOME. Do not point it at a production profile.
The script uses a supplied package, generates both PDFs independently, loads the
real installed extractor, and makes no package installation or model request.
It enables the bundled document extractor only in an in-memory test configuration.
The first page contains the booking sentence; the second contains only a colored
rectangle. The second result is rendered pixels, not recognition of their meaning.
This intentionally stops before NoldoMem capture and answer generation; the
previous preprocessed-hook injection tests remain separate evidence.
