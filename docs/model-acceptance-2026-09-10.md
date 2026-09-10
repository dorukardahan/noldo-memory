# Bounded model observations (2026-09-10)

Eight GPT-5.5 requests through native Codex CLI 0.153.4 completed: four per host.
The [predeclared plan](model-acceptance-plan-2026-09-10.md) and
[synthetic evidence](model-acceptance-results-2026-09-10.json) retain the source
records, native contexts, model-selected arguments, tool results and answers.
This is a small behavioral experiment, not an accuracy benchmark.

## Execution boundary

The product code tested by the model was PR #34 head
`7a1ce04b9e6972a11d20115b06262977a90d6dde`, with test-only stdin bridges.
Hermes used its real pinned provider loader and `MemoryManager`; OpenClaw used
the installed 2026.9.3 loader, hook runner and scoped registered tool factories
under Node 24.19.0. Every memory operation reached a temporary HTTP API and DB.
Model-generated tool arguments were dispatched unmodified through those tools.
No prior ID or revision was selected by the harness.

The inference transport was a bounded JSON action adapter over the existing
native CLI login. These are **native memory integrations plus real inference**,
not full Hermes agent/Gateway/channel inference loops. In particular, the
correction step ends at the successful selected tool, without an additional
model acknowledgment or a completed ordinary gateway turn capture. Learning
messages and acknowledgments were synthetic fixtures; no extra generation
was hidden in setup. Model histories were empty across sessions, except for
returning the selected historical tool result to the final request.

Native CLI HTTP and stream retries were both zero, with no model fallback or
native CLI tool iteration. One 90-second startup timeout emitted no thread or
turn event and did not submit a model turn. The pinned CLI
[prints its thread summary before submitting `turn/start`](https://github.com/openai/codex/blob/042fb41b7c813ac7999105e886b2b7aa715b5081/codex-rs/exec/src/lib.rs#L928).
Normal and explicitly allowlisted launch environments both reported an existing
ChatGPT login; no credential file was inspected or transferred. Metadata-only
startup probes were not inference requests. CLI elapsed times (34-42 seconds)
include process/startup overhead and are not host response-latency measurements.
The requested model ID was GPT-5.5; the server snapshot was not exposed in JSON.

No new embedding or reranker request was made. Missing prerecorded vectors or
the synthetic embedding outage exercised the lexical degraded path. This does
not re-evaluate semantic quality, calibrated admission floors or actual OCR/ASR.

## Observations

| Behavior | Hermes | OpenClaw |
| --- | --- | --- |
| Implicit cross-session context | Preference and synthetic vision derivative injected; model selected another recall when the guide name was absent | Preference, image derivative and audio derivative injected; first answer used evening, violet and 19:30 without an extra memory tool |
| Natural correction selection | Model selected the original preference ID in `supersedes`; native store closed that version | Same result through the real scoped tool factory |
| New-session current state | Automatic context contained the new morning preference, not the closed evening version | Same |
| Historical answer | Model selected `include_history`; final answer distinguished mornings now from evenings before | Same |
| Unknown/other-agent information | Guide name remained unknown; beta's amber dome did not enter alpha context or answers | Same; beta context was separately checked through the native hook |
| Derived provenance | Image source stayed `derived` in storage and context | Image/audio stayed `derived` in storage and context |
| Source disclosure | Final answer did not explicitly name the derivative; source disclosure on request was not tested | Same limitation; no raw-media-quality claim |
| Irrelevant history | Baking record incorrectly injected before the fix, but absent from the final answer | The hook did not capture this baking fixture at all; its absence is not evidence of superior retrieval |

The eight model requests selected five native memory tool operations: Hermes
performed an extra initial recall, a correction store and a history recall;
OpenClaw performed a correction store and a history recall. Retrieval success,
context admission and final answer behavior are separate observations. There
was no native-only or dual-authority model benchmark in these eight requests;
the earlier architecture comparison remains bounded by its own evidence.

## Measured defect and repair

The irrelevant baking result also reproduces against original remote-main
`b9616d0`: it is an existing search defect, not a September host regression.
FTS trigram OR matching can retrieve `flour` through `our`, or unrelated text
through `the`. The entity fallback for `The Aurora` could reintroduce the same
candidate, so filtering only the direct keyword lane was insufficient.

Degraded search now requires a content-bearing substring match for keyword
candidates and graph text fallbacks. Existing bilingual stopwords and a small
set of English question/pronoun words cannot admit a candidate by themselves.
Direct graph source links and explicit type lookup remain valid evidence.
Substring matching still supports Turkish inflections; this is not stemming or
lemmatization. Healthy semantic search, cache admission and scope are unchanged.

The real FTS regression failed before the repair. Search/intent/API/Turkish
acceptance tests passed afterward (73 tests). Native Hermes context then omitted
the baking record while retaining the current preference and image derivative;
OpenClaw's native current/media/scope context also passed on the repaired code.
The eight model requests were **not rerun** after this filter repair. The evidence
JSON records the repaired search source digest and both post-fix native contexts.

The initial full local run passed 489 tests and skipped one; 11 Node-dependent
tests could not start because the isolated test PATH omitted the installed Node.
Those exact 11 passed after correcting only the test environment. Lint, compile,
sdist and wheel build passed. The final source change then passed the 73 focused
checks above, plus a direct-graph/namespace regression (11 search tests passed). Exact-head CI is reported on the PR; no security exception was
added. Dependency resolution/removal evidence from September 9 remains applicable.

## Additional provider-boundary repair

A subsequent source check found a separate NoldoMem-side loss: Hermes automatic
formatting discarded `modality`, `representation`, `observed_at` and `confidence`
even when the API supplied them. A failing formatter regression reproduced it.
The adapter now carries these supplied fields using bounded enum labels and
finite, range-checked numeric values. Zero confidence is preserved; absent legacy
labels add no overhead, and the existing total character bound still applies.
No arbitrary reference/event string is newly promoted into prompt text.

The Hermes adapter suite passed (75 tests, one optional skip); six focused cases
also checked malformed metadata and legacy omission. The real pinned Hermes
loader/MemoryManager/temporary HTTP path retained `modality=image` and
`representation=extracted_text` in another session's context. No model request
was made for this follow-up, so the eight answer observations above still apply
to the earlier product head. This repairs available metadata propagation; it
does not recover the modality already lost by native voice preprocessing.

## Remaining product boundaries

| Boundary | Impact on the original goal | Narrow supported direction |
| --- | --- | --- |
| Raw image/audio/document extraction | Not validated by synthetic descriptions; NoldoMem does not decode bytes itself | Keep using existing host extractors. A separate bounded raw-media test would be needed; no new OCR/ASR service is justified here |
| Full outgoing attachment content and delivery | Generated files or paths do not prove what reached the user | The pinned [OpenClaw `message_sent` contract](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/hook-message.types.ts#L209) exposes delivered text/success but no attachment content list. Preserve confirmed text; complete attachment correlation needs a supported richer host event, including concurrent-turn identity |
| Hermes audio provenance | A successful transcript can look exactly like user-typed quoted text, so the adapter cannot safely infer audio origin | The pinned [voice enrichment](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/gateway/run_inbound.py#L1935) removes the distinguishing wrapper. Pass structured origin evidence through the host/provider boundary; do not label every quotation as audio |
| Replay after forgetting | The earlier model-run candidate reproduced reappearance on both hosts. A subsequent [source-session admission fix](forgetting-sources.md) blocks that replay and provides explicit relearning, verified without more model calls | Reliable source-session identity is required. Unidentified legacy sources and unlinked legacy graph rows remain unprotected; no content fingerprints are retained |
| Full host inference and delivery loops | Loader/hook/MemoryManager plus external inference do not prove every native client, extraction, retry or channel path | Keep that distinction explicit; do not convert these eight requests into a claim of complete two-host E2E coverage |

The recommendation remains one durable authority with agent-isolated storage.
NoldoMem plus supported native context/session helpers has useful evidence for
these episodic/versioned paths. Native memory remains a valid option for small
curated context. NoldoMem now has explicit source-session admission; this does
not establish parity with every native deletion or transcript behavior. Unsupported synchronized dual writers are not
recommended. The owned temporary profiles were removed; the same installed Gateway process remained ready. No production memory arrangement was changed, and the complete
original goal is not marked achieved while the required boundaries above remain.
