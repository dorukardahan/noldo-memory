# Bounded model acceptance plan (2026-09-10)

Candidate: `7a1ce04b9e6972a11d20115b06262977a90d6dde` plus test-only stdin bridges.
Targets remain Hermes `2237be355906fbe6065ce1815711eee52b2d646e` and installed
OpenClaw `2026.9.3`, Node `24.19.0`. No production state is used.

The budget is eight inference requests, including any retry or continuation.
Four requests per host are reserved. No automatic model retry/fallback is allowed.
The existing native Codex CLI supplies GPT-5.5 inference using its saved login
internally. It receives only synthetic messages and the actual host memory
boundary outputs. A bounded JSON action transport dispatches model-selected
arguments through the native registered memory tools. This tests the real memory
integration plus an external model transport, not a complete gateway/channel or
Hermes agent inference loop. The harness must not insert expected memory text,
select a prior ID, or perform a correction before the model chooses it.

Synthetic learning events enter `MemoryManager.sync_all` or native `agent_end`:
quiet evening observatory visits; a violet dome with spiral stairs in a synthetic
vision derivative; an unrelated baking event; and a conflicting beta-agent fact.
Learning acknowledgments are fixture messages, not additional model generations.
The supplied vision description is not evidence of raw image extraction.
Unknown embedding inputs use a deliberate offline failure and the existing
lexical degraded path. No external embedding/reranker call is made. This is not
a fresh evaluation of semantic retrieval or calibrated admission floors.

| Request per host | Input and expected observation |
| --- | --- |
| 1 | New session: plan an Aurora observatory visit via the spiral stairs. Use quiet evenings and the violet dome from automatic context; describe caption content as derived when its source is relevant. Do not add the unrelated baking fact or beta's amber dome. An unknown guide name must remain unknown. |
| 2 | New session: “That is no longer my preference for Aurora observatory visits; I now prefer quiet mornings.” The model must choose a store call with the correct recalled old preference ID in `supersedes`. The harness executes only that chosen tool and then captures the natural turn. |
| 3 | New session: ask for the current preference and what preceded the change. Current injection must not revive the closed version. The model may choose a historical recall with `include_history`; reserve the fourth request for that tool result. |
| 4 | Return the unmodified native tool result, if selected. The answer must distinguish quiet mornings now from quiet evenings before the change, without inventing dates. If no tool is needed, use this request for beta-agent abstention instead. |

Record source IDs, validity/evidence, native context, model output, actual tool
arguments/results and request usage separately. Same-agent sessions use distinct
IDs and empty model histories except for the explicit tool-result continuation.
Check beta's automatic context independently without a model request. Report
individual observations, not a general accuracy percentage or speedup.

Execution notes before remaining requests: Hermes request 1 selected another
recall instead of answering because the guide name was absent. Record the extra
search; do not count this as an automatic-context-only answer. Its final
historical answer request also asks for the original dome/guide details. For
OpenClaw, request 1 includes a synthetic native audio envelope containing the
Friday workshop's 19:30 start; the unknown guide question is reserved for the
later combined question. This adds an audio-derived answer check without extra
model calls. Hermes's loss of audio provenance remains a separate host boundary.
