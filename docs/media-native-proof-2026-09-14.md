# Real image capture and cross-session answer after the correction

Candidate `581754a67c4e6cb5a4672f64f21bf4252ce77e90` passed the bounded
image learning and recall scenario on installed OpenClaw **2026.9.3**, Node
**24.19.0**, Codex plugin **2026.9.3**, `openai/gpt-5.6-sol` through the normal
temporary Gateway lifecycle. This adds real model-triggered evidence to the
[previous source correction and model-free regression](media-history-follow-up-2026-09-14.md).
It does not relabel the earlier failed applications as successes.

## Scenario and observed chain

The same isolated access was reused, with no login or credential transfer.
A fresh synthetic NoldoMem database and two new sessions excluded prior replay
records. Only another synthetic agent had a guide-name fixture. Search used
lexical/degraded mode without an embedding service. No expected image content
was supplied in a prompt, primary memory, or hand-written model context.

1. **Learn from raw input.** Attach the existing independently generated Aurora
   entrance PNG and ask: “Here is the Aurora visit sign. What practical arrival
   detail can you read?” Native input contained one image and no memory fence.
   The answer was “Use the **blue door** as the entrance.” The real `agent_end`
   path automatically stored record `a479b86ee99c4c6d`, tied to source event
   `77f704e3-cc93-402c-8b6a-69ff3cc83fbd:user` and the host's managed image
   reference. The representation is explicitly an unverified, generated
   assistant interpretation, not independent OCR or a confirmed user fact.
2. **Use it in another session.** With no attachment or recall command, ask:
   “I am arriving for the Aurora visit. Which entrance should I use, and who
   will guide me?” The native input had zero images and zero prior history
   messages. Automatic recall inserted the exact stored record and its evidence
   into the model's actual prompt. The model answered blue door and declined to
   name a guide. Neither the injected context nor the answer contained the other
   agent's guide name. No manual memory-tool call or synthetic hook replay was
   used in these applications.

The second mixed declaration/question was also stored as a user conversation;
it did not create a claimed guide or entrance fact. This is distinct from the
earlier question-only capture regression.

## What remains imperfect

The second answer said it could not reliably read the guide from the available
“image context” and asked for the sign again. Its factual abstention passed, but
that wording does not precisely explain that it received a stored interpretation
rather than a fresh image. It also appended an unrelated assistant-naming
question from the native empty-workspace bootstrap instructions. Neither issue
was hidden by changing the fixture or adding a model-specific prompt workaround.

This proves one raw PNG to generated interpretation to automatic storage to
cross-session injection to grounded entrance-answer chain. It does not establish
general media accuracy, independent verification of generated interpretations,
or perfect source explanations.

## Consumption and cleanup

Both applications exited successfully. Native durations were **10.360** and
**12.089 seconds**; the wrapper measured **24.648 seconds total**, within the
120-second per-application and 240-second total limits. Native usage reported
**10,942 + 11,115 = 22,057 tokens**, including cache reads. Physical provider
requests/retries are unknown; two applications are not two guaranteed API calls.
Zero-valued native cost fields are not evidence of no billing. The earlier
two-application allowance and its 10.900 seconds remain consumed separately.

The [synthetic receipt](media-native-proof-2026-09-14.json) links candidate source
hashes, image hash, run/session IDs, stored evidence, actual injected memory,
answers and native usage. It excludes full workspace prompts and operational
paths. Temporary Gateway/API processes stopped, the temporary port closed, and
selected production service PIDs/start times were unchanged. Isolated access
remains retained for related verification as requested; no logout/revoke occurred.

No product code changed for this run. Prior code tests, wheel/security checks
and model-free native regressions remain separate evidence and were not repeated.
Hermes's selected STT prerequisite, earlier OpenClaw audio failure, real-model PDF
learning, real outbound attachment delivery, pending host contributions and
unrecoverable legacy provenance remain open. This successful image scenario does
not complete the original multi-host/media acceptance contract.
