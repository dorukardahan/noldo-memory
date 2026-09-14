# Incoming media: bounded native results and derivative fixes

The [approved upstream plan](media-upstream-check-2026-09-14.md) was executed
with synthetic inputs, retained isolated access and separate temporary APIs/DBs.
No production profile, external recipient or embedding service was used.
Earlier text and PNG tests were not restarted. The [receipt](media-audio-document-results-2026-09-14.json)
separates actual model turns, recorded-event replay and post-fix regressions.

## Actual application results

| Host and native connection | Learning application | New-session application | Application wall time / native usage |
| --- | --- | --- | --- |
| Hermes v2026.9.7 / 0.21.1, `2237be35`, `openai-codex` / `codex_responses`, `gpt-5.6-sol` | Raw PNG interpreted as blue door; local native WAV transcription produced red notebook; native `read_file` extracted east arch from PDF. All three appeared in the answer and automatic capture. | No new attachments or transcript history. Actual `api_content` contained the three first-turn memory records. Answer gave all three details and no invented guide. It also unnecessarily read the PDF again and tried nonexistent synthetic files. | 2 applications, **80.0181 seconds**; native logical `api_calls` **8**; **33,188 tokens**. |
| OpenClaw 2026.9.3, `1391f7cd`, Node 24.19.0, native Codex harness / `openai-chatgpt-responses`, `openai/gpt-5.6-sol` | Native PDF extraction reached the model and its east-arch answer. Audio returned **HTTP 429**; no reliable normalized error code was observed. Completion capture missed the PDF. | After the correction and model-free replay described below, the real Gateway injected the derived PDF record into a fresh session. Answer used east arch and withheld unknown packing/guide details. It made one additional memory-tool recall. | 2 applications, **63.4602 seconds**; **94,223 native tokens**. |

Each application was limited to 120 seconds and each host to 240 seconds.
Physical provider-request counts are unknown; native retries/tool continuations
are not the same as application counts. Token totals are native observations,
not billed-cost estimates. Two observations per host do not establish a p95 or
speed improvement. No outside retry, alternate account/provider or audio fallback
was added after HTTP 429. That status alone does not identify quota policy or
prove that ordinary model access is unusable.

Hermes used the stable `GatewayRunner._prepare_inbound_message_text` and native
image builder, followed by actual `AIAgent.run_conversation`, its provider loader,
MemoryManager and memory-tool loop. This proves native preprocessing and agent
execution, **not full channel/Gateway dispatch or external delivery**. The
second-turn API sidecar, rather than the plain persisted user `content`, is the
source for the injection claim. It includes IDs `5e6fbfe5f91e4cee`,
`0278616159034bc6`, `4ce9885269a74bfd`; the other synthetic agent's guide is absent.
Raw files were not replaced by hand-written descriptions.

The official Hermes voice dependency closure was installed only in a temporary
interpreter, using the previously audited 27-package hash lock, binary wheels and
no dependency re-resolution. `pip check` passed for that voice environment.
The installed host packages were reused read-only. Native local
`faster-whisper 1.2.1`, model `base`, CPU/int8 processed the WAV with the host's
silence/confidence guards intact. Model download/cache stayed temporary. No
cloud STT account or service was added. This is a test prerequisite, not a change
to NoldoMem's Python dependencies or a production installation.

## Two observed NoldoMem defects

### OpenClaw drops a successful PDF derivative at completion

The actual `before_prompt_build` event contained the native successful `<file>`
envelope. Stable Codex's `agent_end` snapshot contained only the original request;
`chat.history` retained matching attachment metadata. Existing capture therefore
stored no PDF fact even though the model answered it. This is separate from the
HTTP 429 audio failure.

`plugin/src/hooks.js::registerAutoCapture` now retains only bounded document
chunks from the official prepared-input hook: at most 32,000 input characters,
the configured capture-item bound, 2,000 characters per derivative and 128 pending
run identities. Completion consumes the entry, including failures. The existing
local Gateway history lookup binds each chunk to the exact agent/session/run,
user idempotency key, timestamp, content and unique managed attachment reference.
The composite model/system prompt is never a capture source. A missing or
mismatched source, remote URL, malformed extraction envelope or denied conversation
access fails closed. Existing preprocessed-mode and already-extracted completion
capture do not gain a duplicate representation.

The stored record stays `derived`, `document`, `extracted_text`, with source
observation time rather than an invented event date. Native `hydrationSuppressed`
prevents subsequent binary hydration; it does not erase text already extracted
in the admitted turn. This distinction does not relax image-response capture's
suppression guard.

The **recorded real first turn** was replayed through the public harness SDK after
normal Gateway activation, with zero model calls. Capture, exact-reference
persistence, deduplication, automatic cross-session injection, agent isolation,
forgetting/replay rejection and explicit API relearning passed. A missing required
`developerInstructions` argument in the replay harness was diagnosed and fixed
before the successful replay; it was not a product/model failure.

The real second application then received record `506470c3fa444fdd`, derived from
source event `61fd9fea-f8b0-42e7-bece-7772aa5df8d8:user`, with zero incoming images
and zero history messages. **Post-fix real first-turn automatic capture was not
rerun**: the repair boundary is recorded-event replay followed by real native
injection/answer. The initial failed capture is not relabeled as success.

### Hermes preserves failed file reads and loses the reader's document flag

The first native `read_file` result contained `extracted_document: true`, line
text and a bound tool-call ID. The adapter previously stored the entire JSON as
ordinary tool text. The second application made 17 file-tool attempts in the
synthetic input area; failed reads were also captured, growing the total from
3 to 22 rows. Its recalled source notice still instructed the model to extract
the already-read PDF before answering.

`adapters/hermes/noldomem/__init__.py::sync_turn` now recognizes successful native
`read_file` results bound to that turn's actual tool call. It preserves extracted
text, document/derived classification, local reference and tool-call identity.
Failed reads without usable content are excluded. Only after a corresponding
successful extraction, the exact stable host's obsolete binary-document read
notice is removed from the user capture; missing-source or failed extraction
does not claim that document contents were learned. Ordinary tool text and other
user statements retain their existing behavior.

The installed native PDF reader and missing-file error, real provider loader,
MemoryManager and temporary HTTP/SQLite store passed a **model-free** regression
on the final adapter: source preservation, error exclusion, obsolete-notice
removal, duplicate prevention, cross-session injection, isolation and forgotten
source replay rejection. The two successful real model applications preceded this
adapter correction. No reduction in model tool calls or latency is claimed yet.

## Verification and unchanged acceptance limits

The full local suite produced 563 passes, one skip and one test-harness failure:
its single-callback map incorrectly overwrote the recall listener when capture
registered another `before_prompt_build` listener. The fixture now models the
host's multiple-listener contract; all assertions remain. The affected 24 tests
passed, followed by 36 media/alignment checks on the final duplicate guard.
The adapter/media set passed 113 tests with one skip. Ruff passed.
Source distribution and wheel were built through the installed setuptools PEP517
backend; metadata still excludes Zeyrek/NLTK. The `build` frontend was absent and
an offline isolated resolution lacked cached `wheel`; neither installed
production packages nor package-manager protections were changed to address it.
Final PR CI is recorded separately for the delivery commit. The filename-only
tracked-secret scanner reported 39 heuristic filenames (38 previously reviewed
and the new synthetic URL-rejection fixture); direct review found no new secret
material. No scanner rule, assertion or security exception was added.
Owned temporary roots and the named Hermes test profile were removed after
evidence collection without logout/revoke; all five observed production service
PIDs matched their pre-test values.

| Original acceptance area | Updated boundary |
| --- | --- |
| Implicit media recall | Hermes PNG/WAV/PDF content and fresh-session answer observed through native preprocessing/agent path. OpenClaw PDF answer/injection observed across a documented repair replay boundary; prior real PNG proof remains valid. |
| Grounding and trust | Both withhold the absent guide; OpenClaw withholds missing audio detail. Hermes stable still loses successful audio clip origin, and native image interpretation remains model-derived content. New metadata cannot recover old missing provenance. |
| Performance | Actual small-run duration/usage recorded. Redundant reads/recall were observed; failed-read capture was repaired without claiming an unmeasured speed gain. |
| Isolation/forgetting | Model answers exclude the foreign guide. Focused source-bound regressions preserve other-agent isolation, deletion and replay barriers. |
| Host-dependent work | Hermes #107369 remains an unpublished metadata contribution with maintainer CI/audit dependency. OpenClaw #109370 still needs the maintainer's delivery-metadata design decision. Neither is a NoldoMem release feature. |
| Unverified media/delivery | OpenClaw successful WAV transcription remains unproved after HTTP 429. Scanned-PDF OCR, universal channel coverage, real outgoing attachment delivery and unrecoverable legacy source links are not established by these inputs. No external messages were sent. |

There is no third host project, production migration, deployment, merge or release.
The native-only/NoldoMem/coexistence recommendation remains the earlier measured,
per-host comparison; this small input batch does not justify a new architecture.
Whole-goal completion is not claimed while the external and verification limits
above remain open. Repeating successful text/PNG tests or adding a new OCR service
would not resolve the outstanding delivery decision.
