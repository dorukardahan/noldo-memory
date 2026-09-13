# Media-path follow-up, 2026-09-13

This follows PR #34 at `a0097d8045bdbfffc773c08a74b8fb308c8f3a80`.
The [two successful native OpenClaw turns](openclaw-native-gateway-2026-09-13.md),
previous Hermes acceptance, and the later question-capture regression remain
separate evidence. None was restarted. This follow-up made **zero model,
embedding, paid extraction or external channel calls**.

## Independently fixed in NoldoMem

- Default OpenClaw `agent_end` capture now uses the same successful-file parser
  as `message:preprocessed`. Previously only the optional preprocessing writer
  recognized document derivatives; default capture could label them reported text.
- The real stable file renderer inserts a blank line before its matching untrusted
  envelope. The existing parser rejected that content. Both writers now accept
  surrounding whitespace, normalize the document placeholder and keep only actual
  extracted content. Failure/path-only/image-render notices do not become facts.
  Matching envelope IDs and prompt-injection screening remain required; derived
  content is still injected inside the untrusted memory boundary.
- In both adapters, binary attachment presence beside text no longer claims
  `extracted_text`. Text remains usable with a `text` representation and conservative
  derived trust. Recognized native derivatives remain derived, not user assertions.
- Hermes recognizes the stable Cloud adapter's `[Content of …]:` text-file wrapper
  as a document derivative. No new decoder, dependency, queue or storage schema.

The new regression cases failed against the starting candidate. The installed
OpenClaw check then exposed the real blank-line mismatch that the earlier
handcrafted-envelope fixture had missed. A diagnostic run confirmed an empty
store despite successful native extraction. Those failed runs are not acceptance
successes. After the parser correction, the same native chain passed.

## Versions and evidence chain

Pinned Hermes **v2026.9.7**, commit `2237be355906fbe6065ce1815711eee52b2d646e`,
was executed from the verified stable checkout, using Python 3.13.7.
Pinned/installed OpenClaw **2026.9.3**, commit
`1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`, was tested using its actual Node
24.19.0 runtime and Python 3.12.3 for the separate candidate API. These are the
already selected targets, not a new claim about latest releases or moving main.
Source digests and bounded results are in [the evidence record](media-path-results-2026-09-13.json).

**Hermes:** `check_hermes_stable.py --media-text-only` writes an independent UTF-8
file containing “The Aurora observatory roof opens at sunrise.” It executes
[`WhatsAppCloudAdapter._inject_document_text`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/gateway/platforms/whatsapp_cloud.py),
the native document-note formatter and
[`_stage_turn_user_message`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/agent/turn_context.py),
then native MemoryManager → candidate provider → temporary HTTP API/DB → a
second session's prefetch. Assertions check content, derived document evidence,
source event ID, adjacent-image text classification and empty other-agent export.
This tests the Cloud file reader and common row/manager path; it does not claim
that every channel, the deployed bridge, a model or a real file delivery ran.

**OpenClaw:** `check_openclaw_gateway_lifecycle.py HOST NODE --media-file` starts
an isolated Gateway through the official update-rehearsal lifecycle, with normal
plugin activation and autonomous sidecars suppressed. Native
[`applyMediaUnderstanding`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/media-understanding/apply.ts)
reads the local UTF-8 file and calls the real
[`renderFileAttachmentOutcome`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/media-understanding/file-attachment-outcomes.ts).
Its actual body enters a synthetic completion through the public harness SDK,
which dispatches the Gateway-activated `agent_end` hook. The API trace contains
one store and two automatic recalls, scoped to the original and other agent.
The first new-session prompt contains the captured ID, text and derived document
labels; the other contains no memory. No direct store/search, private plugin
callback, model answer or live channel is substituted for that chain.

The native utility-preprocessed route is deliberately distinguished from native
harness-owned interpretation. In stable
[`getReplyFromConfig`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/auto-reply/reply/get-reply.ts),
locked model selection lets the harness own image/video/file interpretation and
skips general media/link understanding; explicitly configured audio understanding
can still run. The new test proves a supported processing route, not that every
Codex turn receives that derivative. It does not expand the prior two model turns
into raw-media acceptance.

## Content and source coverage

“Unknown” below means the required observation is absent, not that the host has
no capability. Existing detailed source and projection tests are reused from
[the host boundary report](host-evidence-boundaries-2026-09-10.md).

| Direction / input | Hermes stable | OpenClaw stable | Remaining user-visible limit |
| --- | --- | --- | --- |
| Incoming typed description beside image/audio/file | Text retained; now `text`, conservatively derived | Same default-capture distinction; preprocessing already conservative | A caption cannot prove the contents of adjacent bytes. |
| Incoming image | Text-mode native vision description is recognizable derived image text; native-vision route may supply pixels only | Recognizable native image descriptions remain derived; harness may own raw pixels | No new raw-image decoding/answer test. Without reusable text, an image-only event is not guaranteed cross-session content memory. |
| Incoming audio | Successful quoted transcript remains usable; stable row loses clip provenance | Existing preprocessing transcript/source metadata supported; default recognizes native transcript sections | Raw ASR quality untested here. Hermes cannot reliably distinguish a successful plain quotation from speech or recover its clip. |
| Incoming text document | Actual Cloud reader → native manager → content/evidence recall passed | Actual native file processing → SDK hooks → content/evidence injection passed | Channel-specific enrichment and configured extraction limits still apply. Neither adapter reads all attachments itself. |
| Incoming PDF / scanned file | Previous local native PDF extraction evidence remains decoder-level | Previous native text-PDF extraction passed; textless PDF produced image fallback | Image fallback is not OCR. This follow-up's joined native chain uses UTF-8, not a PDF-to-model answer. |
| Incoming link | URL/user description and supplied external-tool text may be captured; no automatic fetch by NoldoMem | Utility link understanding appends text before the preprocessing hook; capture keeps it derived | A URL is not page content. The prepared composite does not guarantee per-link source attribution or capture from harness-owned browsing. |
| Outgoing assistant text / image caption / spoken text | Assistant text is generated; external tool text is derived | Successful `message_sent` text is delivery-confirmed; available spoken-text projection is reusable | Generated text is not proof that an associated image/audio/file reached the recipient. |
| Outgoing image/audio/document bytes or link destination | Binary-only output is not decoded; generated paths/links do not prove content or delivery | Pre-send media list and transcript mirror are insufficient settled per-item evidence | Full per-file delivery/content recall is not proven. No outside message or attachment was sent. |
| Old absent origin / unlinked graph row | Cannot reconstruct fields never recorded | Same | New metadata applies prospectively. Use only attributable reimport/relearning paths; do not invent old source identities. |

For links, stable `applyLinkUnderstanding` and `formatLinkUnderstandingBody`
append provider text, while the actual `message:preprocessed` projection carries
prepared body but not a structured per-link result list. This explains the
conservative composite trust, rather than proposing an unapproved host project.
For outgoing attachments, the stable emitter's lossy projection and colliding
basename mirror were already behavior-tested; they were not re-tested here.

## External dependencies and acceptance status

Checked once during this follow-up: Hermes [draft #107369](https://github.com/NousResearch/hermes-agent/pull/107369)
remains at `35bbc894ee6ab5378ee01b822844c613f4ecfd54`, without new maintainer response
or CI result. Its successful-clip metadata is a candidate, not a stable feature.
OpenClaw [#109370](https://github.com/openclaw/openclaw/issues/109370) still needs the
maintainer's bounded settled-receipt design decision. No duplicate contribution,
new comment, third host project, deployment or production setting change was made.

These independent NoldoMem defects are fixed. They do not complete the original
multimodal acceptance: raw image/audio interpretation into a later correct answer,
real attachment delivery, complete stable Hermes audio provenance, and unavailable
historical source data remain distinct limits. A repeated text-only model test
would not resolve them. No further OAuth/model authorization is requested merely
to repeat the already passed correction/recall scenario. Maintainer decisions and
an explicitly scoped raw-media/channel test would be needed before claiming those
remaining behaviors; no new infrastructure or extraction provider is assumed.
