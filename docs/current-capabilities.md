# Current capabilities and evidence

This guide consolidates the evidence recorded through **2026-09-14**. It applies
to the identified NoldoMem candidates and host versions, not an already published
release or a production deployment. Earlier failures remain in the linked reports;
later corrections do not retroactively turn those applications into successes.

## Existing integrations do not require unpublished host changes

The native tests used **Hermes v2026.9.7 / 0.21.1** (`2237be35`) and
**OpenClaw 2026.9.3** (`1391f7cd`, Node 24.19.0) with their supported loaders,
memory-provider or plugin surfaces. They did not require the proposed Hermes
audio metadata patch or an OpenClaw delivery patch. Separate agent scope, a
NoldoMem API, host capture/recall permissions and suitable native model/media
access remain prerequisites. Media decoding belongs to the host/provider;
NoldoMem consumes the resulting text or qualified model interpretation.

## Observed behavior

| Behavior | Evidence and limits |
| --- | --- |
| Natural correction and cross-session use | [Hermes native text applications](native-acceptance-2026-09-11.md) and [OpenClaw Gateway applications](openclaw-native-gateway-2026-09-13.md) linked a natural correction to an older record and answered current/previous questions in a fresh session without a recall command. The answers used captured dialogue as well as stored facts; this is not proof of every history-only query. |
| Automatic capture and injection | OpenClaw's text applications and [raw PNG applications](media-native-proof-2026-09-14.md) observed real model-triggered hooks, stored record IDs and the next session's actual injected context. Hermes's later media run observed first-turn capture and the fresh-session `api_content` memory sidecar. No prepared answer was injected by a separate model harness. |
| Incoming image content | OpenClaw's raw PNG entrance detail was automatically stored and used in a new session. [Hermes's native media run](media-audio-document-results-2026-09-14.md) also used its PNG detail. Image interpretation is generated evidence, not independently verified OCR or a confirmed user statement. |
| Incoming speech content | Hermes's actual local native WAV transcription supplied a detail later used in a fresh session. Stable Hermes did not preserve the exact successful clip origin. OpenClaw audio processing returned HTTP 429; successful OpenClaw WAV learning remains unproved, and the status alone does not establish its cause. |
| Incoming document content | Hermes's native PDF reader supplied text later used in a fresh session. OpenClaw extracted its PDF but initially missed capture. The NoldoMem correction passed real-event replay through the native lifecycle; a subsequent real application used that record. A new real first-turn capture on the corrected code was not run. |
| Uncertainty and isolation | The bounded model answers withheld the absent guide and excluded a conflicting other-agent record. OpenClaw withheld the missing audio detail. Some source wording and unnecessary tool use remained imperfect; these few examples are not general accuracy or latency estimates. |
| Forgetting and source replay | [Source-forgetting evidence](forgetting-sources.md) and the later native document regressions cover source-bound replay rejection, cache/index removal, isolated scope and explicit relearning. Source-session digests retain no deleted text or text hash. Legacy records without reliable source identity do not acquire replay protection retroactively. |

The latest media fixes recognize successful document derivatives, exclude failed
or empty file reads and remove an obsolete read instruction only after matching
successful extraction. Their post-fix checks used real host readers/loaders and
temporary APIs/DBs without another model application. The
[media results](media-audio-document-results-2026-09-14.md) distinguish those
checks from earlier real model behavior and report application time/native usage.
Physical provider requests and billing cost are unknown.

## Additional host metadata and unverified paths

| Boundary | Effect on current use |
| --- | --- |
| Optional Hermes audio metadata proposal | Would preserve successful transcript-to-clip attribution and processing outcome. It is a general host change, not a NoldoMem dependency. Transcript content can already be remembered without claiming its missing clip origin. |
| OpenClaw delivery metadata proposal | Would improve correlation between originating runs, logical deliveries, attempts and platform message parts. It is a general host proposal, not an implemented requirement for recall. Current evidence cannot establish complete per-attachment delivery across all channels. |
| Draft versus delivered content | Generated content and attachment presence can be captured with their evidence labels. Neither alone proves delivery. No real external message/attachment delivery was performed in these tests. |
| Raw-media and channel coverage | The fixtures were a PNG, WAV and text PDF. Scanned-PDF OCR, all channel dispatch paths and live URL expiry/re-fetch are not covered. Hermes media used native Gateway preprocessing followed by AIAgent, not a complete external-channel Gateway dispatch. |
| Historical source loss | New metadata cannot recover previously unrecorded source identity or link old disconnected graph records. No production migration or reconstruction was performed. |

These boundaries do not prevent use of the existing integrations. They also do
not count as completed acceptance: the full original goal still has unverified
media/delivery paths. There is no need to install an unpublished host patch to
review or use the currently supported NoldoMem paths.

## Choosing a memory arrangement

The [same-corpus architecture comparison](platform-memory-alignment-2026-09-09.md#architecture-comparison)
includes native-only, NoldoMem and supported coexistence. Native standing context
is a simpler option for a small curated preference set. NoldoMem provides scoped
episodic evidence and explicit validity history with retrieval overhead. A single
durable authority plus same-agent native session/procedural helpers is the
supported coexistence pattern; two unsynchronized durable writers are not.
The small lexical comparison did not establish a universal winner. No production
memory choice was changed.

The [Turkish helper transition](turkish-helper-migration.md) removes Zeyrek/NLTK
without replacing them with an optional extra. Capture/index/recall comparisons
preserved quality in the tested corpus. External Python morphology callers have
a documented breaking transition that requires coordinated major-version release
handling; preparing this contribution does not itself publish that release.
