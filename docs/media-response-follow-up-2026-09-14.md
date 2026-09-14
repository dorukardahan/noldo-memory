# Source-linked media response episodes

Later evidence: the [real follow-up and native-history correction](media-history-follow-up-2026-09-14.md)
found that the stable completion snapshot omits media which its persisted source
retains. The source-field assumptions below describe the earlier model-free
checkpoint; they did not prove that every real completion supplied those fields.

The [raw-media model run](raw-media-results-2026-09-13.md) correctly described
an image in its first answer but did not retain it. The change below adds a
bounded durable representation of **what the model said in a media-bearing
turn**. It does not claim that every sentence is an extracted image fact, or
that the earlier failed cross-session model test now passes.

## Mechanism and scope

The pinned OpenClaw stable source, commit
[`1391f7cd`](https://github.com/openclaw/openclaw/tree/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7),
provides the necessary admitted-input fields:

- [`buildRunUserTurnIdempotencyKey`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/sessions/user-turn-transcript.ts)
  binds a source input to `runId:user`.
- [`prepareChatSendUserTurn`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/server-methods/chat-send-user-turn.ts)
  retains ordered media facts and image placement slots on the prepared input.
- [`resolveFinalCodexMirrorMessages`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/user-prompt-message.ts)
  enriches the persisted transcript input. The later real run showed that this
  enrichment does not reach its `agent_end` snapshot. Native finalization does
  supply the current run context and terminal assistant status.

Within opted-in automatic capture, a local/WebChat completion can retain one
bounded response episode when the latest input matches the current run, has an
active local image slot, and ends with a successful terminal assistant message.
The record keeps uncertainty present in the captured answer excerpt (up to
1,500 UTF-16 units) and user request (up to 300), with surrogate-safe boundaries;
it uses `role=assistant`, `assertion=inferred`, `delivery=generated`,
`representation=text`. It is not a user assertion, OCR/caption output or a
channel-delivery receipt. A multi-attachment input is linked as one source turn;
individual answer claims are not assigned to a particular attachment.

Capture fails closed for missing source fields, old/history input, suppressed
media, non-user handoffs, unfinished/error replies, prompt-injection markers and
NoldoMem tool calls/results in the turn. A narrow presence check on the host's
submitted context also excludes answers supplied with a NoldoMem memory fence:
otherwise a recalled source could be relabeled as new input. The composite
prompt is never stored or mined for facts. Missing submitted-context evidence
also skips this writer. This conservative choice can miss a new image detail
when the same turn also recalled existing memory; it is not universal media
learning or general dependency tracking for arbitrary native context helpers.

The existing completion item bound and agent/session source binding apply.
Preprocessing remains the sole inbound writer in `preprocessed` mode; completion
can separately record the qualified generated episode. Known external-channel
responses remain owned by `message_sent`, avoiding a second automatic response
writer there. This does not add per-attachment channel delivery coverage. Generated
and delivered observations are not independent confirmations of a fact.

`/v1/store` no longer automatically promotes derived/inferred evidence to a
user rule merely because its text contains a directive. Explicit reported-user
rules retain the existing behavior. Automatic context reminds the model that
inferred/generated entries are not confirmed user facts or delivery proof.

## Evidence

The new source-linked episode assertion fails on baseline `763bf0ed` and passes
on the candidate. **89 focused tests** cover adapter admission, API persistence,
recall, exact replay deduplication, agent isolation, forgetting, independent
sources, explicit relearning and unchanged capture paths. Another 102 ingestion
tests and Ruff passed. An offline wheel built without dependency installation;
its API bytes match the candidate and its metadata contains neither NLTK nor
Zeyrek. The unavailable local `build` frontend was not installed; exact-head CI
also checks distribution builds. The negative
preprocessing assertion now verifies actual absence of duplicate inbound writes,
rather than requiring the entire completion hook to be absent.

The installed **OpenClaw 2026.9.3 / Node 24.19.0** loaded the candidate in a fresh
profile with a temporary API/DB, without authenticating a model provider. Its official update
rehearsal started the Gateway plugin lifecycle; public harness SDK completion and
prompt-build entry points then exercised capture and cross-session injection.
The captured ID appeared in the next prompt with inferred/generated labels and
the original uncertainty. Another agent received none of it. Repeated completion
created one row; forgetting blocked the replay and removed subsequent injection;
explicit API relearning allowed capture again.

This is **model-free native lifecycle/API evidence**: completion content and
media metadata were synthetic contract fixtures, not newly decoded pixels or a
real generated answer. No model turn, OAuth login, embedding, media provider or
channel send ran. [Receipt and source hashes](media-response-results-2026-09-14.json)
identify native-tested and final bytes separately. An earlier native check is retained separately;
the follow-up exercised explicit WebChat context after adding the external-channel
exclusion. A later surrogate-boundary fix passed 36 focused tests, including its
new Unicode regression; the unchanged native ASCII scenario was not repeated.
An initial
runtime-path preflight failed before creating a test root; the installed
interpreter path was then verified and corrected. Owned processes/root were
removed; selected production service identities remained unchanged.

The existing agent-local **source-session** tombstone applies. This does not
add file-level replay detection, recover legacy ownership, delete host history
or authorize another source automatically after forgetting.

## Original follow-up proposal (subsequently attempted)

At this checkpoint the technical condition had changed: there was a source-qualified durable
episode path with native model-free capture/injection/forgetting evidence.
The proposed model check was **two OpenClaw applications**, at most 120 seconds
each and 240 total, using the normal isolated Gateway startup and existing
OpenAI/Codex route. It needed a fresh isolated device-code login approved by the
user because earlier temporary OAuth state was removed. Existing credentials
must not be copied. The later linked report records its authorization, actual
attempts, failure, correction and the now-preserved isolated profile.

First attach only the already prepared synthetic entrance PNG and ask for the
practical visit detail, without supplying its contents. Then use a different
session to ask which entrance to use and who the guide is, without mentioning
media or recall. Expected evidence is a genuinely generated first response,
its source-qualified memory ID, automatic second-session injection, and a
grounded answer that preserves uncertainty and excludes the other agent's guide.
No outside retry or replacement turn is proposed. Report native usage rather
than equating two applications with two physical API calls. Cleanup only the
owned test root/processes, without logout/revoke.

This would test the changed image-response mechanism, not retry failed ASR or
prove PDF learning, Hermes media coverage, real outbound delivery or pending host
metadata contributions. Those original acceptance gaps remain separate.
