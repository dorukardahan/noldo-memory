# OpenClaw lifecycle and temporal contract follow-up, 2026-09-13

The installed **OpenClaw 2026.9.3 / Node 24.19.0** now has a passing model-free
Gateway-startup/capture/injection check. Both adapters have clearer source/time
tool instructions. The previous real model failures remain failures; no model,
OAuth, embedding, media extraction or external delivery request ran in this follow-up.
The earlier allowance remains exhausted at five application turns / 180.46 seconds.

## Actual lifecycle boundary

All host source references below are pinned to stable
[`1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`](https://github.com/openclaw/openclaw/tree/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7).

| Stage | Source and behavior |
| --- | --- |
| Earlier native entry | [`agentExecCommand`, `src/commands/agent-exec.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/commands/agent-exec.ts) invokes `agentCommand` with `oneShotCliRun: true`. It does not start a Gateway. |
| Agent registry | [`agentCommandFromIngress`, `src/agents/agent-command.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/agents/agent-command.ts) uses `withAgentPluginRegistry` when no prepared generation owns the run. [`loadAgentRuntimePluginRegistryHandle`, `runtime-plugins.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/agents/runtime-plugins.ts) loads a discovery registry with `activate: false`; `withPluginRuntimeRegistryScope` binds that handle, without initializing the global runner. |
| Tools | [`resolvePluginToolLoadState`, `src/plugins/tools.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/tools.ts) also selects `activate: false`. Tool visibility therefore does not establish hook activation. |
| Gateway owner | [`startGatewayServer`, `src/gateway/server.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/server.ts) enters normal startup/bootstrap and `loadGatewayPlugins`. [`activatePluginRegistry`, `loader-shared.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/loader-shared.ts) initializes the global hook runner. |
| Before model | [`prepareCodexAttemptPrompt`, `run-attempt-prompt.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/run-attempt-prompt.ts) uses `resolveAgentHarnessBeforePromptBuildResult`. The helper consults the global runner and incorporates returned context in the prompt. |
| After model | [`runCodexAgentEndHook`, `run-attempt-lifecycle.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/run-attempt-lifecycle.ts) dispatches agent-end side effects, which call the public harness lifecycle helper and global runner. |

Thus `activate: false` is real standalone host code, not solely a flag invented by
the test. Choosing `agent exec` instead of a Gateway-owned turn was the test's
entry-point choice. The earlier preflight's private registry was not proof of
activation in the later native CLI process. These sources and the separate
[registry comparison](openclaw-hook-activation-2026-09-12.json) explain that boundary;
they do not prove every possible CLI/backend generation lacks hooks. The supported
Gateway path suffices for the next test; no plugin-side global initializer, core
patch, new bridge, service installation or third host contribution is needed.

## Model-free behavior, not just registration

[`check_openclaw_gateway_lifecycle.py`](../scripts/check_openclaw_gateway_lifecycle.py)
creates a fresh temporary API/SQLite DB with embeddings unavailable and uses native
`config patch --stdin` only in the temporary profile. A temporary loopback Gateway
starts through `startGatewayServer`, using the host's `updateCanary` rehearsal
option. This preserves startup plugin activation while suppressing autonomous
scheduled work; it is not a normal channel/model run.

The [JavaScript probe](../scripts/check_openclaw_gateway_lifecycle.mjs) then submits
synthetic events through the **public `agent-harness-runtime` SDK**:
`awaitAgentHarnessAgentEndHook` and `resolveAgentHarnessBeforePromptBuildResult`.
It does not directly invoke a NoldoMem callback, construct a private hook runner,
or manually call store/search. It does manually supply completion/prompt events:
this remains a model-free lifecycle test, not proof of dispatch from a real model
completion or an incoming channel message.

The [receipt](openclaw-gateway-lifecycle-2026-09-13.json) links:

- One `agent_end` event to `/v1/store`, source `plugin-auto-capture`, and memory
  `92af12dbebe5484a`, scoped to synthetic agent `alpha` and its learning session.
- Another session's ordinary planning prompt to automatic `/v1/recall` and the
  exact captured text/ID in an untrusted-memory prompt section.
- The same prompt for `beta` to a separate recall with no injected event or stored
  row. No shared memory pool was introduced.

Two harness failures were diagnosed rather than counted as success: the initial
bundle resolver missed unaliased exports; a subsequent assertion-passing run
closed SQLite from the wrong thread during cleanup. The final probe fixed those
harness issues and passed with cleanup. No model requests were involved. The owned
Gateway/API/profile were removed and selected production process IDs were unchanged.

## Time and source: what the model evidence actually says

The [previous synthetic trace](openclaw-native-results-2026-09-12.json) first adds a
calendar date and `Europe/Istanbul` in the model's successful `/v1/store` request
for the booking, before correction or final recall. NoldoMem persisted that text;
recall did not invent those fields. The later answer repeated the ungrounded stored
claim. The source label `user` was itself supplied by the model and is not proof
that every detail was said by the user. Existing inference cannot be repaired by
relabelling those old rows or silently editing their text.

Before the first correction, recalled rows had null validity boundaries. The model
then explicitly sent `valid_from=1799659270` (2027-01-11T09:21:10Z) with the correct
old preference ID. No earlier returned validity boundary supplied that value.
The trace establishes the model-request origin, not the internal reasoning or
clock calculation that produced it. The server did not turn an omitted date into
that future date. Existing interval/lineage protections rejected subsequent repair
attempts; they remain unchanged.

The exact provider request containing the old tool catalogue was not retained.
Instead, the pinned tool-file digest, actual tool/API requests and an installed
host [schema reconstruction](openclaw-tool-contract-2026-09-13.json) establish the
available contract. The reconstruction uses the Codex catalogue's
`normalizeOpenAIStrictCompatSchema` then `projectRuntimeToolInputSchema` sequence
and native validation. It is not retroactively labelled a captured wire request.

The old OpenClaw wording said unspecified validity means now, but did not explicitly
forbid calculating a timestamp for an undated correction. Hermes only said
“Validity start as Unix seconds; omitted means now.” The old content description
also lacked a source-only date/timezone rule. The minimal shared wording now says:

- No explicit effective date: omit/null the field, do not calculate or invent Unix
  time, use the server default. Hermes now explicitly accepts null in this schema.
- Explicit past/future effective dates remain valid. Event time, ingestion time
  and revision validity are distinct; no future-date ban or silent clock override.
- Store only source-supported event details. Do not fill missing calendar dates or
  timezones from the current clock, locale or model inference.

Both native projected schemas still accept null, zero and a future timestamp.
Adapter parity/dispatch tests preserve supplied values and exact source text;
existing API current/history/future/lineage tests pass. These checks verify the
contract and its compatibility, **not improved model compliance**. The old temporal
and answer-grounding failures remain open. No fixture answer was relaxed or sent
to a model, and no model-specific prompt workaround was added.

Focused local suites passed **164 tests, 1 skipped** (alignment behavior/API,
Hermes adapter and plugin packaging). Ruff and sdist/wheel builds passed; built
wheel metadata still contains no NLTK/Zeyrek dependency. No package/dependency
changed. The filename-only secret scan returned the same 36 heuristic filenames;
it is not a secret-free certificate. Final-head CI/security status is recorded
on PR #34, and quota-unavailable automated review is not called clean.

## Next model test: proposed only

The changed technical conditions are Gateway-owned activation and the clarified
shared tool contract. Do not repeat the standalone `agent exec` test.

1. With separate approval, create a new temporary state/profile and synthetic agent,
   keeping installed OpenAI/Codex plugins read-only. The deleted OAuth root is not
   recoverable test access. In this same isolated owner, the official command is
   `openclaw models auth login --provider openai --method device-code --agent synthetic-memory --profile-id openai:noldomem-acceptance`.
   The user must approve the displayed device code with the existing account.
   No credential copy or new provider/account is proposed. Native status must
   confirm usable access before any model turn. Local-store separation does not
   guarantee zero provider-side session effects; cleanup must not revoke/sign out
   the existing account.
2. Start the official **temporary Gateway**, with the candidate and capture/recall
   grants. Use Gateway-owned `agent` RPC / the Gateway-first `openclaw agent`
   command, not `agent exec`, and no external message destination. Keep unrelated
   jobs/channels off in this fresh profile. Seed only synthetic prior booking and
   preference via the already-proven model-free lifecycle path, not hand-written
   model context; this does not claim another model-learning test.
3. Propose **two application turns**, each at most **120 seconds**, at most
   **240 seconds** total model wall time. First: a natural undated preference
   correction, with native prefetch/capture and model-chosen revision. Second:
   another session's indirect current/previous question plus an unknown guide.
   Require the revision family's current validity, not merely a duplicate current
   statement; require no invented date/timezone or other-agent detail. Observe
   hook HTTP calls, actual injected prompt, selected memory IDs, and answer
   separately. Native continuations belong to the turn; no external retry loop.

This is a plan, not new OAuth/model authorization. Failure or timeout must be
retained; stop at either deadline, and do not claim that process termination
proves provider cancellation. Physical requests/tokens are reported only if native
metadata exposes them. Remove only the new owned test area/processes, without
logout/revoke. No production config/auth/DB/service changes are proposed.

Hermes #107369 maintainer CI/audit and OpenClaw #109370's settled-delivery design
remain separate external gates; a single current comment check found no new
maintainer direction. Earlier Hermes model evidence is reused. Raw image/audio,
image-only PDF OCR, real outbound attachment delivery and irrecoverable legacy
provenance remain unchanged limitations. Full original acceptance is not complete.
