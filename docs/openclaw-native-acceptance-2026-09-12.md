# OpenClaw native acceptance report, 2026-09-12

Four additional application turns ran through the installed OpenClaw 2026.9.3 /
Node 24.19.0 **official `agent exec` CLI**, OpenAI `gpt-5.6-sol`, and the installed
Codex runtime over stdio. Trace timestamps place execution on 2026-09-11 UTC.
The [synthetic receipt](openclaw-native-results-2026-09-12.json) links each native
session, model/tool envelope, HTTP request, returned memory ID and answer.
This is a real host/model/tool loop, not the earlier external inference bridge.

The outcome is **partial acceptance**: the new-store schema defect was repaired;
implicit cross-session retrieval and unknown-guide abstention were observed.
Natural correction did not establish the intended current validity interval, and
answers added an unsupported calendar date/timezone. A native `ok` envelope is
therefore not treated as a passing behavioral scenario.

## Isolation, preflight and budget

Separately provisioned temporary agent-local access was supplied before this run.
No login, credential copy, cooldown recovery or access canary was performed by
the test. The same temporary auth owner, config and state were retained. Only
that config was changed through the official CLI; installed OpenAI/Codex paths
and permissions were preserved while adding the candidate plugin. Native status
reported OpenAI usable. `plugins inspect --runtime` confirmed the candidate file
was imported, all five memory tools were registered, and `agent_end`,
`before_prompt_build` and `message_sent` were registered with the required grants.
Metadata-only inspection was distinguished from this actual runtime import.

The candidate API used a fresh temporary SQLite store in **keyword-only degraded
mode**, without embedding/reranker calls or fabricated vectors. The primary
synthetic agent began empty. Only a different synthetic agent knew a guide named
`Kestrel`. The model had only memory tools; workspace bootstrap and memory flush
were disabled in the temporary profile. No production conversation or memory
context was supplied. Every invocation used a new native session; correction and
final recall therefore had to obtain prior facts from memory, not supplied history.

The earlier failed turn remains **1 turn / 5.906240866985172 seconds**. These four
additional turns used **174.5515283383429 seconds**, giving **5/5 turns** and
**180.45776920532808 seconds** in total. All exited before their 120-second outer
deadline; the 600-second host limit was not reached. There was no external retry
loop. One new learning attempt followed a diagnosed product change; the failed
baseline remains in the record. No further model turn was started after the cap.

## Planned scenarios and actual results

| Cumulative turn | Expected behavior | Observed behavior | Wall time |
| --- | --- | --- | --- |
| 2, original candidate | Learn Friday 19:30 booking and quiet/non-group preference. | Ten native tool calls, nine failures, zero primary records. New-store requests sent `supersedes=""`, then invented IDs. API rejected them. Final acknowledgement did not prove storage. | 62.71 s |
| 3, schema fix | Learn the same facts through repaired native tools. | Two successful stores, zero tool failures, two records. Optional revision/time fields were omitted on the HTTP wire. | 25.19 s |
| 4, new session | Relate “now prefer guided group visits” to the prior preference and make it current. | Model recalled and selected the correct old ID, but supplied a future `valid_from`. Two repair attempts hit existing 409 protections; model then stored an independent current statement. Temporal acceptance failed. | 57.27 s |
| 5, new session | Indirectly answer booking/current/previous preference; abstain on unknown guide. | Four model-selected recalls; correct Friday 19:30, guided group now and quiet/non-group before; no Kestrel disclosure. Answer also repeated a calendar date/timezone not supplied by the user. Full grounding did not pass. | 29.38 s |

No prompt asked the agent to “remember”, search memory or call a named tool.
The expected behaviors were recorded before execution. These few examples do not
establish a general success rate, p95, speedup or superiority over native memory.
The two-store corrected learning case took fewer calls than its failed baseline;
model variability and unequal failure work prevent interpreting that as a latency
benchmark. Native counts total 23 tool calls, with 11 failures across all four
turns, including the failed baseline and temporal repair attempts.

Native token counters total input **32,309**, output **2,969**, cache-read
**51,200**, cache-write **0**, and reasoning **1,177** (reported separately, not
added again). Native total is **86,478**. Physical request count is unknown;
application turns, assistant turns and tool calls are not physical API calls.
Reported native cost is zero, but billing was not independently verified and the
test is not described as free. Earlier coordination calls are not included.

## Product correction and model-free regression

Candidate base was `92de23f3c9251ea8d56ab7131f138af52f693747`. The fixed
`plugin/src/tools.js` SHA-256 is
`1dc53b95a741c75473756974c37c290c164bcb7b4fac4394a46885ae309215a0`.
The original tool schema admitted omission but not null for optional arguments.
The observed native request instead supplied an empty revision ID. Optional
properties now admit null; the client omits null revision/time fields and retains
the existing API defaults. Required content/IDs remain non-nullable. Empty or
invented IDs still reach API rejection, explicit timestamps remain explicit, and
false/zero recall values are preserved. Descriptions distinguish validity from
the date/time of the event being remembered. No storage schema or dependency
changed, and no API guard was weakened.

Focused Python/Node regressions passed **27 tests**, and Ruff passed. The new
[`check_openclaw_optional_tools.mjs`](../scripts/check_openclaw_optional_tools.mjs)
also passed through the installed native loader, native schema projection and
validation, actual registered tools, HTTP API and temporary DB. It checks null
new-store/default behavior, a correctly timed revision, current/history recall,
rejection of an empty ID and another agent's revision, and receipt-based relearning
with a null alternative identifier. No model is called by that regression.
This scripted correctly timed revision does **not** replace the failed natural
correction scenario above. A transport interruption prevented one final-script
invocation from starting; no model scenario was restarted because of it. The final
model-free script was subsequently executed and passed before cleanup.

## Retrieval, injection and temporal evidence are separate

The corrected learning turn created booking `44adbf24403c4707` and preference
`a8e7b7ece1f1421d`. The natural correction selected that preference but supplied
`1799659270` as its validity boundary, far later than the trace's execution time.
The resulting revision `e29c021410354a33` was legitimately hidden from ordinary
current searches. The API rejected attempts to replace that future revision with
an earlier interval and to supersede its already-superseded parent again.
Those protections were retained; no test harness revision repaired the family.

The model's independent current record `acc5d3673e2f43f6` then explained the newer
preference to the final answer. The first final-session recall returned that
record, the original preference and booking; it did not return the future
revision. Thus the final answer's current/previous distinction is observed, but
does not prove that the authoritative revision family's current state is correct.
The model also expanded “Friday” into a specific date and timezone. Those additions
are unsupported by the synthetic user's input and are not counted as established
booking facts, even if the runtime supplied a current date/timezone.

HTTP traces correlate model-selected recalls with returned records and subsequent
native answers. **Automatic hook capture/prefetch was not observed in these CLI
turns**, despite the preflight registrations and permissions. All observed writes
were explicit native tool invocations. Existing model-free hook tests remain
valid at their narrower scope. No plugin/core monkey-patch or new Gateway was
introduced to turn this absence into an apparent pass, and it is not evidence
that production Gateway hooks or Hermes memory are broken.

### Model-free follow-up: registry activation

A fresh, credential-free fixture on the same installed host compared two native
loading paths with the same candidate and explicit hook grants. Native
`resolvePluginTools` returned all five tools while `getGlobalHookRunner()` remained
null. Calling the normal `loadOpenClawPlugins` with activation enabled then made
both `before_prompt_build` and `agent_end` available. The
[result](openclaw-hook-activation-2026-09-12.json) and
[reproducer](../scripts/check_openclaw_hook_activation.mjs) record this comparison.
No hook callback, HTTP request or model was invoked; the fixture was removed.
This adds no model-budget consumption and is not a new full native acceptance pass.

At stable `1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`,
[`resolvePluginToolLoadState` in `src/plugins/tools.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/tools.ts)
builds tool discovery with `activate: false`.
[`activatePluginRegistry` in `loader-shared.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/loader-shared.ts)
initializes the global hook runner. Harness prompt/agent-end helpers consult that
runner. NoldoMem already declares `activation.onStartup: true`;
[`shouldConsiderForGatewayStartup`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/gateway-startup-plugin-config.ts)
uses that flag for Gateway startup selection, not as a request to activate every
standalone tool-discovery registry.

This demonstrates a host lifecycle boundary consistent with the missing automatic
requests in the CLI trace, rather than a missing NoldoMem enable flag or an
inability to register its hooks. It does not prove the complete CLI invocation's
registry state at every phase, or that all official entry points lack activation.
No redundant manifest field, manual global-registry initialization in NoldoMem,
core patch or additional host contribution was introduced. Full automatic native
dispatch remains open; the successful activated-loader control is not substituted
for it. The previous model budget remains exhausted at five application turns.

## Original acceptance mapping and remaining work

| Criterion | Effect of this evidence |
| --- | --- |
| Verified platform/research baseline | Reuses the pinned research; confirms the installed native CLI/Codex route. No latest-version chase. |
| Fair architecture comparison | Earlier controlled native/NoldoMem comparisons stand; this isolated NoldoMem run is not an equivalent native-only or dual-writer model benchmark. |
| Implicit multimodal/temporal recall | Native tool-driven cross-session recall observed. Natural revision timing, fully grounded answers and native automatic hook use remain open. No new raw-media claim. |
| Quality-preserving performance | Removes the observed optional-field failure; records actual timing/tokens/calls. No general speed claim. |
| Agent isolation/public privacy | Synthetic other-agent guide stayed out of answers; model-free cross-agent revision was rejected. No shared memory pool or private data in this receipt. |
| Reviewable delivery | Scoped code, tests and docs remain on PR #34. Exact-head CI/review status is recorded on the PR; quota-unavailable review is not clean review. |

A model-generated Unix value cannot by itself prove the user's intended effective
time. Descriptions/null support did not prevent this run's erroneous explicit
timestamp. Silently replacing it with the server clock would also corrupt valid
scheduled/backdated requests, so that workaround was not applied. The next temporal
design needs a reliable distinction between a current correction and an explicitly
dated change, preserving the existing scheduled/backdated API. A further model
validation requires a new bounded allowance; none is assumed here. Removing
explicit temporal control or adding a new host bridge is not smuggled into this fix.

Hermes's prior successful native turns and subsequent model-free capture-echo fix
are reused. Hermes audio provenance draft #107369 still depends on maintainer CI
approval and its previously reported dependency findings; OpenClaw attachment
receipt proposal #109370 still awaits its design decision. They were not polled
again. Raw audio/image extraction, image-only PDF OCR and actual outbound attachment
delivery remain unexecuted; earlier free local derivative/text-layer evidence does
not prove them. Source-less legacy memories and unlinked historical graph records
cannot gain lost provenance retroactively. Full original acceptance remains open.

Only owned test processes and the temporary root were removed. No logout/revoke
was invoked; local removal does not establish provider-side grant expiry. The
installed Codex plugin was preserved. Relevant production process IDs were unchanged;
no production config/auth/DB write, restart, deployment, merge or release occurred.
