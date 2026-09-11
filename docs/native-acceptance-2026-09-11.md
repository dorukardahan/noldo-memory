# Native acceptance, 2026-09-11

Three real Hermes application turns completed through `AIAgent.run_conversation`,
native `openai-codex` / `codex_responses`, NoldoMem tools, and a temporary HTTP API
with its own SQLite database. The model was `gpt-5.6-sol`; candidate code was
`ac875a79bf490e15bff9d3ba0fda64513cbf02a1`. Installed Hermes was 0.21.1,
v2026.9.7, with the stable transport/source identity established in the
[preflight](native-runtime-preflight-2026-09-11.md). This was the stable host,
not the audio-metadata draft or an external inference bridge.

## Isolation and limits

The earlier **0/10 physical-request allowance remains a historical unused record**.
Its physical-count/retry-off preconditions were replaced before this run by at
most five application turns per host, 120 seconds per turn and 600 seconds total
host model-test wall time. Native continuations/retries were allowed; no external
retry loop or failed-scenario restart was used. Both native `run_budget_seconds`
and an outer process deadline bounded each turn. All three exited normally.

A fresh official named Hermes profile reused saved access only through native
resolution. No credential was read, copied or supplied by the test harness.
The profile was not cloned or selected as the active profile. Native standing
memory, user-profile injection, workspace context loading, background review,
title generation and compression were disabled in the temporary profile.
Only the candidate memory tools were exposed. The candidate endpoint and agent
scope were asserted before inference; that agent initially had no records.
Another synthetic agent alone knew the guide name `Kestrel`.

The API deliberately ran in **degraded lexical-only mode**, with no embedding or
reranker service call and no fabricated vectors. This evaluates native behavior
under that mode, not production semantic quality or latency.

## Planned behavior and observations

| Turn | Input and expected behavior | Observed native path | Wall time |
| --- | --- | --- | --- |
| 1, session A | Learn Friday 19:30 booking and quiet/non-group preference. | Two model-selected stores; answer acknowledged both. | 21.44 s |
| 2, session A | Natural correction to guided group visits; link the old preference. | Recall followed by model-selected store with `supersedes`; current record starts where old validity ends. | 23.26 s |
| 3, fresh session B | Indirect booking/current-versus-before question, no recall command; abstain on unknown guide. | Model-selected recall with history, then correct time/current/earlier preference; no invented or other-agent guide. | 18.22 s |

Turn 2 received actual turn-1 conversation history. Turn 3 started without that
history; the harness did not inject prepared memory context into a separate model.
The actual response was: “Booking: Friday at 19:30.” It described guided group
visits as current, quiet visits as earlier, and said no guide name was recorded.

The [synthetic receipt](native-acceptance-results-2026-09-11.json) separates:

- **Storage/revision:** old preference `93450dfd9d954c4b` and replacement
  `3f79a8156c044c32` have the same boundary timestamp; the latter supersedes the
  former. The model selected the revision tool arguments; no scripted revision
  API call prepared the correction.
- **Retrieval:** the fresh-session tool returned five `default`-namespace
  transcript records. It did **not** return the explicit `user`-namespace
  revision family. Thus the correct answer demonstrates use of captured dialogue,
  not a direct read of that revision family in the final turn.
- **Injection/answer:** native tool results preceded the model continuation and
  grounded the correct answer. A separate automatic prefetch sidecar was not
  independently observed in persisted messages. Tool-driven implicit recall and
  prefetch admission are different claims.

Total model-test wall time was **62.94 seconds**, three application turns. Native
`api_calls` counters sum to **7 logical calls**, not a verified physical-request
count. Native counters report input 12,264, output 519, cache-read 11,648 and
cache-write 0 tokens. Physical requests, billing and unexposed internal retries
are unknown. These three examples establish neither an aggregate success rate
nor a performance improvement.

## Finding and scoped correction

The real transcript exposed a capture feedback loop: Hermes `sync_turn` stored
NoldoMem tool results as fresh tool observations, and later recall returned one
of those nested old results. The adapter now recognizes its own tool-call IDs
or explicit result names and excludes those results from capture. External tool
derivatives remain captured as derived, generated evidence. OpenClaw already
excludes its own memory tools in `shouldCaptureToolEvent`; no sibling change was
needed.

The changed adapter was checked without additional model calls through the
installed native loader/MemoryManager, real tool dispatch and a fresh temporary
API/DB: duplicate sync produced no recalled-memory copy; an external document
result survived; forgetting removed the fact; replaying the old tool result did
not restore it or put it into prefetched context. The adapter digest is recorded
in the receipt. Reproduce with `scripts/check_hermes_stable.py --host <checkout>
--memory-echo-only` in an isolated environment with the host and API dependencies.
The installed-runtime execution used separate native/API interpreters, avoiding
incompatible dependency environments. Earlier harness setup failures sent no
model request and are not counted as passing integration checks.

Focused adapter tests: **97 passed, 1 skipped**. Required Ruff checks passed.
`pip-audit` on the isolated no-NLP environment found no known dependency
vulnerabilities, with no new ignore; the local unpublished NoldoMem distribution
itself has no PyPI advisory mapping. The bounded filename-only secret scan returned
the same 36 heuristic files as before, requiring review rather than certifying
absence of secrets. No dependency, embedding model or schema changed here.

This is prospective exclusion, not retroactive cleanup of captured tool echoes.
User/assistant quotations, independently captured dialogue and unidentified old
tool results are not automatically assigned a revision lineage. Their temporal
interpretation still depends on contextual evidence; the three successful
answers do not prove arbitrary stale-transcript suppression.

## Remaining acceptance

OpenClaw 2026.9.3 / Node 24.19.0 subsequently started **one of five** permitted
application turns, using 5.91 seconds of the 600-second host budget. The existing
Hermes counts are unchanged. The [OpenClaw receipt](openclaw-native-attempt-2026-09-11.json)
records the failed turn; four application turns remain, without a budget reset.
The independently reported access canary is not this test or memory acceptance.

The candidate native loader, tools, `before_prompt_build` and `agent_end` hooks
passed preflight with a synthetic config. The real `agentExecCommand` then ran
with `baseConfig` supplied in memory, temporary workspace/state, candidate plugin,
separate temporary API/SQLite DB and a narrow memory-tool allowlist. The selected
route was OpenAI / `gpt-5.6-sol`, native Codex runtime, stdio, agent-scoped home.
No production config snapshot was passed into the run. The native entry point
scoped saved-access resolution internally; the harness never read credentials.

The native envelope returned `status=error`, `kind=exception` and
`Explicit auth order for openai has no usable profiles.` It reported no model,
provider, answer or token usage; zero memory tools ran and no primary-agent memory
record was captured. The wrapper process exited normally after returning that
error envelope, which is **not** a successful native turn. Physical requests remain
unknown; application-turn counts are not physical request counts.

This is a distinct isolation/auth-ownership boundary, not a repeated cooldown
claim or evidence that the normal Gateway route is broken. Stable
[`agentExecCommand`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/commands/agent-exec.ts)
calls `withAuthProfileStoreAgentDir` after repointing run state. In
[`auth-profiles/store.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/agents/auth-profiles/store.ts),
that function retains only portable static credentials from the shared store to
avoid creating a second OAuth refresh owner. The installed public bundle has the
same filter; its artifact digest is in the receipt. The upstream
[`agent-exec.auth.test.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/commands/agent-exec.auth.test.ts)
explicitly expects shared OAuth to be absent even when `copyToAgents=true`.
This is source/test-contract evidence, not a claim that this upstream test suite
was rerun locally. `selectProviderModelAuthSources` in
[`provider-model-route-auth.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/agents/provider-model-route-auth.ts)
emits the observed rejection when the explicit profile order resolves empty or
unavailable. No ownership filter was bypassed or auth snapshot copied to a new store.

The remaining learning, correction, fresh-session current/history and abstention
checks are unexecuted on OpenClaw's full native loop. Existing loader/hook/API/DB
proof remains separate. The narrow next option requiring authorization is a
fresh, official OAuth sign-in to a temporary **agent-local** test store using the
existing account, followed by the remaining bounded turns. It must not copy or
migrate a shared credential, alter production auth, or substitute a paid route.
The installed `models auth login --provider openai --agent <test-agent>` supports
explicit agent targeting; it was not invoked here. No host bridge or new account
is proposed. This option still needs isolated access validation before another
model turn; it is not a promise that signing in alone completes acceptance.

Hermes audio origin remains an unreleased
[draft host contribution](https://github.com/NousResearch/hermes-agent/pull/107369),
with maintainer CI approval and previously reported host dependency findings
outstanding. OpenClaw settled attachment identity awaits its existing
[maintainer design decision](https://github.com/openclaw/openclaw/issues/109370#issuecomment-5618872251).
Neither was polled again to create activity. No third host bridge was introduced.

Raw audio/image recognition, image-only PDF OCR and real channel attachment
delivery remain unexecuted. Earlier local text-layer PDF and synthetic derivative
checks are reused, not presented as those missing behaviors. Legacy records with
no source and unlinked old graph data cannot gain lost provenance retrospectively.
Those limits remain in the original acceptance contract.

The supported recommendation remains one durable memory authority per agent,
chosen for its workload, with compatible native context/session helpers. These
observations support NoldoMem's native Hermes path without establishing a universal
winner over native memory or synchronized dual writers. Full acceptance remains
open. Owned temporary test processes, profiles and database directories were
removed; the three relevant production process IDs were unchanged. Production
rollout, merge and release were not performed.
