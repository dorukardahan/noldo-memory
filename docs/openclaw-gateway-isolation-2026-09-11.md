# Existing Gateway isolation boundary, 2026-09-11

The installed OpenClaw 2026.9.3 / Node 24.19.0 was rechecked. The official source
examined is stable commit `1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`.
NoldoMem was `a01eff3c51a4249a5b8674251eb46b80686b23a1`, with a clean tree.
This is a source/control-flow review, supplemented by installed public-bundle
markers, **not a new native integration test**. No Gateway agent/session request,
login, logout, auth refresh or model call was made in this investigation.
The earlier failed isolated exec was not retried; normal access remains a separate
previously established fact. No new cooldown diagnosis was performed.

## Why the existing Gateway does not supply the required isolation

The following functions are all at the stable commit above:

| Entry or boundary | Actual flow and consequence |
| --- | --- |
| [`agentRunHandler`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/server-methods/agent-run-handler.ts) | Validates the request and calls preflight/turn service. It does not accept a replacement test config. |
| [`prepareAgentRequestPreflight`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/agent-turn/agent-request-preflight.ts) | Starts with `context.getRuntimeConfig()`. `cwd` is restricted to plugin-owned subagent runs. Internal/suppression options are guarded handoff controls, not memory endpoint or state-root overrides. |
| [`prepareAgentRequestRouting`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/agent-turn/agent-request-routing.ts) | Rejects an unknown agent ID against `listAgentIds(cfg)`. A new synthetic agent would need registration in the active configuration. |
| [`createAgentTurnService`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/agent-turn/agent-turn-service.ts) / [`persistAgentSessionPhase`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/agent-turn/agent-session-persist.ts) | Uses the Gateway config to prepare the session/store. Ordinary visible runs patch the session entry. Suppressing visible effects can skip that patch; it does not replace the plugin client or make all state disposable. |
| [`sessions.create`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/gateway/server-methods/sessions-create.ts) | Also resolves against `context.getRuntimeConfig()` and the Gateway session lifecycle. It is not a temporary profile factory. |
| [`attempt-execution.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/agents/command/attempt-execution.ts), `disableTools` | `opts.modelRun === true` disables tools. A raw one-shot infer request cannot demonstrate native memory-tool learning and recall. |

This conclusion does not rely only on the closed
[`AgentParamsSchema`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/packages/gateway-protocol/src/schema/agent.ts).
The downstream plugin boundary matters: the host
[`api-builder.ts`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/plugins/api-builder.ts)
passes its resolved config/pluginConfig to registration. NoldoMem's
[`register`](../plugin/index.js) builds one client from `api.pluginConfig.baseUrl`
and registers tools/capture/recall with that client. A new session key, `internal`
flag or prompt text does not select a temporary NoldoMem API/DB or load candidate
code into the already-running Gateway. Turning auto hooks off also leaves the
registered memory tools using that client.

Installed public bundles contained the corresponding runtime-config, unknown-agent,
plugin-owned-cwd and `modelRun` guards. These string checks establish artifact
correspondence only; they are not behavioral test passes. No production session
was created to demonstrate a write that the source already makes explicit.

Thus the examined supported Gateway paths cannot combine its existing OAuth owner
with this candidate and a disposable memory/session configuration under the current
no-production-mutation constraint. This is a test isolation/access limitation,
not evidence that production memory or every future Gateway API is broken. No
new bridge, proxy, host patch or memory-sharing mechanism is proposed.

## One proposed authorization plan, not executed

1. Under the existing unprivileged runtime user, create a private, empty test root.
   Set HOME, OpenClaw state/config paths and Codex home explicitly inside it with
   an allowlisted child environment. Generate a synthetic config, never clone a
   production profile. Register a **non-main** `synthetic-memory` agent only there,
   candidate NoldoMem with a temporary API/DB, and the installed OpenAI/Codex
   runtime. Configure the native Gateway target to an unused loopback port with
   no remote target or inherited Gateway overrides. No new Gateway service is
   started. Verify all configured roots and the model-free loader path first;
   stop if any repair would require modifying the installed runtime.
2. Only after explicit authorization, run the installed CLI in that environment:
   `openclaw --profile noldomem-acceptance models auth login --provider openai --method oauth --agent synthetic-memory --profile-id openai:noldomem-acceptance`.
   Use the existing account's interactive native flow; no new account, credential
   copy, `--force`, `--set-default`, export or OAuth material in reports/arguments.
   Any manual callback is entered by the operator into the native prompt, not
   relayed through chat or an agent tool command.
3. Validate access using native redacted status before inference. Then use native
   `agent exec` with the synthetic config's same non-main auth owner and retained
   temporary run state, exposing only candidate memory tools. Plan three turns:
   learn in session A; naturally correct in A; indirectly ask current/previous
   details and an unknown detail in new session B. The fourth remaining turn is
   available only for a distinct necessary follow-up, not an external retry loop.
   Existing limits remain four turns total, 120 seconds per turn and 594.09 seconds
   remaining host model time. A timeout stops the local process; remote cancellation
   and billing may remain unknown. Native continuations/retries are not separately
   counted as application turns.
4. Stop only owned test processes and remove only the owned temporary root after
   retaining public-safe synthetic evidence. Do **not** invoke logout, revoke or
   account-wide sign-out. Local deletion removes the temporary credential/state
   artifacts; it does not revoke a provider grant or guarantee its remote expiry.
   Confirm production service identities remain unchanged through safe metadata.

### Store ownership and side effects behind this plan

[`runProviderAuthMethod` / `persistProviderAuthResult`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/commands/models/auth.ts)
passes the selected agent directory to the locked login upsert and updates auth
ordering. A non-main agent directory stays agent-local under
[`prepareFreshSharedAuthStoreWrite`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/agents/auth-profiles/shared-store-bootstrap.ts)
and [`runAuthProfileWriteTransaction`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/agents/auth-profiles/sqlite.ts):
the intended owner is the synthetic agent's `openclaw-agent.sqlite`, wholly under
the test root. Shared-owner resolution must also stay inside that root; selecting
`--agent` alone is insufficient.

OpenAI's [`runOpenAICodexOAuth`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/openai/openai-chatgpt-provider.ts)
returns a config patch. Login may therefore write the **test config** even without
`--set-default`, and attempts native runtime repair. It also unconditionally calls
[`refreshRunningGatewayAuthState`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/commands/models/auth-refresh.ts),
which sends `models.authStatus(refresh=true)` to its configured target. This is
why the plan pins an unused temporary target instead of relying on a profile
name alone. No production Gateway refresh is authorized by this plan.

The native [`loginOpenAICodex`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/openai/openai-chatgpt-oauth-flow.runtime.ts)
creates a new authorization flow and exchanges the code for access/refresh tokens.
The source does not promise that another login leaves every existing provider-side
session/refresh token unaffected. Its token implementation explicitly handles
invalidated/reused refresh tokens. **Local owner isolation is not a guarantee of
zero provider-side effect on the existing account.** No claim of an observed
invalidation or guaranteed disruption is made either.

[`modelsAuthLogoutCommand`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/commands/models/auth-logout.ts)
removes profiles across local owner stores and refreshes its Gateway target; it
does not call a provider revoke endpoint in that handler. The proposed cleanup
avoids it entirely so that neither cross-owner removal nor a Gateway refresh is
needed. No login/cleanup behavior is claimed as executed. Authorization for this
new login, including its unguaranteed provider-side effects, remains the single
access decision before attempting the remaining OpenClaw native scenarios.

The budget remains **1/5 turns, 5.906240866985172 seconds used**, as recorded in the
[earlier receipt](openclaw-native-attempt-2026-09-11.json); physical model requests
remain unknown. Hermes native evidence, pending host metadata contributions and
unexecuted raw-media/real-delivery acceptance are unchanged. No goal criterion is
marked complete by this investigation. Production process IDs were unchanged;
no temporary test process or runtime directory was created in this investigation.
