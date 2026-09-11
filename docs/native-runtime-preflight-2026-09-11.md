# Native runtime preflight, 2026-09-11

This corrects the backend interpretation in the [September 10 follow-up](host-candidate-follow-up-2026-09-10.md).
It adds no host bridge, auth adapter, provider substitution or production change.
The ten-request follow-up allowance remains **0 used: Hermes 0/5, OpenClaw 0/5**.
The earlier eight requests remain separate, limited evidence. No inference request
was attempted during this preflight, including failed or ambiguous requests.

## Backend selection is not a memory-health test

Hermes v2026.9.7 (`2237be355906fbe6065ce1815711eee52b2d646e`, runtime version 0.21.1)
maps `openai-codex` to `codex_responses` in
[`hermes_cli/runtime_provider.py`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/hermes_cli/runtime_provider.py).
`_maybe_apply_codex_app_server_runtime` selects `codex_app_server` only with the
corresponding explicit runtime preference. A saved Codex CLI login does not select
that backend for a Hermes installation.

Narrow native config/status checks established that the previously identified
app-server omission does **not** describe the checked Hermes primary routes.
Provider, model, memory provider and runtime preference were checked separately;
an external-memory profile and a native-memory profile are not interchangeable.
Dynamic routing can change a particular turn, so configured defaults are not a
claim about every historical response. No conversations were inspected.

The installed direct-Responses transport, provider resolver and dispatch helper
matched the selected stable source bytes. This verifies the applicability of the
transport probe below, not every downstream host file or answer behavior.

| Public source file | SHA256 |
| --- | --- |
| `agent/codex_runtime.py` | `4840d9a3c2f419bbe899d9d955c03c9acd311e060220801f801cd111c152326c` |
| `hermes_cli/runtime_provider.py` | `92802d9e089e829db36d0fac8bd1e61133ac9b6a1dbcbb63c18594c7a184aef2` |
| `agent/chat_completion_helpers.py` | `4377f55f4965b0f09a7c1a848af47d23607e65d650255d036fac80e6237d3e47` |

## Hermes: the no-retry precondition fails before inference

[`run_codex_stream`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/agent/codex_runtime.py#L866)
sets `max_stream_retries = 1` internally. An `httpx.ReadTimeout` reconnects through
`responses.create` once. The outer API-attempt setting and `HERMES_STREAM_RETRIES=0`
do not disable this inner retry. The latter setting controls another dispatch path.
[`_dispatch_nonstreaming_api_request`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/agent/chat_completion_helpers.py#L678)
also dispatches `codex_responses` through this function; disabling UI streaming
does not change it. `pre_api_request` in `agent/turn_api_request.py` runs outside
this reconnect loop, so counting that hook is insufficient.

The following probe was run in an isolated temporary HOME/HERMES_HOME with the pinned
host's existing dependencies and `HERMES_STREAM_RETRIES=0`. It invokes the real
transport function through its explicit client parameter. The synthetic client
cannot send a network request; it raises before creating a stream. No host source
or installed client was patched.

```python
import logging
from types import SimpleNamespace
from httpx import ReadTimeout
from agent.codex_runtime import run_codex_stream

logging.disable(logging.CRITICAL)
attempts = []

def create(**kwargs):
    attempts.append(1)
    raise ReadTimeout("synthetic offline timeout; no request sent")

agent = SimpleNamespace(
    model="synthetic-model", provider="openai-codex", session_id="",
    _interrupt_requested=False, _api_max_retries=1,
    _client_log_context=lambda: "synthetic", _touch_activity=lambda *_: None,
)
try:
    run_codex_stream(
        agent, {"model": "synthetic-model", "input": "synthetic"},
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
except ReadTimeout:
    pass
assert len(attempts) == 2
```

Observed: **two synthetic stream-factory attempts, zero physical requests**. This
is failure-path evidence, not an inference timeout or successful memory test.
It prevents starting the planned run under its automatic-retries-disabled rule;
it does not establish a production memory defect or authorize another host patch.

The native global-auth fallback in `hermes_cli/auth.py` can separate a profile's
state from existing native access without copying credentials. A fresh profile
without an auth store exits the fork-healing path before its lock/write in
`hermes_cli/auth_oauth_grants.py`. Consequently, the earlier general concern about
status/healing is not proof that native profile reuse is impossible. Actual OAuth
resolution/refresh was not invoked: the transport precondition had already failed.

## OpenClaw: supported isolation exists; acceptance remains unstarted

The installed package is OpenClaw 2026.9.3, running Node 24.19.0. Source references
use stable `1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7`.
[`agent exec`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/docs/cli/agent.md#agent-exec)
is the existing native temporary-state entry point. In
[`agentExecCommand`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/commands/agent-exec.ts),
configuration is composed in memory, sessions use separate state, and
`withAuthProfileStoreAgentDir` supplies a scoped native auth view. This is a
supported option to assess, not a reason to develop an auth-copying workaround.
Its shared-store view intentionally excludes non-portable OAuth ownership.

Both the installed-runtime CLI and the normal native CLI reported `missing` for
the selected OpenAI/Codex auth route through `models status --json` without a
probe. No raw config or credential contents were inspected. That selected CLI result does
not diagnose all Gateway sessions, alternate routes or global login health.
Native status metadata showed more than one runtime in use; the Hermes-specific
app-server omission cannot be transferred to OpenClaw's separate implementation.

The inspected native budgets are not an established physical-request cap:
`createAgentToolExecutionBudget` counts tool admissions, while
[`prepareCodexAttemptTurnRequest`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/codex/src/app-server/run-attempt-turn-request.ts)
explicitly labels model diagnostics `observationUnit: "turn"`. Neither demonstrates
counting internal model continuations/reconnects. This is an unverified precondition,
not a claim that every possible native configuration lacks such a limit.
No paid test was started to discover that limit after the fact. No alternate paid
provider, live agent or production memory was used.

## What remains

The existing five-request-per-host learning/correction/fresh-session/current-history/
unknown-detail plan is unchanged. It has no new retrieval, injection or answer
result. The concrete preflight need is a supported isolated invocation with usable
existing access and enforceable physical-request/no-retry controls. More request
allowance alone would not establish that capability; the allowance is still unused.

Hermes audio provenance stays in [draft #107369](https://github.com/NousResearch/hermes-agent/pull/107369),
with fork CI awaiting approval and the previously reported dependency audit findings
unresolved. OpenClaw settled attachment evidence stays behind the existing
[#109370 design decision](https://github.com/openclaw/openclaw/issues/109370#issuecomment-5618872251).
No duplicate contribution or third host bridge was created. Raw recognition and
real external attachment delivery remain separate unexecuted acceptance; local
PDF extraction and synthetic metadata tests do not close them. Old missing source
identity still cannot be reconstructed by adding new metadata.
