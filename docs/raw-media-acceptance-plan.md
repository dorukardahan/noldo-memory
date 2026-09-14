# Bounded incoming raw-media acceptance plan

Status: authorized and partially executed on **2026-09-13**;
[results and remaining limits](raw-media-results-2026-09-13.md) recorded on
2026-09-14. OpenClaw used both application turns; Hermes stopped at preflight
before model calls. Earlier text-only allowances and successful tests are separate.
The [later PNG proof](media-native-proof-2026-09-14.md) and
[remaining upstream prerequisites](media-upstream-check-2026-09-14.md) supersede
its pending-image and fresh-login assumptions. The existing isolated access is
retained; the original execution instructions below are historical, not a new grant.

## What changes technically

The independent native UTF-8 file chains now work and the adapters distinguish
raw attachment presence from supplied derivatives. The next missing observation
is whether raw PNG/audio/PDF bytes, processed by the actual native host, create
usable text/evidence that later affects an answer. SDK completions, local decoders
and manually supplied descriptions do not establish that behavior. This test may
still fail if a harness interprets bytes without a reusable memory derivative;
that result must be diagnosed rather than replaced with hand-written context.

## Prepared synthetic inputs and expected behavior

A local review bundle contains only these independent fixtures. Its expected-answer
manifest stays outside the host/model workspace and is never injected:

| File | Source | Expected observation |
| --- | --- | --- |
| `entrance.png` | Locally drawn 640 × 320 sign and blue door | Aurora visit entrance is the blue door. |
| `packing.wav` | Local synthetic speech, “For the Aurora visit, bring a red notebook.” | Bring a red notebook. This is TTS input, not prior ASR evidence. |
| `meeting.pdf` | One-page locally generated text PDF | Meet at the east arch. |

No speaker identity, date, timezone or guide is supplied. A distinct synthetic
agent can have a different guide record to test non-leakage; that setup is not
counted as learning by the tested agent. These files contain no private material.

## One bounded execution proposal

- Use the already selected Hermes v2026.9.7 native gateway/row/MemoryManager path
  with `codex_responses`, and OpenClaw 2026.9.3's normal Gateway startup with
  `startupSettled` and the native Codex harness. Identify the actual tested host
  commit and any approved candidate metadata separately from released stable.
- Use separate temporary HOME/profile/workspace and NoldoMem API/DB per host.
  Hermes uses its previously verified supported saved-access path. OpenClaw
  requires a **new isolated device-code login approved in the browser** because
  the earlier test OAuth store was deleted. Never copy existing credentials or
  run a production profile. No new account/provider or install is assumed.
- Before any provider call, verify isolated memory scope and that the host's
  existing image/audio/file processing is available through approved native
  access. If it needs a new service, auth copying or an unapproved extraction
  provider, stop that host's scenario before calls; do not silently substitute.
- **Two application turns per host, four maximum**. Cap each turn at 120 seconds
  and each host at 240 seconds. Native tools/continuations are part of a turn;
  do not add an outside retry loop or restart failed scenarios. Physical request
  counts may exceed application counts; report safe native usage when available.
- Turn 1: attach the three files through the host's native input surface with
  only “These are the Aurora visit notes. Summarize the practical details.”
  No remember command, expected facts, manual memory write or injected derivative.
  Capture actual extraction/row/tool/hook and resulting memory IDs separately.
- Turn 2, a different session with no attachments: “For the Aurora visit, which
  entrance should I use, what should I bring, where should I meet, and who is the
  guide?” Expect the three sourced details and an unknown guide. Verify that
  automatic recall/injection and the answer use the first turn's records and
  exclude the other agent's record. Keep generated interpretation lower-trust;
  do not claim clip provenance that stable Hermes did not supply.
- Permit at most one input batch of those three files per host. Existing native
  vision/ASR calls are **additional raw-media processing authorization**, not
  hidden inside an exhausted old allowance. No price estimate or zero-cost
  guarantee is supplied. Stop on a newly required paid route/provider decision.
- Delete only owned temporary processes/profiles/test data afterward, without
  logout/revoke or production changes. A local timeout does not prove remote
  cancellation or absence of charges.

## What this will not close

No external message or attachment is sent by this plan. Complete outgoing
per-file delivery remains dependent on the OpenClaw settled-receipt design and
channel-specific receipt evidence, and needs a separately identified test
recipient before an executable send plan is possible. Likewise, stable Hermes
successful-clip provenance requires the pending host metadata contribution.
Scanned PDF OCR, live URL expiry/re-fetch, universal channel coverage and missing
legacy provenance are not implied by three incoming fixtures. No third host
project, new OCR/ASR infrastructure or embedding experiment is proposed.
