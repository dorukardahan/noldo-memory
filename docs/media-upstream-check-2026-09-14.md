# Remaining media prerequisites checked against upstream

This is a narrow follow-up to the [successful real PNG scenario](media-native-proof-2026-09-14.md),
not a new research cutoff or a change to the tested host targets. No model,
transcription provider, login, package installation or production mutation was
performed for this check.

## Installed/tested versus newer stable

| Host | Tested source | Latest stable inspected on September 14 | Relevant result |
| --- | --- | --- | --- |
| OpenClaw | 2026.9.3, `1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7` | [2026.9.4](https://github.com/openclaw/openclaw/releases/tag/v2026.9.4), `3a9d69db306cd7f081e06254cb89c4bcc14a7107`, published September 11 at 03:46:22 UTC | Contextual OpenAI audio provider and multipart adapter are byte-identical. Runner changes rename prepared-catalog reads; they do not repair the observed HTTP rejection. |
| Hermes | v2026.9.7, relevant installed files matched `2237be355906fbe6065ce1815711eee52b2d646e` | [v2026.9.11](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.9.11), `939e45c91d751fadd94dcd1b873ac3cb44846213`, published September 11 at 19:20:31 UTC | `transcription_tools.py`, `transcription_local.py` and `transcription_cloud.py` are byte-identical. Updating these paths would not supply an absent local engine. |

Current main was also identified once: OpenClaw `94ef3b6083a1ea9a4c3f74628577f1e77e00b0e5`,
Hermes `5eb99eb2844b22ebb723711b8e6a0bbb80bb5f04`. These are not the stable
versions used in acceptance. No host was upgraded.

## OpenClaw audio: supported authentication is not account entitlement

[PR #135855](https://github.com/openclaw/openclaw/pull/135855) merged on
September 2 as `9bbba7bca436a3a9d99616d36cf4915ef7874582`, an ancestor of the
tested stable. It already restored OAuth audio selection. Its author's successful
synthetic requests establish that account's behavior, not this test account's
entitlement or a guarantee that every subscription includes batch transcription.

In [`transcribeOpenAiAudioWithContext`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/extensions/openai/audio-transcription.ts),
an explicit profile stays locked; native OAuth uses the official endpoint without
custom overrides. [`transcribeOpenAiCompatibleAudio`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/media-understanding/openai-compatible-audio.ts)
builds the multipart request and sends it to `/audio/transcriptions`. An actual
HTTP failure does not authorize silently switching accounts or providers.
The earlier `ProviderHttpError` therefore cannot be explained as the old local
API-key-only filter. It also cannot establish quota exhaustion or a malformed WAV.

The installed public `openclaw/plugin-sdk/provider-http` was exercised with
synthetic `Response` objects, with **zero network/provider/model calls**:

| Simulated status | Preserved code | Status retained by `String(error)` |
| --- | --- | --- |
| 401 | `invalid_api_key` | Yes |
| 403 | `permission_denied` | Yes |
| 429 | `insufficient_quota` | Yes |

[`createProviderHttpError`](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/agents/provider-http-errors.ts)
preserves numeric status and normalized code. The prior bounded receipt retained
only the error class. That evidence loss belongs to the test observation, not
NoldoMem capture. The old status cannot be recovered from the retained class name.
A subsequent authorized request must retain only status and a bounded code label,
never raw response bodies, headers or credentials. No third host contribution or
new transport is needed for that observation.

## Hermes audio: existing implementation and unreleased alternatives

The stable [`_get_provider`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/tools/transcription_tools.py)
honors explicit local selection. [`_resolve_openai_audio_client_config`](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/tools/transcription_cloud.py)
uses the selected direct audio credentials or managed gateway; the working
`codex_responses` text session is not itself that audio credential path.

[PR #77840](https://github.com/NousResearch/hermes-agent/pull/77840) and its broader
successor [PR #106640](https://github.com/NousResearch/hermes-agent/pull/106640)
remain open and unmerged. They propose Codex subscription speech through private
ChatGPT endpoints and additional ownership/lifecycle handling. They are neither
stable support nor a small NoldoMem plugin fix. No code was ported, no duplicate
contribution was opened, and the private route was not used.

The existing stable local route is sufficient in principle: its official
[`voice` extra and lazy feature](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/tools/lazy_deps.py)
pin `faster-whisper==1.2.1`, `sounddevice==0.5.5`, `numpy==2.4.3`. The missing
engine is an isolated test prerequisite; it does not establish broken production
voice or text memory. Installing into the running host is unnecessary.

The inspected official lock also pins `ctranslate2==4.7.1`,
`onnxruntime==1.27.0`, `tokenizers==0.22.2`, `av==17.0.0`, and
`huggingface-hub==1.24.0`. Registry metadata for these eight packages has candidate
Linux/Python 3.11 or platform-independent wheels, none yanked, all older than seven
days. A clean `uv 0.8.22 pip compile` then resolved the complete **27-package**
closure for Linux x86_64/Python 3.11, constrained by that official lock, with
distribution hashes, `--only-binary :all:` and an exclusion cutoff of September 7.
No package was installed or built. The resolver reported its local Python 3.14.7
because Python 3.11 was absent locally; this is cross-platform metadata resolution,
not proof that imports or native CPU kernels work on the target runtime.

The existing `pip-audit 2.10.0` audited those complete pinned requirements with
dependency resolution/pip execution disabled and found **no known vulnerabilities**.
All 27 entries were also checked against PyPI's version-specific advisory metadata.
No ignore/exception or lifecycle/age control was weakened. This is not proof that
packages are uncompromised, nor an installation approval. The [receipt](media-upstream-check-2026-09-14.json)
records source hashes, resolved versions, lock digest and the native error probe.
Selected target hardware metadata confirms x86_64, glibc 2.39 and SSE4.1/AVX/AVX2/FMA.
Native engine imports and kernel execution remain untested until installation.

## Smallest remaining execution, without changing the product architecture

1. **Hermes prerequisite:** prepare a temporary interpreter using the pinned
   official local voice dependency set, with no write to the installed runtime.
   Use the host's existing local transcription function and a temporary model
   cache; retain its silence/confidence guards. Perform target-wheel/import
   and CPU checks before the existing synthetic WAV. This needs the previously
   withheld new-dependency authorization. The original two Hermes application
   turns remain unused; this is not a budget reset or a new cloud provider.
2. **OpenClaw remaining answer check:** reuse the retained isolated OAuth and
   normal Gateway, enable the already installed document extractor only in the
   test profile, and admit the prepared WAV/PDF in one input batch. Preserve a
   numeric audio failure status/code if it fails. One learning application and
   one new-session packing/meeting/unknown-guide question, at most 120 seconds
   each and 240 total, would test the remaining audio/PDF answer chain. Do not
   rerun the completed PNG or text/correction scenarios. This requires an
   additional bounded application/media allowance; previous allowances stay spent.
   A failed audio upload stops without an alternate account/provider or retry loop.

The private next-run harness is prepared, but not executed: it uses official
`gateway run --verbose`, a native `gateway call health` preflight before any model
slot, a fresh synthetic DB and a new ledger that preserves previous consumption.
Only numeric audio-error status and allowlisted error-code labels enter its
diagnostic receipt; a missing field remains unknown. This changes the deficient
observation condition instead of blindly repeating the old audio attempt.

This plan is not permission to perform either new operation. It does not require
new OAuth, a live deployment, an embedding experiment, a proxy, or a host bridge.
Real outbound delivery still requires a defined test recipient and separate send
authorization. Scanned PDF OCR and missing legacy provenance remain separate.

The existing [Hermes metadata PR #107369](https://github.com/NousResearch/hermes-agent/pull/107369)
remains open at `35bbc894ee6ab5378ee01b822844c613f4ecfd54`, without CI results.
[OpenClaw #109370](https://github.com/openclaw/openclaw/issues/109370) still has no
maintainer response to the proposed bounded terminal receipt scope. Their public
contract/CI gates were checked once; no repeated poll, comment or duplicate work
was added.
