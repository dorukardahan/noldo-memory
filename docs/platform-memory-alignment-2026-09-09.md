# Platform memory alignment evidence

Status: scoped implementation and native memory integration validation delivered in PR #34; full functional acceptance remains incomplete. See the [September 10 bounded model observations](model-acceptance-2026-09-10.md).
Research cutoff: 2026-09-09T08:17:03Z (Asia/Singapore: 16:17:03).
Only public sources and independently constructed synthetic examples are used.

## Frozen source baseline

| Component | Ref | Publication / observation |
| --- | --- | --- |
| NoldoMem main | `b9616d093feb1e2ce48338ab679ed6730f30ed6b` | 2026-08-07T22:13:25Z; documentation change |
| NoldoMem release | `v1.27.16` | 2026-07-23T13:44:50Z |
| Last substantive adapter change | `35d06a6`, PR #20 | 2026-07-23; lifecycle and cache isolation |
| Hermes stable | `v2026.9.7`, `2237be355906fbe6065ce1815711eee52b2d646e` | 2026-09-07T22:17:01Z; source release, no binary assets |
| OpenClaw stable | `v2026.9.3`, `1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7` | 2026-09-08T14:15:53Z |

Refs verified using the official GitHub repositories and releases API. Target
tags are stable, not prereleases. The research interval begins at the main
commit, but pre-existing deficiencies are tracked separately. Production
installation state is outside this public evidence record and is not inferred
from release availability.

## Acceptance matrix

| Area | Baseline experiment / evidence | Acceptance |
| --- | --- | --- |
| Host compatibility | pinned host loader, hooks and native memory source | actual isolated execution where feasible; disclose unexecuted paths |
| Architecture | same synthetic episodes, queries and context budget in native / NoldoMem / coexistence | compare authority, duplicate capture/injection, forgetting, tokens |
| Implicit recall | indirect event, paraphrase, typo, preference continuation, unrelated turn | separate retrieval coverage from actual answer grounding |
| Media | incoming/outgoing text, image, audio, document, link | distinguish extracted evidence, reference only, draft and delivered |
| Time | correction, historical query, inference, forgetting | current and previous validity remain distinguishable; deletion honored |
| Isolation | two agents with conflicting same-name events | capture, DB/FTS/vector, caches, workers and injection remain scoped |
| Performance | call counts, cold/warm timings, volume, context size | baseline first; no quality/isolation trade for latency |
| Delivery | focused tests, full applicable audit/build, direct diff, exact-head CI/review | no deployment, merge, release or production data mutation |

## Investigation log

- Existing open PR #31 changes legacy TODO-line capture; #33 changes README
  positioning. Their changes are not imported or represented as this work.
- Codex review availability confirmed by actual reviews on PR #20; CI has one
  lint/test/build/security job. An old review is not review of this branch.
- Baseline defects reproduced before correction: OpenClaw tool scope was read
  from the execute argument; implicit recall depended on punctuation/keywords;
  current-turn capture replayed older user turns; semantic merge combined distinct
  statements. These defects predate the research interval.

## Source and product-impact matrix

Links below identify the frozen sources, not a moving `main` checkout. Earlier
published host support statements (OpenClaw 2026.5.x and Hermes 2026.5.28/v0.19)
are historical declarations, not evidence of currently installed versions. No
production host build or configuration is certified by this report.

| Source / date | Verified behavior and affected surface | Decision / risk / evidence |
| --- | --- | --- |
| [Hermes stable release](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.9.7), September 7 | Source tag pinned above; no binary release assets | Exercise source loader and MemoryManager; do not equate current main with stable |
| [Hermes MemoryManager](https://github.com/NousResearch/hermes-agent/blob/2237be355906fbe6065ce1815711eee52b2d646e/agent/memory_manager.py) | One external provider plus builtin; serialized background sync; accumulated `messages`; bounded external prefetch and output spill | Adapter accepts text blocks from the current turn, keeps host-owned workers; real loader/HTTP test |
| [Hermes alias mirroring fix](https://github.com/NousResearch/hermes-agent/commit/4fbb2539041751c3cf32432af592c7d0425264c2), September 7 | Successful native aliases also notify providers with replacement metadata | Exposes the need for update/delete mapping; refusing unsupported mirroring is safer than recapturing deleted text |
| [Hermes checkpoint contract](https://github.com/NousResearch/hermes-agent/commit/9e551d293133791ae3b3e32dde19488dc15390c1), August 25 | API v2 opts into fail-closed pre-compress checkpoints | NoldoMem does not claim v2 durability; best-effort capture remains a limitation |
| [OpenClaw stable release](https://github.com/openclaw/openclaw/releases/tag/v2026.9.3), September 8 | Source/artifact release pinned above; SDK remains experimental | Tool factories receive trusted scope; `execute` third argument is not agent context. Regression test calls with AbortSignal |
| [OpenClaw QMD removal](https://github.com/openclaw/openclaw/commit/8b0735e89f226fe06db7776b83e8942bfa7ced2f), August 9 | QMD removed from core; builtin remains core engine, separate memory plugins still exist | No QMD integration or obsolete backend recommendation |
| [OpenClaw memory search](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/docs/concepts/memory-search.md) | Hybrid retrieval, importance/recency weighting, MMR, trusted promoted-entry trigger recall; optional session indexing | Correct old README claims. Native is a serious alternative; no extra NoldoMem graph justified |
| [OpenClaw provenance](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/docs/concepts/memory-provenance.md), lineage docs August 26 | Session-origin tracking, retained rewrite preimages, forgotten-session admission prevents tracked replay | Native deletion scope is stronger than NoldoMem's record-only purge; neither erases every external transcript/copy |
| [OpenClaw message media projection](https://github.com/openclaw/openclaw/blob/1391f7cd2d40ab5bbcf2f5f831d3a64f520e72d7/src/hooks/message-hook-media.ts) | Paths/URLs/kind/transcribed facts, not OCR/ASR content by themselves | Reuse existing textual extraction; references alone do not become a claimed memory |
| [sqlite-vec v0.1.9](https://github.com/asg017/sqlite-vec/releases/tag/v0.1.9), March 31 | Fixes deletes with long metadata text columns; outside research interval | NoldoMem vec0 contains the embedding column, not that metadata layout. No dependency upgrade required. Existing default L2 index preserved; fallback corrected to match |
| [sentence-transformers v6.0.1](https://github.com/huggingface/sentence-transformers/releases/tag/v6.0.1), August 31 | PyLate prefix/prompt loading and Router preprocessing fixes; v6 requires newer Torch/Transformers | Optional NoldoMem CrossEncoder path does not use these multi-vector models. No blind major upgrade or new visual index |
| [FlagEmbedding v1.4.2](https://github.com/FlagOpen/FlagEmbedding/releases/tag/v1.4.2), August 24 | Transformers 5 tokenizer compatibility | A model-family dependency, not the HTTP API serving contract. No measured need to replace the serving stack |
| [llama.cpp v0.4.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0), September 4 | Serving, model-loading, state/API and multimodal changes | None proves a BGE-M3 recall improvement. No runtime upgrade or reindex; actual serving build unverified |
| [Qwen3 Embedding source](https://github.com/QwenLM/Qwen3-Embedding/tree/44548aa5f0a0aed1c76d64e19afe47727a325b8f) | Separate embedding/reranking models; repo default names are not runtime identity | Keep OpenAI-compatible HTTP boundary and configured dimensions; no model swap without a matched corpus test |

The code defaults to a Qwen embedding API model, with optional MiniLM L6/L12 or
BGE reranking and a separate API reranker path. The experiment instead used a
`bge-m3-Q8_0.gguf` serving label, 1024 dimensions. The label is not a verified
weight checksum. Its twenty synthetic vectors are replayed offline. HTTP SDK,
NumPy/SQLite/FTS, token counting and Turkish normalization remain existing
components; no new OCR, ASR, graph service, subscription or model download was
introduced.

Official accounts were checked against the OpenClaw README and the
NousResearch GitHub organization profile. The local read-only X client found
[NousResearch's pluggable-memory announcement](https://x.com/NousResearch/status/2090455690519167306)
(August 20) and [profile portability announcement](https://x.com/NousResearch/status/2087592096769147377)
(August 12). The [OpenClaw podcast announcement](https://x.com/openclaw/status/2095349641005170953)
(September 3) discusses memory and skills. Announcements are context, not proof
that an advertised feature exists in the pinned artifact. Cross-bot/profile
sharing mentioned in related announcements is excluded from this design.

## Academic mechanisms and bounded experiments

| Primary work / evidence class | Mechanism → product need | Cost, limitation and experiment decision |
| --- | --- | --- |
| [LongMemEval](https://arxiv.org/html/2410.10813v2), [ICLR 2025 paper](https://openreview.net/pdf?id=pZiyCaVuti) | Turn granularity, fact-aware indexing and time-aware retrieval → distinguish retrieval from correct answer use | Keep source sessions and explicit validity; test current/history and abstention separately. Published scores are not NoldoMem scores |
| [LoCoMo](https://aclanthology.org/2024.acl-long.747/), ACL 2024 | Long multi-session, multimodal episodes → event-based questions about media content | Reuse textual derivatives; test incoming/outgoing modalities and missing sources. Derivative fixtures do not evaluate vision/ASR accuracy |
| [LoCoMo-Plus](https://aclanthology.org/2026.acl-long.1150/), ACL 2026; [official evaluation code](https://github.com/xjtuleeyf/Locomo-Plus) | Implicit constraints and cue mismatch → recall without “remember” or matching media words | Include declarative preference continuation, typo and paraphrase; measure admitted evidence separately. No generated-answer constraint-consistency score is claimed |
| [A-MEM](https://arxiv.org/abs/2502.12110), paper lists NeurIPS 2025; [official code](https://github.com/agiresearch/A-mem) | Linked, evolving notes → associative retrieval and richer episode context | Automatic note generation/linking costs model calls and can rewrite source meaning. Defer a new graph layer; retain distinct evidence before considering enrichment |
| [MemoryBank](https://arxiv.org/abs/2305.10250), cited arXiv preprint | Importance-weighted forgetting/reinforcement → selective retention | Existing decay already addresses this mechanism. No new decay curve or “human memory” claim; explicit forgetting remains stronger than retention heuristics |
| [Generative Agents](https://arxiv.org/abs/2304.03442), primary research | Relevance, recency, importance and reflection → selective retrieval/procedural lessons | Existing rank layers cover most of this. Simulated social-agent results do not establish personal assistant accuracy; preserve lesson scope, avoid another reflection model |

No research implementation was copied. Papers motivate small tests rather than
justify a new framework or benchmark-score marketing claim.

## Architecture comparison

The common corpus contains eight independently authored synthetic episodes and
four unrelated queries. All fit a 3500-character curated-memory budget; the
retrieval replay asks for five candidates. A native startup snapshot and a
selective search have different costs, which are kept separate below.

| Mode | Same-corpus evidence | Authority / cost / limitation |
| --- | --- | --- |
| Hermes native only | Real stable MemoryStore retains 8/8 texts across reopening; 907-character formatted block; real replace/remove succeed | Best small-set coverage in this test with no retrieval HTTP/embedding round trip. Standing context also appears on unrelated turns; that is not by itself a wrong answer. Old value history requires transcript/other native evidence |
| NoldoMem only | 7/8 expected records retrieved; same with calibrated 0.50 admission; other agent absent; API revision/history/forget tests pass | Selective evidence with explicit versions and media provenance, but one indirect image case is missed. One HTTP recall plus an embedding on cache miss; no generated-answer success rate |
| Controlled coexistence | Hermes native session search/skills remain separate from external provider; OpenClaw session tools and procedural skills are independently configurable | Supported pattern is one durable authority plus host helpers. It is not synchronized dual long-term storage. Do not mirror the same facts into two writers |
| Two durable writers | Native replacement/deletion cannot address external IDs; adapter refuses unsupported operations | Reject as a recommended mode. Native additions can still mirror for legacy compatibility; using that path creates a second copy that must be managed separately. Duplicate injection costs up to native block plus NoldoMem context |
| OpenClaw native only | Pinned source documents native search, promoted trigger recall, dreaming, session search and conditional image/audio indexing | Full Gateway/reader experiment not completed. Native cannot be scored as inferior from an unexecuted path |

Recommendation: native-only is the simplest supported choice for a small curated
preference set that fits its budget. For evidence-rich episodic history and
explicit validity, evaluate NoldoMem as the single durable authority with native
session/procedural helpers retained. This evidence does **not** establish a
universal winner or justify changing a production profile. Two independent
writers are not recommended.

### Host configuration boundaries

For pinned OpenClaw, `tools.sessions.visibility: "agent"` preserves the same
agent's session history, and `tools.agentToAgent.enabled: false` blocks ordinary
cross-agent access. `dmScope` alone is not an access boundary. Review spawned
session exceptions and filesystem permissions too. Do not enable native import
of another profile's memory. The plugin's strict tool scope is not a substitute
for scoped server credentials or host access control.

A NoldoMem-only profile must account separately for native MEMORY/USER bootstrap,
trigger recall, dreaming/promotion, flush and direct writes. Disabling legacy
`memorySearch.enabled` is not proof that all those surfaces stopped. Conversely,
disabling all memory-related tools can accidentally remove useful transcript
search. Keep one capture/injection owner per event: typed plugin plus duplicate
legacy hooks is not a latency optimization.

For Hermes, separate agent profiles and explicit distinct adapter `agent` values
are required. The fallback name `hermes` is not automatically unique. Disabling
native `memory_enabled` / `user_profile_enabled` removes their standing context;
keep the external provider's tool surface, transcript search and skill surfaces
as separate decisions. Verify flags against the pinned host before rollout.

## Data flow and trust

```text
host event + trusted agent/session scope
  → existing text blocks or already-produced extraction
  → bounded evidence {role, modality, representation, assertion, delivery,
                      observed_at, event_id, reference, confidence}
  → own agent DB / text + evidence + source_session + validity
  → own FTS/vector candidates → optional rerank → scoped cache
  → current-valid or explicitly historical results
  → untrusted-data context with IDs, validity and provenance
```

| Surface | Captured evidence | Boundary / uncovered part |
| --- | --- | --- |
| Incoming text | Latest user turn; source session | Heuristics still omit some short events; not exhaustive transcript capture |
| Incoming image/audio/document/link | Existing text derivatives through capture API; mixed blocks in OpenClaw; Hermes text blocks | No binary understanding or network fetch. A URL/path alone is not extracted content; OCR/ASR/caption quality is not measured |
| Outgoing text | OpenClaw successful `message_sent` marks delivered; Hermes sync marks generated | Generation success is not transport receipt. Missing host scope is rejected |
| Outgoing media | API can preserve a supplied delivery/extraction assertion | Pinned outgoing hook does not expose the full attachment payload. No proof of end-to-end delivered media capture |
| Expiring/missing attachments | Derivative remains usable with original reference stripped of query/fragment | Does not recover lost content; reference-only input does not invent facts |
| Long text | Capture heuristics inspect the bounded prefix; successful outgoing delivery and full-text injection guards still apply | Text beyond the existing 2000 UTF-16-unit bound is not retained; truncation does not split a surrogate pair |
| Duplicate/async events | Exact provenance-matched retries deduplicate before embeddings; in-flight cache generations fence writes/forget | Unkeyed replay after deletion is not a forgotten-source admission system |
| Same-agent subtasks | Trusted scope plus child source session | Cross-agent target session is refused; no preference-sharing pool |

OpenClaw screens raw evidence values, record IDs and type labels before JSON
escaping in both automatic and explicit recall. Its existing automatic text
screening and untrusted-data envelope remain. Hermes renders only selected
provenance fields and applies the stable host sanitizer to the assembled line.

Assertions `reported`, `derived` and `inferred` describe the supplied evidence,
not cryptographic proof of authorship. Text is still untrusted input. A caption
or assistant draft is never automatically elevated to a confirmed user update.
No stored reference is dereferenced by the API.

### Validity, recovery and deletion

`POST /v1/store` with `supersedes` atomically closes one current row and creates
its replacement. `valid_from` is event-valid time; `created_at` is learning time.
Omitting `valid_from` uses current learning time and must not be described as a
known earlier event date. Old records have unknown start times. Similarity alone
never establishes a correction. Derived/inferred corrections are rejected; the
caller must identify a confirmed user correction and the prior ID.

Normal recall excludes closed/future versions. `as_of` selects an interval;
`include_history` returns previous versions. A narrow historical-language hint
is a convenience, not complete temporal understanding. The model still has to
recognize a correction and call the tool; automatic natural-language revision
accuracy has not been measured. `/v1/rule` preserves supplied evidence/session
provenance and directs explicit revision requests to `/v1/store`. Concurrent stale revisions fail instead of
creating competing current branches.

Synthetic tests cover rollback after insertion failure and export/import of
validity/evidence. Imports reject malformed intervals and cyclic/cross-namespace
lineage before writes. Partial imports validate references against retained
existing rows when duplicate IDs are skipped, and reject references to skipped
empty-content parents or branching histories. Imports prepare embeddings
before taking a write lock, then revalidate retained parent/child boundaries
and commit all rows/vectors atomically. Overwriting an existing revision must
keep its parent link; detaching or reparenting it is rejected even in a batch.
Failed or raced imports leave no partial
restore. Consolidation does not merge provenance-bearing or
versioned rows. Decay still adjusts their ranking strength but does not archive
explicit revision families; they remain available until explicit forgetting.
Unversioned rows retain the existing decay/archival policy. No existing live database was
migrated or reindexed.

`DELETE /v1/forget` and scoped `noldomem_forget` delete the connected revision
family, FTS/vector entries and linked temporal facts, invalidating caches.
Historical recall includes closed/current versions, excluding future starts unless
an explicit `as_of` selects that time. One request-start clock snapshot governs
all search lanes. Explicit `as_of` suppresses the query-derived ingestion-date
filter, so a fact learned later can still be found at its valid time.
Query-based forgetting also searches closed/future revisions, so an old-only
phrase can identify the family. Forgetting does not erase source transcripts, previously emitted context, backups, native
memory or arbitrary re-imported copies. Native/OpenClaw session-level forgotten
admission is a stronger capability in that specific respect.

## Measurements and decisions

Machine-readable results: [baseline](alignment-baseline-results.json) and
[candidate](alignment-candidate-results.json). Baseline is the pinned main.
The original full suite passed 438 tests with one skip before implementation.

Twenty real synthetic embedding calls produced the replay fixture, with median
284.512 ms. No p95 is reported for twenty calls. This excludes any claim about
serving hardware, artifact identity or production throughput. Subsequent
benchmarks make zero model/network calls and use those same vectors.

The eight-case retrieval result is unchanged at 7/8. All four unrelated queries
initially returned candidates. On this same small calibration set:

| Admission floor | Expected evidence admitted | Unrelated queries with context | Total related-context text characters |
| --- | --- | --- | --- |
| 0.45 | 7/8 | 1/4 | 3581 |
| 0.50 | 7/8 | 0/4 | 2461 |
| 0.55 | 6/8 | 0/4 | 1226 |

These scores use the existing `1 - L2/2` scale, not cosine probabilities. The
floor is optional and has no universal default. Explicit search stays available
when automatic admission abstains. A held-out, multilingual deployment corpus
is needed before claiming general precision or setting a production threshold.
Character counts are exact; token budgeting uses the existing approximate
estimator, not an identified host tokenizer. Generated answers and complete
host response latency are not measured.

The candidate adds scope, validity and provenance work. Offline cold/warm
medians and sample p95 (100 observations each) are in the JSON results. Volume
stress uses 1000/10000 repetitions of identical encoded text, 30 observations
per size, reporting medians only. It tests storage cost, not larger-corpus recall
quality. A join-before-limit FTS candidate regressed the 10000-row median from
23.080 to 43.125 ms. `EXPLAIN QUERY PLAN` identified materialization/sorting;
progressive ranked FTS candidate widening reduced that candidate to 29.303 ms
while preserving sparse scope/validity admission. No overall speedup is claimed.
The correctness budget prioritizes no stale/cross-agent evidence; the remaining
small local overhead is reported rather than hidden behind remote model time.

Exact `/store` retry avoids a second embedding when synchronous embedding is
configured. `/capture` deduplicates both within a batch and against stored
provenance before embedding. These are call-count improvements, not a measured
end-to-end response speedup. Hermes's local TTL defaults to zero because another
session cannot invalidate that private cache; server cache and own-write
invalidation remain. Shared reranker score caches include the agent, record ID and current text,
preventing cross-agent score reuse and reuse after an in-place edit.
Only results admitted after API semantic/token filtering receive access boosts.
An absent embedder is also degraded, including a zero admission floor.
Degraded and deliberately lexical-only searches do not populate the semantic
result cache; its new key namespace excludes old entries without degradation
metadata. Repeated outage tests cover both zero and nonzero admission floors.
Search returns a list-compatible batch with request-local mode/degradation
status; overlapping healthy/outage calls cannot change each other's admission
or cache eligibility. Legacy last-completed diagnostics are not used by recall.
Versioned searches currently bypass search-result caching
to avoid future validity-boundary staleness, a deliberate performance cost.

| Decision | Reason and verification |
| --- | --- |
| Implement | Trusted OpenClaw tool factories; implicit prompt gate; current-turn capture; evidence and validity; exact retry dedup; write fences; scoped forgetting; FTS admission and L2 parity |
| No-op | Existing per-agent DB design, host worker ownership, reranker deployment, decay curve, model/embedding dimensions |
| Defer | Automatic graph enrichment, binary OCR/ASR pipeline, guaranteed automatic semantic supersession, forgotten-source replay admission, synchronized dual-authority mode |
| Unknown | Generated-answer correctness, full delivered-media extraction, production host identity/configuration, real OpenClaw Gateway integration and complete response latency |

## Reproducing the isolated checks

Use a disposable HOME/data directory and an existing reviewed Python test
environment. No production archive is an input.

```sh
python scripts/evaluate_memory_alignment.py --repo /path/to/pinned-baseline
python scripts/evaluate_memory_alignment.py --repo /path/to/candidate
python scripts/check_hermes_stable.py --host /path/to/hermes-v2026.9.7-source
python -m pytest tests/test_alignment_api.py tests/test_alignment_behavior.py
bash scripts/audit.sh
```

`check_hermes_stable.py` creates its own temporary profile and loopback API,
loads the real stable provider and exercises actual MemoryManager capture,
recall and context assembly. Embedding failure is intentionally injected, so
this proves the degraded integration path, not semantic model accuracy.
Native MemoryStore is also exercised through its real file-backed API.

The follow-up `runtime_test_api.py` and `check_openclaw_runtime.mjs` use the
installed stable OpenClaw 2026.9.3 loader and hook runner on its Node 24.19.0,
with a separate candidate directory, unprivileged temporary HOME/state and real
temporary HTTP API/DB. The loader source must resolve to the candidate plugin.
Native current-turn capture, confirmed outgoing text, cross-session automatic
injection and agent isolation passed. The native conversation-access grant is
required; without `hooks.allowConversationAccess: true`, non-bundled capture
hooks do not register. The supported grant is now documented.

Two independently authored image/audio derivative envelopes matching the pinned
host format also passed capture, semantic recall and context injection. The
installed formatter is private, so the harness supplies those text envelopes;
it does not claim to invoke raw OCR/ASR or verify the caption/transcript itself.
The stable host's exact empty-audio failure sentinel is excluded from capture;
separate user text and other successful media sections are retained. This is
tested through both the handler and the installed native hook runner.
The Hermes real loader/MemoryManager test also passes its native text-only vision
envelope as derived image evidence and recalls it in another session. Successful
Hermes voice preprocessing can lose media identity before the provider boundary;
a quoted string alone cannot prove audio provenance. Generated outgoing Hermes
text still does not prove channel delivery. Complete binary-attachment coverage
remain unverified. This was the September 9 checkpoint; the [September 10 follow-up](model-acceptance-2026-09-10.md) adds eight bounded generated-answer/tool-selection observations, with explicit remaining limits.

These are real native loader/hook/provider integration tests, not a running
Gateway/model conversation. Existing host installation/configuration is unchanged;
all temporary host data was removed and the same Gateway process remained ready.
No age-gate exception, runtime install, saved credential transfer or restart was
used. The earlier local SDK-only attempt remains an incomplete historical attempt.

The local unified compile/lint/test audit passed after the review fixes
(498 tests, one optional skip, in a clean environment without NLTK/Zeyrek). Python sdist and
wheel build succeeded in a separate pinned build environment.
On PR #34's initial head `9e7d2dc`, CI tests/lint and build succeeded. The security
audit failed on `nltk==3.10.3`, [PYSEC-2026-3740](https://github.com/pypa/advisory-database/blob/main/vulns/nltk/PYSEC-2026-3740.yaml).
A separate full dependency resolution and audit of the unchanged baseline
requirements reproduced the same finding. The [official NLTK advisory](https://github.com/nltk/nltk/security/advisories/GHSA-8mgp-746c-j5xp),
published August 12 and updated August 31, lists model-artifact path sandbox
bypasses through 3.10.3 and no patched release. NoldoMem reaches NLTK through
Zeyrek's sentence/word tokenizers; this path does not call the listed model
import/export APIs. That narrower reachability is not a clean dependency audit.
The advisory database has conflicting fixed-version metadata, so it is not used
to assert a fix. No audit exception, tokenizer substitution or dependency
protection override was applied. The follow-up removed the unused-on-the-main-path mandatory dependency after
an independent Turkish non-regression experiment. The [helper migration](turkish-helper-migration.md)
explicitly documents the breaking Python helper transition and required next
major release coordination. Fresh core/development and reranker-extra resolutions
contain neither Zeyrek nor NLTK; built wheel metadata and a wheel-only runtime
smoke confirm the same. All freshly resolved core/development pins passed a local
security audit without exceptions. The old NLTK/joblib CI ignores were removed;
pre-existing unrelated runner/tooling exceptions were not expanded. Final-head
CI is still a separate delivery gate. The configured Codex review identified
revision archival, administrative historical inference, manifest completeness,
partial-import lineage and admission-side-effect gaps; focused regressions
accompany the fixes. Final-head CI/review must be checked separately.
Deployment, merge and release are separate actions.

Public-artifact review used the tracked-only, filename-only secret scanner. It
reported 33 baseline files and 35 candidate files. The two additional files are
the synthetic Hermes harness and reranker isolation test, both with literal
`YOUR_API_KEY` placeholders. Existing
heuristic matches are not a clean-scan certificate. No private corpus, production
paths, account identities, credentials or operational logs were added.
