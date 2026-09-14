# NoldoMem 2.0.0 upgrade notes

Release date: 2026-09-14. Publication does not update existing installations.
The previous release is
[v1.27.16](https://github.com/dorukardahan/noldo-memory/releases/tag/v1.27.16).

## What changes

Version 2.0.0 brings scoped automatic capture/recall, explicit validity history,
qualified media evidence and source-bound forgetting from
[PR #34](https://github.com/dorukardahan/noldo-memory/pull/34).
The [capability guide](current-capabilities.md) separates real native model
tests from model-free regressions and unresolved media/delivery coverage.
Existing supported integrations do not require unpublished host patches.

The major version reflects a **breaking Python helper change**: package-provided
Turkish morphology is removed together with Zeyrek/NLTK. There is no NLP extra.
Consumers needing analysis must supply their own analyzer under the documented
[helper contract](turkish-helper-migration.md). HTTP search and capture retain
their interfaces. No embedding-model change or reindex is required for this
helper transition.

## Upgrade and recovery sequence

Publication and production rollout are separate operations. Before either,
verify the final commit's CI and package versions. The following is a rollout
plan, not evidence that any production step has run:

1. Inventory the installed API, adapters, background/import writers and each
   agent database through authorized status surfaces. Pin the target commit and
   retain the previous code and dependency environment. Do not read or transfer
   runtime credentials. Prepare a separate target environment rather than
   uninstalling packages from the running one.
2. Before changing live data, explicitly authorize a maintenance window and
   quiesce all memory writers, including host capture and import jobs. Account
   for queued writes. Preserve every agent DB with the SQLite Backup API and
   verify recovery on protected copies; memory-only JSON export is insufficient.
   Do not put backups or private records in the repository or CI.
3. Keep production traffic closed while starting the candidate API. Normal
   initialization adds the schema described in [source forgetting](forgetting-sources.md).
   Existing records remain; missing historical source links are not reconstructed.
   Query `/openapi.json` on the candidate API endpoint and require
   `info.version == "2.0.0"`; this reports the serving application's package
   version, not a CLI or checkout version. Confirm the pinned commit through
   the deployment receipt as well: a version string does not identify a commit.
   Then validate health and database integrity without exporting memory text.
   Exercise adapters with a separate synthetic scope and DB before reopening.
4. If validation fails **before production writes resume**, stop the candidate
   and restore the matched pre-upgrade code/environment and complete DB snapshot.
   Never run an old writer against the upgraded DB: it does not enforce source
   deletion markers. Verify recovery before reopening traffic.
5. After production writes resume, the old snapshot is no longer a lossless
   rollback: it can discard new records and undo subsequent forgetting. Stop
   writes and preserve current state if recovery becomes necessary. Prefer a
   forward fix retaining revision and deletion policy; do not automatically
   restore the old snapshot or fall back to an old writer.

The deployment mechanism must enforce this sequence for the actual installation.
A code-only updater that excludes dependency/schema changes or merely checks out
old code is insufficient for this upgrade. Do not weaken that updater's guards.

## Publication checklist

- Keep all seven runtime versions listed in `CONTRIBUTING.md` at `2.0.0`.
- Build and inspect the sdist and wheel; verify neither declares Zeyrek/NLTK.
- Complete the release PR's exact-head CI and direct review.
- After approval, merge the publication-notes PR, verify its resulting commit
  includes the finalized changelog date, then
  tag and publish manually using the actual publication date in the changelog.
- Deploy only through a separately authorized, verified rollout and recovery plan.
