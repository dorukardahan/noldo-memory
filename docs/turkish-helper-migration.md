# Turkish helper transition (unreleased)

NoldoMem's real capture/index/recall path uses embeddings and SQLite FTS5
trigrams. It does not call `lemmatize`, `lemmatize_tokens`, `normalize_text` or
`tokenize_for_search`; the API imports only `parse_temporal` from this module.
That does not prove external Python consumers never call the other helpers.

The next release removes Zeyrek and its NLTK dependency entirely. There is no
NLP extra, replacement model download, tokenizer service or audit waiver.
Successful package-provided morphological analysis is a removed capability of
the Python helpers. This is a breaking helper transition, not a claim that a
dependency-free stemmer is equivalent to a lemmatizer. It must be included in
the next major-version release notes and coordinated manifest version update;
this PR does not publish a release.

- `normalize_text(text)` now performs lexical lowercasing, folding, stopword
  removal and deduplication. Its default `use_lemma` changes to `False`.
- `tokenize_for_search(text)` performs lexical normalization without morphology.
- `lemmatize` and `lemmatize_tokens` retain their names and return types. Without
  a supplied analyzer they warn with `FutureWarning` and return/split unchanged
  text. This preserves the former unavailable-dependency fallback, not successful
  analysis. Do not treat that fallback as a lemma-quality test.
- Consumers that own a morphology implementation may pass `analyzer=` to these
  helpers. It must provide `lemmatize(token) -> [(token, [lemma, ...])]`. The
  legacy conversion selects the first lemma and strips `mak`/`mek`, so it can
  return a stem rather than a lemma. NoldoMem neither imports nor installs the
  caller's implementation. `normalize_text(..., use_lemma=True, analyzer=...)`
  explicitly uses that adapter.
- `parse_temporal`, HTTP API payloads, embeddings, persisted text and indexes
  keep their existing contracts. No migration or reindex is required.

## Evidence and limits

The [recorded comparison](turkish-dependency-results-2026-09-09.json) starts at
`009f7c3`, with six independent Turkish events, eight indirect queries, two
unrelated queries and a conflicting event owned by another synthetic agent.
Cases include suffixes, dotted/dotless uppercase letters, names, ASCII spelling,
a typo, a paraphrased event and an implied preference.

Seventeen real BGE-M3 embeddings were recorded once. The same snapshot then ran
through the real ASGI capture, temporary SQLite/vector/FTS storage and recall API
in three conditions: existing environment, that environment with imports of both
NLP packages blocked, and a clean candidate environment without either package.
All ranked records and semantic scores matched: 8/8 expected first results,
0/2 unrelated contexts at the previously evaluated 0.50 floor, and no cross-agent
records. Neither NLP package was imported by the baseline flow. This is a small
non-regression comparison, not general language or generated-answer accuracy.
No alternative NLP library was justified by this result.

The previous helper test accepted both `hatırla` and unchanged `hatırlıyorum`.
An isolated helper probe with installed Zeyrek 0.1.3/NLTK 3.9.4 returned unchanged
`hatırlıyorum`, `kitaplıkların` and `koşuyor`; that does not prove every installed
analyzer has always been ineffective. New tests separately assert the explicit
fallback/warning and caller-analyzer transformation contract.

Built wheel metadata contains neither NLP dependency, and the installed wheel
passed the same real capture/index/recall comparison. Lexical helpers also handle
Turkish dotted/dotless uppercase letters without introducing combining-dot token
fragments.

Fresh, wheel-only, hash-pinned core/development resolution contained 58 packages,
neither NLP package, and passed `pip-audit` without exceptions. The existing
reranker-extra resolution also contains neither package. The audit covers every
fully resolved pin; disabling a second resolver does not omit transitive
dependencies. No installed environment was uninstalled from or modified.
