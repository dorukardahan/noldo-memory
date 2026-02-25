"""Hybrid search with Reciprocal Rank Fusion (RRF).

Five-layer search:
1. **Semantic**   — sqlite-vec cosine similarity (weight 0.50)
2. **Keyword**    — FTS5 BM25 ranking (weight 0.25)
3. **Recency**    — Exponential decay (weight 0.10)
4. **Strength**   — Ebbinghaus retention score (weight 0.07)
5. **Importance** — write-time message importance (weight 0.25)

Results from each layer are fused via RRF:
    ``score = Σ 1 / (k + rank_i)``  where k = 60

Adapted from Mahmory's ``hybrid_search.py`` — rewritten for sqlite-vec + FTS5.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import Config, load_config
from .reranker import CrossEncoderReranker
from .storage import MemoryStorage
from .triggers import get_confidence_tier

logger = logging.getLogger(__name__)

# Background two-pass quality rerank control (process-wide)
_BG_TWO_PASS_LOCK: Optional[asyncio.Lock] = None
_BG_TWO_PASS_PENDING: Set[str] = set()


def _get_bg_two_pass_lock() -> asyncio.Lock:
    global _BG_TWO_PASS_LOCK
    if _BG_TWO_PASS_LOCK is None:
        _BG_TWO_PASS_LOCK = asyncio.Lock()
    return _BG_TWO_PASS_LOCK


def normalize_query(text: str) -> str:
    """Normalize query text for better cache hits.

    - lowercase, strip
    - collapse whitespace
    - remove trailing punctuation
    """
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[?!.,;:]+$", "", t)
    return t.strip()


def _tokenize_for_rerank(text: str) -> set[str]:
    """Very light lexical tokenizer for reranking.

    Keeps alnum tokens with minimum length 3 to reduce noise.
    """
    toks = re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", (text or "").lower(), re.UNICODE)
    return {t for t in toks if len(t) >= 3}


def _lexical_overlap(query: str, text: str) -> float:
    """Query coverage score in [0, 1] for lightweight reranking."""
    q = _tokenize_for_rerank(query)
    if not q:
        return 0.0
    d = _tokenize_for_rerank((text or "")[:4000])
    if not d:
        return 0.0
    return len(q & d) / len(q)


@dataclass
class SearchWeights:
    """Configurable weights for each search layer.

    Rebalanced for 5-layer RRF with explicit importance lane.
    """
    semantic: float = 0.50
    keyword: float = 0.25
    recency: float = 0.10
    strength: float = 0.07
    importance: float = 0.08


@dataclass
class SearchResult:
    """A single search result with scores from each layer."""
    id: str
    text: str
    category: str
    importance: float
    created_at: float
    score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    recency_score: float = 0.0
    strength_score: float = 0.0
    importance_score: float = 0.0
    rerank_score: float = 0.0
    confidence_tier: str = "LOW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "importance": self.importance,
            "created_at": self.created_at,
            "score": round(self.score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "recency_score": round(self.recency_score, 4),
            "strength_score": round(self.strength_score, 4),
            "importance_score": round(self.importance_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "confidence_tier": self.confidence_tier,
        }


def _recency_score(created_at: float, decay_rate: float = 0.01) -> float:
    """Exponential decay: ``exp(-decay_rate * days_old)``."""
    days_old = max(0.0, (time.time() - created_at) / 86400.0)
    return math.exp(-decay_rate * days_old)


def _strength_score(last_accessed_at: float, strength: float) -> float:
    """Ebbinghaus retention score: exp(-days_since_access / strength)."""
    try:
        s = float(strength or 1.0)
        if s <= 0:
            s = 1.0
    except Exception:
        s = 1.0

    days = max(0.0, (time.time() - float(last_accessed_at)) / 86400.0)
    return math.exp(-days / s)


def _mmr_diversify(
    results: List["SearchResult"],
    limit: int,
    lambda_param: float = 0.7,
) -> List["SearchResult"]:
    """Maximal Marginal Relevance: greedily select results balancing
    relevance (score) and diversity (low text overlap with already selected).

    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity.
    """
    if len(results) <= 1:
        return results

    selected: List["SearchResult"] = [results[0]]
    remaining = list(results[1:])

    while remaining and len(selected) < limit:
        best_idx = 0
        best_mmr = -1.0
        for i, cand in enumerate(remaining):
            relevance = cand.score
            # Max similarity to any already-selected result (text overlap)
            max_sim = max(
                _lexical_overlap(cand.text, sel.text)
                for sel in selected
            )
            mmr = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


def _rrf_fuse(
    ranked_lists: List[List[str]],
    weights: List[float],
    k: int = 60,
) -> Dict[str, float]:
    """Reciprocal Rank Fusion across multiple ranked ID lists.

    ``score(d) = Σ weight_i / (k + rank_i(d))``
    """
    scores: Dict[str, float] = {}
    for ids, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


class HybridSearch:
    """Five-layer hybrid search engine with RRF fusion."""

    def __init__(
        self,
        storage: MemoryStorage,
        embedder: Optional[Any] = None,  # OpenRouterEmbeddings
        weights: Optional[SearchWeights] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_weight: float = 0.20,
        bg_reranker: Optional[CrossEncoderReranker] = None,
        bg_two_pass_enabled: bool = False,
        bg_rerank_weight: float = 0.35,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.weights = weights or SearchWeights()
        self.reranker = reranker
        self.rerank_weight = max(0.0, min(1.0, float(rerank_weight)))
        self.bg_reranker = bg_reranker
        self.bg_two_pass_enabled = bool(bg_two_pass_enabled and bg_reranker is not None)
        self.bg_rerank_weight = max(0.0, min(1.0, float(bg_rerank_weight)))
        # Graceful degradation: track if last search used all layers
        self.last_search_degraded: bool = False
        self.last_search_mode: str = "full"  # full | keyword_only | cache_hit

    def _schedule_background_quality_rerank(
        self,
        *,
        q_norm: str,
        cache_query_norm: Optional[str] = None,
        limit: int,
        min_score: float,
        agent: str,
        seed_results: List[SearchResult],
    ) -> None:
        """Fire-and-forget second pass (quality model) to refresh cache.

        Designed to be VPS-safe:
        - single in-process worker lock
        - drops new tasks when worker is busy
        - de-duplicates by cache key
        """
        if not self.bg_two_pass_enabled or self.bg_reranker is None:
            return
        if len(seed_results) < 2:
            return

        cache_query_norm = cache_query_norm or q_norm
        cache_key = f"{agent}|{cache_query_norm}|{limit}|{min_score:.4f}"
        if cache_key in _BG_TWO_PASS_PENDING:
            return

        # Copy results to avoid mutating the response path.
        snapshot = [SearchResult(**r.__dict__) for r in seed_results]

        try:
            asyncio.create_task(
                self._run_background_quality_rerank(
                    cache_key=cache_key,
                    q_norm=q_norm,
                    cache_query_norm=cache_query_norm,
                    limit=limit,
                    min_score=min_score,
                    agent=agent,
                    seed_results=snapshot,
                )
            )
        except RuntimeError:
            # No running loop (shouldn't happen in API path)
            return

    async def _run_background_quality_rerank(
        self,
        *,
        cache_key: str,
        q_norm: str,
        cache_query_norm: str,
        limit: int,
        min_score: float,
        agent: str,
        seed_results: List[SearchResult],
    ) -> None:
        if self.bg_reranker is None:
            return

        lock = _get_bg_two_pass_lock()
        if lock.locked():
            # CPU-safe mode: skip instead of queueing a backlog.
            return

        _BG_TWO_PASS_PENDING.add(cache_key)
        t0 = time.time()
        try:
            async with lock:
                cands = [SearchResult(**r.__dict__) for r in seed_results]
                top_n = min(len(cands), max(1, int(self.bg_reranker.top_k)))
                if top_n < 2:
                    return

                docs = [r.text for r in cands[:top_n]]
                ids = [r.id for r in cands[:top_n]]

                ce_scores = await asyncio.to_thread(
                    self.bg_reranker.score,
                    q_norm,
                    docs,
                    ids,
                )
                if not ce_scores or len(ce_scores) != top_n:
                    return

                rerank_map: Dict[str, float] = {}
                for i, s in enumerate(ce_scores):
                    rerank_map[cands[i].id] = float(s)

                base_ranked = [r.id for r in cands]
                rerank_ranked = [
                    mid for mid, _ in sorted(rerank_map.items(), key=lambda x: x[1], reverse=True)
                ]

                rerank_scores = _rrf_fuse(
                    [base_ranked, rerank_ranked],
                    [1.0 - self.bg_rerank_weight, self.bg_rerank_weight],
                    k=60,
                )
                cands.sort(key=lambda rr: rerank_scores.get(rr.id, 0.0), reverse=True)
                for rr in cands:
                    rr.score = rerank_scores.get(rr.id, rr.score)
                    if rr.id in rerank_map:
                        rr.rerank_score = rerank_map[rr.id]

                if len(cands) > limit:
                    cands = cands[:limit]

                results_json = json.dumps([r.to_dict() for r in cands])
                self.storage.cache_search_result(
                    query_norm=cache_query_norm,
                    limit_val=limit,
                    min_score=min_score,
                    agent=agent,
                    results_json=results_json,
                )

                logger.debug(
                    "Two-pass quality cache refresh done key=%s top_n=%d elapsed=%.2fs",
                    cache_key,
                    top_n,
                    time.time() - t0,
                )
        except Exception as exc:
            logger.debug("Two-pass quality rerank skipped: %s", exc)
        finally:
            _BG_TWO_PASS_PENDING.discard(cache_key)

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        use_semantic: bool = True,
        use_keyword: bool = True,
        use_recency: bool = True,
        agent: str = "main",
        time_range: Optional[Tuple[float, float]] = None,
        namespace: Optional[str] = None,
    ) -> List[SearchResult]:
        """Run hybrid search and return fused, ranked results.

        Graceful degradation: if the embedding API is unavailable,
        lexical + non-vector layers (keyword/recency/strength/importance)
        are still used.
        """
        # Reset degradation flags
        self.last_search_degraded = False
        self.last_search_mode = "full"

        # 1. Query Normalization + Result Cache Check
        q_norm = normalize_query(query)
        # Skip cache for temporal queries — time_range changes daily
        if time_range is None:
            cached_json = self.storage.get_cached_search_result(
                query_norm=q_norm, limit_val=limit, min_score=min_score, agent=agent
            )
            if cached_json:
                try:
                    cached_data = json.loads(cached_json)
                    self.last_search_mode = "cache_hit"
                    return [SearchResult(**r) for r in cached_data]
                except Exception as exc:
                    logger.warning("Failed to parse cached search results: %s", exc)

        candidate_limit = max(limit * 4, 20)

        # Collect candidates from each layer ---------------------------------
        # Semantic + keyword run in parallel when both are enabled.
        semantic_ids: List[str] = []
        keyword_ids: List[str] = []
        all_candidates: Dict[str, Dict[str, Any]] = {}

        async def _semantic_search() -> List[Dict[str, Any]]:
            """Embed query + vector search (async)."""
            query_vec = await self.embedder.embed(q_norm)
            return self.storage.search_vectors(
                query_vec, limit=candidate_limit, min_score=0.0,
                namespace=namespace,
            )

        async def _keyword_search() -> List[Dict[str, Any]]:
            """FTS5 BM25 search (sync, runs in event loop — fast enough)."""
            return self.storage.search_text(q_norm, candidate_limit, namespace)

        sem_results: List[Dict[str, Any]] = []
        kw_results: List[Dict[str, Any]] = []

        # Run both layers in parallel
        if use_semantic and self.embedder is not None and use_keyword:
            sem_task = asyncio.create_task(_semantic_search())
            kw_task = asyncio.create_task(_keyword_search())

            try:
                sem_results = await sem_task
            except Exception as exc:
                logger.warning("Semantic search failed (BM25 fallback): %s", exc)
                self.last_search_degraded = True
                self.last_search_mode = "keyword_only"

            try:
                kw_results = await kw_task
            except Exception as exc:
                logger.warning("Keyword search failed: %s", exc)

        elif use_semantic and self.embedder is not None:
            try:
                sem_results = await _semantic_search()
            except Exception as exc:
                logger.warning("Semantic search failed (BM25 fallback): %s", exc)
                self.last_search_degraded = True
                self.last_search_mode = "keyword_only"

        elif use_keyword:
            try:
                kw_results = await _keyword_search()
            except Exception as exc:
                logger.warning("Keyword search failed: %s", exc)

        # Merge semantic results
        for r in sem_results:
            mid = r["id"]
            semantic_ids.append(mid)
            r["_sem_score"] = r.get("score", 0.0)
            all_candidates[mid] = r

        # Merge keyword results
        for r in kw_results:
            mid = r["id"]
            keyword_ids.append(mid)
            if mid not in all_candidates:
                all_candidates[mid] = r

        if not all_candidates:
            return []

        # Temporal filter: remove candidates outside time_range
        if time_range is not None:
            t_start, t_end = time_range
            before = len(all_candidates)
            all_candidates = {
                mid: c for mid, c in all_candidates.items()
                if t_start <= c.get("created_at", 0) <= t_end
            }
            # Also filter ranked ID lists
            semantic_ids = [mid for mid in semantic_ids if mid in all_candidates]
            keyword_ids = [mid for mid in keyword_ids if mid in all_candidates]
            if before > 0 and len(all_candidates) == 0:
                logger.debug("Temporal filter removed all %d candidates", before)
                return []

        # Layer 3 — Recency: rank all candidates by created_at
        recency_ranked = sorted(
            all_candidates.keys(),
            key=lambda mid: all_candidates[mid].get("created_at", 0),
            reverse=True,
        )

        # Layer 4 — Strength: rank by Ebbinghaus retention
        strength_scored: List[tuple[str, float]] = []
        for mid, cand in all_candidates.items():
            last_acc = cand.get("last_accessed_at") or cand.get("created_at") or 0.0
            st = cand.get("strength") or 1.0
            strength_scored.append((mid, _strength_score(float(last_acc), float(st))))
        strength_ranked = [mid for mid, _ in sorted(strength_scored, key=lambda x: x[1], reverse=True)]
        strength_map = {mid: sc for mid, sc in strength_scored}

        # Layer 5 — Importance: rank by write-time importance score
        importance_scored: List[tuple[str, float]] = []
        for mid, cand in all_candidates.items():
            try:
                imp = float(cand.get("importance", 0.5) or 0.5)
            except Exception:
                imp = 0.5
            # Clamp to [0, 1]
            imp = max(0.0, min(1.0, imp))
            importance_scored.append((mid, imp))
        importance_ranked = [mid for mid, _ in sorted(importance_scored, key=lambda x: x[1], reverse=True)]
        importance_map = {mid: sc for mid, sc in importance_scored}

        # Memory type bonus layer: prioritize factual and preference memories.
        memory_type_bonus: Dict[str, float] = {}
        for mid, cand in all_candidates.items():
            memory_type = str(cand.get("memory_type", "") or "").strip().lower()
            if memory_type in {"fact", "preference"}:
                memory_type_bonus[mid] = 0.1

        # RRF fusion ----------------------------------------------------------
        ranked_lists: List[List[str]] = []
        weights_list: List[float] = []

        if semantic_ids:
            ranked_lists.append(semantic_ids)
            weights_list.append(self.weights.semantic)
        if keyword_ids:
            ranked_lists.append(keyword_ids)
            weights_list.append(self.weights.keyword)
        if use_recency and recency_ranked:
            ranked_lists.append(recency_ranked)
            weights_list.append(self.weights.recency)
        if strength_ranked:
            ranked_lists.append(strength_ranked)
            weights_list.append(self.weights.strength)
        if importance_ranked:
            ranked_lists.append(importance_ranked)
            weights_list.append(self.weights.importance)

        if not ranked_lists:
            return []

        rrf_scores = _rrf_fuse(ranked_lists, weights_list)
        for mid, bonus in memory_type_bonus.items():
            if mid in rrf_scores:
                rrf_scores[mid] += bonus

        # Build SearchResult objects ------------------------------------------
        pre_limit = limit
        if self.reranker is not None:
            # Friend-repo style: gather a wider candidate pool before reranking.
            pre_limit = max(limit, min(candidate_limit, max(self.reranker.top_k, 15)))

        results: List[SearchResult] = []
        for mid, rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            if len(results) >= pre_limit:
                break
            cand = all_candidates.get(mid)
            if cand is None:
                continue

            score = rrf
            if score < min_score:
                continue

            sr = SearchResult(
                id=mid,
                text=cand.get("text", ""),
                category=cand.get("category", "other"),
                importance=cand.get("importance", 0.5),
                created_at=cand.get("created_at", 0.0),
                score=score,
                semantic_score=cand.get("_sem_score", 0.0),
                keyword_score=0.0,  # BM25 rank-based, not a similarity
                recency_score=_recency_score(cand.get("created_at", 0.0)),
                strength_score=float(strength_map.get(mid, 0.0)),
                importance_score=float(importance_map.get(mid, 0.0)),
                confidence_tier=get_confidence_tier(score / 0.02),  # normalise to ~1
            )
            results.append(sr)

        # Cross-encoder reranker on top-N candidates.
        # Fallback: lightweight lexical overlap if cross-encoder is unavailable.
        # CPU-safe gating: rerank only when ranking is ambiguous.
        if len(results) > 1:
            try:
                query_token_count = len(_tokenize_for_rerank(q_norm))

                # Adaptive reranker gating: skip when top results are clearly
                # separated (score spread > threshold).  This saves ~500ms+ on
                # unambiguous queries while preserving quality for close calls.
                _scores = [r.score for r in results[:5]]
                _spread = (_scores[0] - _scores[-1]) if len(_scores) >= 2 else 0.0
                _RERANK_SPREAD_THRESHOLD = 0.005  # tune: higher = rerank less often
                need_rerank = (
                    query_token_count >= 2
                    and _spread < _RERANK_SPREAD_THRESHOLD
                )

                if need_rerank:
                    base_ranked = [r.id for r in results]

                    # Candidate set for reranking
                    top_n = len(results)
                    if self.reranker is not None:
                        top_n = min(len(results), self.reranker.top_k)
                    cands = results[:top_n]

                    rerank_map: Dict[str, float] = {}
                    rerank_ranked: List[str] = []
                    used_cross_encoder = False

                    if self.reranker is not None:
                        ce_scores = await asyncio.to_thread(
                            self.reranker.score,
                            q_norm,
                            [r.text for r in cands],
                            [r.id for r in cands],
                        )
                        if ce_scores and len(ce_scores) == len(cands):
                            used_cross_encoder = True
                            for r, s in zip(cands, ce_scores):
                                rerank_map[r.id] = float(s)
                            rerank_ranked = [
                                mid for mid, _ in sorted(
                                    rerank_map.items(), key=lambda x: x[1], reverse=True
                                )
                            ]

                    if not rerank_ranked:
                        # Fallback lexical overlap signal
                        overlap_map = {r.id: _lexical_overlap(q_norm, r.text) for r in cands}
                        rerank_map = {k: float(v) for k, v in overlap_map.items()}
                        rerank_ranked = [
                            mid for mid, _ in sorted(
                                overlap_map.items(), key=lambda x: x[1], reverse=True
                            )
                        ]

                    # Combine original ranking + reranker ranking via RRF
                    w_rerank = self.rerank_weight
                    rerank_scores = _rrf_fuse(
                        [base_ranked, rerank_ranked],
                        [1.0 - w_rerank, w_rerank],
                        k=60,
                    )
                    results.sort(key=lambda rr: rerank_scores.get(rr.id, 0.0), reverse=True)
                    for rr in results:
                        rr.score = rerank_scores.get(rr.id, rr.score)
                        rr.rerank_score = float(rerank_map.get(rr.id, 0.0))

                    if used_cross_encoder:
                        logger.debug("Cross-encoder reranker applied to top %d", top_n)
                else:
                    logger.debug("Reranker skipped (query too short): tokens=%d", query_token_count)
            except Exception as exc:
                logger.debug("Reranker skipped: %s", exc)

        # Two-pass refresh: run heavy quality reranker in background and update cache.
        if time_range is None:
            self._schedule_background_quality_rerank(
                q_norm=q_norm,
                limit=limit,
                min_score=min_score,
                agent=agent,
                seed_results=results,
            )

        # MMR diversity pass: remove near-duplicate results.
        # Uses text overlap as proxy for similarity (avoids extra embedding calls).
        if len(results) > 2:
            results = _mmr_diversify(results, limit, lambda_param=0.7)

        # Return only requested count after optional reranking over pre-limit pool.
        if len(results) > limit:
            results = results[:limit]

        # Spaced repetition: boost strength on top hits
        for r in results[:3]:
            try:
                self.storage.boost_strength(r.id)
            except Exception:
                pass

        # 3. Store Results in Cache (skip for temporal queries)
        if time_range is None:
            try:
                results_json = json.dumps([r.to_dict() for r in results])
                self.storage.cache_search_result(
                    query_norm=q_norm,
                    limit_val=limit,
                    min_score=min_score,
                    agent=agent,
                    results_json=results_json,
                )
            except Exception as exc:
                logger.warning("Failed to cache search results: %s", exc)

        return results
