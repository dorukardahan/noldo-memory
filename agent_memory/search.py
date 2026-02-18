"""Hybrid search with Reciprocal Rank Fusion (RRF).

Four-layer search:
1. **Semantic** — sqlite-vec cosine similarity (weight 0.40)
2. **Keyword**  — FTS5 BM25 ranking (weight 0.25)
3. **Recency**  — Exponential decay (weight 0.15)
4. **Strength** — Ebbinghaus retention score (weight 0.20)

Results from each layer are fused via RRF:
    ``score = Σ 1 / (k + rank_i)``  where k = 60

Adapted from Mahmory's ``hybrid_search.py`` — rewritten for sqlite-vec + FTS5.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, load_config
from .storage import MemoryStorage
from .triggers import get_confidence_tier

logger = logging.getLogger(__name__)


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


@dataclass
class SearchWeights:
    """Configurable weights for each search layer.

    Rebalanced 2026-02-17 [S2]: semantic-first RRF. Previous 35% time-based
    weight (recency+strength) drowned semantic relevance → 3.2/10 search quality.
    """
    semantic: float = 0.55   # was 0.40 — semantic match is primary signal
    keyword: float = 0.25    # unchanged — BM25 is second-best
    recency: float = 0.10    # was 0.15 — still a factor but less dominant
    strength: float = 0.10   # was 0.20 — tiebreaker only


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
    """Four-layer hybrid search engine with RRF fusion."""

    def __init__(
        self,
        storage: MemoryStorage,
        embedder: Optional[Any] = None,  # OpenRouterEmbeddings
        weights: Optional[SearchWeights] = None,
    ) -> None:
        self.storage = storage
        self.embedder = embedder
        self.weights = weights or SearchWeights()

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
    ) -> List[SearchResult]:
        """Run hybrid search and return fused, ranked results.

        Graceful degradation: if the embedding API is unavailable, only
        BM25 + recency layers are used.
        """
        # 1. Query Normalization + Result Cache Check
        q_norm = normalize_query(query)
        cached_json = self.storage.get_cached_search_result(
            query_norm=q_norm, limit_val=limit, min_score=min_score, agent=agent
        )
        if cached_json:
            try:
                cached_data = json.loads(cached_json)
                return [SearchResult(**r) for r in cached_data]
            except Exception as exc:
                logger.warning("Failed to parse cached search results: %s", exc)

        candidate_limit = max(limit * 4, 20)

        # Collect candidates from each layer ---------------------------------
        semantic_ids: List[str] = []
        keyword_ids: List[str] = []
        all_candidates: Dict[str, Dict[str, Any]] = {}

        # Layer 1 — Semantic (sqlite-vec)
        if use_semantic and self.embedder is not None:
            try:
                query_vec = await self.embedder.embed(q_norm)
                sem_results = self.storage.search_vectors(
                    query_vec, limit=candidate_limit, min_score=0.0
                )
                for r in sem_results:
                    mid = r["id"]
                    semantic_ids.append(mid)
                    r["_sem_score"] = r.get("score", 0.0)
                    all_candidates[mid] = r
            except Exception as exc:
                logger.warning("Semantic search failed (BM25 fallback): %s", exc)

        # Layer 2 — Keyword (FTS5 BM25)
        if use_keyword:
            try:
                kw_results = self.storage.search_text(q_norm, limit=candidate_limit)
                for r in kw_results:
                    mid = r["id"]
                    keyword_ids.append(mid)
                    if mid not in all_candidates:
                        all_candidates[mid] = r
            except Exception as exc:
                logger.warning("Keyword search failed: %s", exc)

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

        if not ranked_lists:
            return []

        rrf_scores = _rrf_fuse(ranked_lists, weights_list)

        # Build SearchResult objects ------------------------------------------
        results: List[SearchResult] = []
        for mid, rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            if len(results) >= limit:
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
                confidence_tier=get_confidence_tier(score / 0.02),  # normalise to ~1
            )
            results.append(sr)

        # Spaced repetition: boost strength on top hits
        for r in results[:3]:
            try:
                self.storage.boost_strength(r.id)
            except Exception:
                pass

        # 3. Store Results in Cache
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
