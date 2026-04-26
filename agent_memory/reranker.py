"""Cross-encoder reranker for high-precision top-K ordering.

Optional-by-design:
- If sentence-transformers is unavailable or model load fails,
  callers should gracefully fall back to lexical/no rerank.

Includes:
- model presets (fast|balanced|quality)
- lazy model load + startup prewarm support
- thread-safe inference lock
- TTL score cache
- API reranker (OpenRouter/Cohere) for higher quality without local GPU
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder reranker with TTL cache.

    Notes:
    - Scores are transformed with sigmoid to [0,1] for easier fusion.
    - ``model_name`` supports presets: fast|balanced|quality.
    """

    MODEL_PRESETS: Dict[str, str] = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "quality": "BAAI/bge-reranker-v2-m3",
    }

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "quality",
        top_k: int = 12,
        cache_ttl_sec: int = 600,
        cache_max: int = 5000,
        max_doc_chars: int = 1000,
        torch_threads: int = 4,
    ) -> None:
        self.enabled = enabled
        self.model_input = model_name
        self.model_name = self._resolve_model_name(model_name)
        self.top_k = max(1, int(top_k))
        self.cache_ttl_sec = max(30, int(cache_ttl_sec))
        self.cache_max = max(100, int(cache_max))
        self.max_doc_chars = max(200, int(max_doc_chars))
        self.torch_threads = max(1, int(torch_threads))

        self._model = None
        self._lock = threading.Lock()
        # Serialize inference to avoid CPU spikes on VPS under parallel recalls.
        self._infer_lock = threading.Lock()
        self._cache: Dict[str, Tuple[float, float]] = {}

    @classmethod
    def _resolve_model_name(cls, value: str) -> str:
        key = (value or "").strip().lower()
        return cls.MODEL_PRESETS.get(key, value)

    @property
    def available(self) -> bool:
        return bool(self.enabled)

    def warmup(self) -> bool:
        """Load model eagerly. Safe to call repeatedly."""
        return self._ensure_model()

    def _ensure_model(self) -> bool:
        if not self.available:
            return False

        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        try:
                            from sentence_transformers import CrossEncoder  # type: ignore
                        except Exception as exc:  # pragma: no cover - optional dependency
                            logger.warning("Cross-encoder dependency unavailable: %s", exc)
                            return False

                        torch_module: Optional[Any]
                        try:
                            import torch as torch_module  # type: ignore
                        except Exception:  # pragma: no cover - optional dependency
                            torch_module = None

                        if torch_module is not None:
                            try:
                                torch_module.set_num_threads(self.torch_threads)
                                torch_module.set_num_interop_threads(1)
                            except Exception:
                                pass
                        logger.info("Loading cross-encoder reranker model: %s", self.model_name)
                        self._model = CrossEncoder(self.model_name)
                        logger.info(
                            "Cross-encoder reranker ready (model=%s input=%s torch_threads=%s)",
                            self.model_name,
                            self.model_input,
                            self.torch_threads,
                        )
                    except Exception as exc:
                        logger.warning("Cross-encoder load failed: %s", exc)
                        self._model = None
                        return False
        return self._model is not None

    def _cache_key(self, query: str, doc_id: Optional[str], text: str) -> str:
        h = hashlib.sha1()
        h.update(query.encode("utf-8", errors="ignore"))
        h.update(b"\n")
        if doc_id:
            h.update(str(doc_id).encode("utf-8", errors="ignore"))
        else:
            h.update(text.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[float]:
        item = self._cache.get(key)
        if item is None:
            return None
        score, ts = item
        if time.time() - ts > self.cache_ttl_sec:
            self._cache.pop(key, None)
            return None
        return score

    def _cache_put(self, key: str, score: float) -> None:
        self._cache[key] = (score, time.time())
        if len(self._cache) > self.cache_max:
            # Evict oldest 20%
            n_drop = max(1, int(self.cache_max * 0.2))
            oldest = sorted(self._cache.items(), key=lambda kv: kv[1][1])[:n_drop]
            for k, _ in oldest:
                self._cache.pop(k, None)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def score(self, query: str, docs: List[str], doc_ids: Optional[List[str]] = None) -> List[float]:
        """Return normalized scores in [0,1] for docs.

        Empty list means reranker unavailable/failure.
        """
        if not docs:
            return []
        if not self._ensure_model():
            return []

        # Restrict to top_k for compute safety
        docs_cut = docs[: self.top_k]
        ids_cut = (doc_ids or [])[: len(docs_cut)]

        to_predict_idx: List[int] = []
        pairs: List[Tuple[str, str]] = []
        scores: List[Optional[float]] = [None] * len(docs_cut)

        for i, txt in enumerate(docs_cut):
            txt_norm = (txt or "")[: self.max_doc_chars]
            doc_id = ids_cut[i] if i < len(ids_cut) else None
            k = self._cache_key(query, doc_id=doc_id, text=txt_norm)
            cached = self._cache_get(k)
            if cached is not None:
                scores[i] = cached
            else:
                to_predict_idx.append(i)
                pairs.append((query, txt_norm))

        if pairs:
            try:
                with self._infer_lock:
                    raw = self._model.predict(pairs, show_progress_bar=False)  # type: ignore[attr-defined]
                raw_list = [float(x) for x in raw]
                for idx, val in zip(to_predict_idx, raw_list):
                    norm = self._sigmoid(val)
                    scores[idx] = norm
                    txt_norm = (docs_cut[idx] or "")[: self.max_doc_chars]
                    doc_id = ids_cut[idx] if idx < len(ids_cut) else None
                    self._cache_put(self._cache_key(query, doc_id=doc_id, text=txt_norm), norm)
            except Exception as exc:
                logger.warning("Cross-encoder scoring failed: %s", exc)
                return []

        # Fill any None defensively
        return [float(s or 0.0) for s in scores]


class APIReranker:
    """API-based reranker via OpenRouter/Cohere rerank endpoint.

    Same interface as CrossEncoderReranker; drop-in replacement.
    Scores are clamped to [0,1] (Cohere API returns scores in this range).
    """

    def __init__(
        self,
        enabled: bool = True,
        api_key: str = "",
        api_key_file: str = "",
        api_url: str = "https://openrouter.ai/api/v1/rerank",
        model: str = "cohere/rerank-4-pro",
        top_k: int = 20,
        cache_ttl_sec: int = 600,
        cache_max: int = 5000,
        max_doc_chars: int = 1000,
        timeout_sec: int = 10,
    ) -> None:
        self.enabled = enabled
        self.model = model
        self.api_url = api_url
        self.top_k = max(1, int(top_k))
        self.cache_ttl_sec = max(30, int(cache_ttl_sec))
        self.cache_max = max(100, int(cache_max))
        self.max_doc_chars = max(200, int(max_doc_chars))
        self.timeout_sec = max(3, int(timeout_sec))
        self._cache: Dict[str, Tuple[float, float]] = {}

        # Resolve API key from the reranker-specific config/env only. Avoid
        # OPENROUTER_API_KEY because some deployments reuse it for embedding.
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.environ.get("AGENT_MEMORY_RERANKER_API_KEY", "")
        if not self.api_key:
            key_file = api_key_file or os.environ.get("AGENT_MEMORY_RERANKER_API_KEY_FILE", "")
            fallback_files = [key_file, os.path.expanduser("~/.openrouter_key")]
            for path in [p for p in fallback_files if p]:
                try:
                    with open(path) as f:
                        self.api_key = f.read().strip()
                except OSError:
                    continue
                if self.api_key:
                    break

        if self.enabled and self.api_key:
            logger.info("API reranker configured: model=%s url=%s", self.model, self.api_url)
        elif self.enabled:
            logger.warning("API reranker enabled but no API key found")

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.api_key)

    def warmup(self) -> bool:
        """API reranker has no model to load; just verify key exists."""
        return self.available

    def _cache_key(self, query: str, doc_id: Optional[str], text: str) -> str:
        h = hashlib.sha1()
        h.update(query.encode("utf-8", errors="ignore"))
        h.update(b"\n")
        if doc_id:
            h.update(str(doc_id).encode("utf-8", errors="ignore"))
        else:
            h.update(text.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[float]:
        item = self._cache.get(key)
        if item is None:
            return None
        score, ts = item
        if time.time() - ts > self.cache_ttl_sec:
            self._cache.pop(key, None)
            return None
        return score

    def _cache_put(self, key: str, score: float) -> None:
        self._cache[key] = (score, time.time())
        if len(self._cache) > self.cache_max:
            n_drop = max(1, int(self.cache_max * 0.2))
            oldest = sorted(self._cache.items(), key=lambda kv: kv[1][1])[:n_drop]
            for k, _ in oldest:
                self._cache.pop(k, None)

    def score(self, query: str, docs: List[str], doc_ids: Optional[List[str]] = None) -> List[float]:
        """Return normalized scores in [0,1] for docs.

        Empty list means reranker unavailable/failure (triggers fallback).
        """
        if not docs:
            return []
        if not self.available:
            return []

        docs_cut = docs[: self.top_k]
        ids_cut = (doc_ids or [])[: len(docs_cut)]

        to_score_idx: List[int] = []
        docs_to_score: List[str] = []
        scores: List[Optional[float]] = [None] * len(docs_cut)

        for i, txt in enumerate(docs_cut):
            txt_norm = (txt or "")[: self.max_doc_chars]
            doc_id = ids_cut[i] if i < len(ids_cut) else None
            k = self._cache_key(query, doc_id=doc_id, text=txt_norm)
            cached = self._cache_get(k)
            if cached is not None:
                scores[i] = cached
            else:
                to_score_idx.append(i)
                docs_to_score.append(txt_norm)

        if docs_to_score:
            try:
                payload = json.dumps({
                    "model": self.model,
                    "query": query,
                    "documents": docs_to_score,
                }).encode("utf-8")
                req = urllib.request.Request(
                    self.api_url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))

                if "error" in body:
                    logger.warning("API reranker error: %s", body["error"])
                    return []

                # Map API response index to score
                api_scores: Dict[int, float] = {}
                for r in body.get("results", []):
                    idx = r.get("index")
                    s = r.get("relevance_score")
                    if idx is not None and s is not None:
                        api_scores[int(idx)] = max(0.0, min(1.0, float(s)))

                for local_pos, orig_idx in enumerate(to_score_idx):
                    norm = api_scores.get(local_pos, 0.0)
                    scores[orig_idx] = norm
                    txt_norm = (docs_cut[orig_idx] or "")[: self.max_doc_chars]
                    doc_id = ids_cut[orig_idx] if orig_idx < len(ids_cut) else None
                    self._cache_put(
                        self._cache_key(query, doc_id=doc_id, text=txt_norm), norm
                    )
            except Exception as exc:
                logger.warning("API reranker call failed: %s", exc)
                return []

        return [float(s or 0.0) for s in scores]
