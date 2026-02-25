"""Lightweight token counting and budget trimming.

Uses a simple word/character heuristic instead of tiktoken to avoid
external dependencies.  Approximation: ~4 chars per token (English/Turkish).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Average chars per token for multilingual text (conservative estimate).
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def trim_results_to_budget(
    results: List[Dict[str, Any]],
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Trim a list of recall results to fit within a token budget.

    Results are assumed to be in relevance order (most relevant first).
    Each result is included if it fits; once budget is exceeded, the rest
    are dropped.  Individual results are NOT truncated — they are included
    or excluded as whole items.
    """
    trimmed: List[Dict[str, Any]] = []
    used = 0
    for r in results:
        text = r.get("text", "")
        cost = estimate_tokens(text)
        if used + cost > max_tokens and trimmed:
            # Budget exceeded and we have at least one result
            break
        trimmed.append(r)
        used += cost
    return trimmed
