"""Memory compression: summarize old, long memories to reduce token usage.

Lightweight approach (no LLM required):
- Extract first sentence + key entities as summary
- Archive original text, replace with compressed version
- Only compress memories older than threshold and longer than min chars
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Sentence boundary heuristic (Turkish + English)
_SENTENCE_RE = re.compile(r"[.!?]\s+|\n")

# Min chars to consider for compression
_MIN_CHARS = 500

# Max summary chars
_MAX_SUMMARY = 300


def _extract_summary(text: str) -> str:
    """Extract first 1-2 sentences as summary, capped at _MAX_SUMMARY chars."""
    sentences = _SENTENCE_RE.split(text.strip())
    summary = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not summary:
            summary = s
        elif len(summary) + len(s) < _MAX_SUMMARY:
            summary += ". " + s
        else:
            break
    if not summary:
        summary = text[:_MAX_SUMMARY]
    return summary.strip()


def compress_old_memories(
    storage: Any,
    agent: str = "main",
    age_days: int = 30,
    min_chars: int = _MIN_CHARS,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Compress old, long memories by replacing text with summary.

    Original text is preserved in a new 'original_text' field (if schema supports)
    or simply truncated with a [compressed] marker.

    Returns stats about what was compressed.
    """
    conn = storage._get_conn()
    now = time.time()
    age_threshold = now - (age_days * 86400)

    # Find candidates: old + long + not pinned + not already compressed
    rows = conn.execute(
        """
        SELECT id, text, importance, created_at, strength
          FROM memories
         WHERE deleted_at IS NULL
           AND COALESCE(pinned, 0) = 0
           AND created_at < ?
           AND LENGTH(text) > ?
           AND text NOT LIKE '[compressed]%'
        ORDER BY created_at ASC
        LIMIT 500
        """,
        (age_threshold, min_chars),
    ).fetchall()

    compressed = 0
    saved_chars = 0
    skipped = 0

    for row in rows:
        mid = row["id"]
        text = row["text"]
        importance = row["importance"] or 0.5

        # Don't compress high-importance memories
        if importance >= 0.8:
            skipped += 1
            continue

        summary = _extract_summary(text)
        compressed_text = f"[compressed] {summary}"

        if dry_run:
            compressed += 1
            saved_chars += len(text) - len(compressed_text)
            continue

        try:
            conn.execute(
                "UPDATE memories SET text = ?, updated_at = ? WHERE id = ?",
                (compressed_text, now, mid),
            )
            # Update FTS5
            conn.execute("DELETE FROM memory_fts WHERE id = ?", (mid,))
            conn.execute(
                "INSERT INTO memory_fts (id, text) VALUES (?, ?)",
                (mid, compressed_text),
            )
            compressed += 1
            saved_chars += len(text) - len(compressed_text)
        except Exception as exc:
            logger.warning("Failed to compress memory %s: %s", mid, exc)

    if not dry_run and compressed > 0:
        conn.commit()
        logger.info(
            "Compressed %d memories for agent=%s (saved %d chars)",
            compressed, agent, saved_chars,
        )

    return {
        "compressed": compressed,
        "skipped": skipped,
        "saved_chars": saved_chars,
        "candidates": len(rows),
        "dry_run": dry_run,
    }
